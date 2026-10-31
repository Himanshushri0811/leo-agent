/**
 * LEO Voice Mode — Siri-Style Continuous Voice Conversation Module
 *
 * Architecture:
 *  - Uses Web Speech API (SpeechRecognition) with continuous=true + interimResults
 *    so the microphone stream STAYS OPEN throughout the session.
 *  - VAD is handled natively by the browser's SpeechRecognition engine.
 *  - A state machine (IDLE → LISTENING → PROCESSING → SPEAKING → INTERRUPTED → ERROR)
 *    controls the orb animation, labels, and audio playback.
 *  - Barge-in: onspeechstart fires while LEO is speaking → stopCurrentAudio() called,
 *    server barge-in POST sent, then transcript is collected from onresult.
 *  - Reuses the existing live-stream TTS pipeline (/api/v1/demo/calls/{id}/live-stream).
 *  - Microphone stream is held open for the entire session, released on exit.
 *
 * Dependencies (from parent app.js scope via window.__LEO_VOICE_DEPS__):
 *  - state, el (dashboard state & elements)
 *  - stopCurrentAudio, playLiveStream, getOrCreateAudioContext
 *  - triggerAudioActivity
 */

(function () {
  'use strict';

  // ═══════════════════════════════════════════════════════════════════════
  // Voice State Machine
  // ═══════════════════════════════════════════════════════════════════════
  const VS = Object.freeze({
    IDLE:        'idle',
    LISTENING:   'listening',
    PROCESSING:  'processing',
    SPEAKING:    'speaking',
    INTERRUPTED: 'interrupted',
    ERROR:       'error',
  });

  const voiceState = {
    current:          VS.IDLE,
    sessionActive:    false,
    micStream:        null,       // MediaStream (kept alive)
    recognition:      null,       // SpeechRecognition instance
    pendingTranscript:'',         // Accumulated interim text
    activePlayToken:  null,       // Token for current TTS playback
    bargeInFired:     false,      // Debounce guard for barge-in
    bargeInCooldown:  false,      // Prevent rapid repeated barge-ins
    speakingEndTimer: null,       // Timer to return to LISTENING after TTS ends
    restartPending:   false,      // Whether recognition needs a restart
    lastUserText:     '',         // Last confirmed user utterance
    lastLeoText:      '',         // Last LEO response text
  };

  // ═══════════════════════════════════════════════════════════════════════
  // DOM refs
  // ═══════════════════════════════════════════════════════════════════════
  const vmEl = {
    overlay:       document.getElementById('voice-mode-overlay'),
    closeBtn:      document.getElementById('vm-close-btn'),
    stateLabel:    document.getElementById('vm-state-label'),
    stateSubtitle: document.getElementById('vm-state-subtitle'),
    orbContainer:  document.querySelector('.vm-orb-container'),
    orb:           document.getElementById('vm-orb'),
    orbBars:       document.getElementById('vm-orb-bars'),
    transcriptArea:document.getElementById('vm-transcript-area'),
    langSelect:    document.getElementById('vm-lang-select'),
    retryBtn:      document.getElementById('vm-retry-btn'),
    permPrompt:    document.getElementById('vm-permission-prompt'),
    grantMicBtn:   document.getElementById('vm-grant-mic-btn'),
    mainUI:        document.getElementById('vm-main-ui'),
    activateBtn:   document.getElementById('btn-voice-mode'),
  };

  // ═══════════════════════════════════════════════════════════════════════
  // State → UI map
  // ═══════════════════════════════════════════════════════════════════════
  const STATE_META = {
    [VS.IDLE]:        { label: 'Ready',                subtitle: 'Click Voice Mode to start a conversation.' },
    [VS.LISTENING]:   { label: 'Listening…',           subtitle: 'Speak now — LEO is ready for your question.' },
    [VS.PROCESSING]:  { label: 'Thinking…',            subtitle: 'LEO is generating a response.' },
    [VS.SPEAKING]:    { label: 'LEO is speaking…',     subtitle: 'Speak to interrupt at any time.' },
    [VS.INTERRUPTED]: { label: 'Interrupted',          subtitle: 'Captured your input — processing…' },
    [VS.ERROR]:       { label: 'Something went wrong', subtitle: 'Tap Retry to resume listening.' },
  };

  // ═══════════════════════════════════════════════════════════════════════
  // Helpers: get shared dashboard state injected by app.js
  // ═══════════════════════════════════════════════════════════════════════
  function getDashState() {
    return window.__LEO_DASHBOARD__?.getState?.() || null;
  }

  function getActiveCallId() {
    return getDashState()?.activeCallId || null;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // State Machine: transition + render
  // ═══════════════════════════════════════════════════════════════════════
  function transitionTo(newState, subtitleOverride) {
    if (voiceState.current === newState) return;
    console.info(`[VoiceMode] ${voiceState.current} → ${newState}`);
    voiceState.current = newState;
    renderVoiceState(subtitleOverride);
  }

  function renderVoiceState(subtitleOverride) {
    const s = voiceState.current;
    const meta = STATE_META[s] || STATE_META[VS.IDLE];

    // Update overlay class for CSS-driven animations
    vmEl.overlay.className = `vm-state-${s}${voiceState.sessionActive ? ' active' : ''}`;
    if (voiceState.sessionActive) vmEl.overlay.classList.add('active');

    vmEl.stateLabel.textContent = meta.label;
    vmEl.stateSubtitle.textContent = subtitleOverride || meta.subtitle;

    // Show/hide retry button
    if (s === VS.ERROR) {
      vmEl.retryBtn.classList.add('visible');
    } else {
      vmEl.retryBtn.classList.remove('visible');
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Transcript Bubbles inside Voice Mode overlay
  // ═══════════════════════════════════════════════════════════════════════
  function addVoiceBubble(role, text) {
    // Keep only last 4 bubbles to avoid overflow
    while (vmEl.transcriptArea.children.length >= 4) {
      vmEl.transcriptArea.removeChild(vmEl.transcriptArea.firstChild);
    }

    const bubble = document.createElement('div');
    bubble.className = `vm-transcript-bubble ${role}`;
    bubble.textContent = text;
    vmEl.transcriptArea.appendChild(bubble);
    vmEl.transcriptArea.scrollTop = vmEl.transcriptArea.scrollHeight;
  }

  function updateLastBubble(role, text) {
    const bubbles = vmEl.transcriptArea.querySelectorAll(`.vm-transcript-bubble.${role}`);
    if (bubbles.length > 0) {
      bubbles[bubbles.length - 1].textContent = text;
    } else {
      addVoiceBubble(role, text);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Microphone Permission
  // ═══════════════════════════════════════════════════════════════════════
  async function requestMicPermission() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });
      voiceState.micStream = stream;
      return true;
    } catch (err) {
      console.error('[VoiceMode] Mic permission denied:', err);
      return false;
    }
  }

  function releaseMicStream() {
    if (voiceState.micStream) {
      voiceState.micStream.getTracks().forEach(t => t.stop());
      voiceState.micStream = null;
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // SpeechRecognition — Continuous Listening Engine
  // ═══════════════════════════════════════════════════════════════════════
  function createRecognition() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) return null;

    const rec = new SR();
    rec.continuous      = true;    // ← keeps mic alive; no click-per-utterance
    rec.interimResults  = true;    // ← interim results for live display
    rec.maxAlternatives = 1;

    // Language from dropdown
    const lang = vmEl.langSelect?.value;
    rec.lang = (!lang || lang === 'auto') ? 'en-IN' : lang;

    // ── onstart ──────────────────────────────────────────────────────────
    rec.onstart = () => {
      console.info('[VoiceMode] Recognition started (continuous)');
      if (voiceState.current !== VS.SPEAKING && voiceState.current !== VS.PROCESSING) {
        transitionTo(VS.LISTENING);
      }
    };

    // ── onspeechstart — VAD triggers here ──────────────────────────────
    rec.onspeechstart = () => {
      console.info('[VoiceMode] Speech detected (VAD)');

      // Barge-in: if LEO is speaking, interrupt immediately
      if (voiceState.current === VS.SPEAKING && !voiceState.bargeInCooldown) {
        handleBargeIn();
      } else if (voiceState.current === VS.LISTENING) {
        // Normal listening — user started speaking
        transitionTo(VS.LISTENING);
      }
    };

    // ── onresult — accumulate transcript ───────────────────────────────
    rec.onresult = (event) => {
      let interim = '';
      let finalText = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          finalText += result[0].transcript;
        } else {
          interim += result[0].transcript;
        }
      }

      // Show interim text live in the orb subtitle
      if (interim && voiceState.current === VS.LISTENING) {
        vmEl.stateSubtitle.textContent = interim;
      }

      if (finalText.trim()) {
        const utterance = finalText.trim();
        voiceState.lastUserText = utterance;
        console.info('[VoiceMode] Final utterance:', utterance);

        // Show user bubble
        addVoiceBubble('user', utterance);

        // Proceed to process only if not currently in barge-in cooldown
        if (voiceState.current !== VS.PROCESSING && voiceState.current !== VS.SPEAKING) {
          processUserUtterance(utterance);
        } else if (voiceState.current === VS.INTERRUPTED) {
          processUserUtterance(utterance);
        }
      }
    };

    // ── onerror ───────────────────────────────────────────────────────
    rec.onerror = (event) => {
      console.warn('[VoiceMode] Recognition error:', event.error);

      // 'no-speech' is non-fatal — just keep listening
      if (event.error === 'no-speech') {
        if (voiceState.sessionActive && voiceState.current === VS.LISTENING) {
          // Restart silently
          scheduleRecognitionRestart(300);
        }
        return;
      }

      // Aborted errors during barge-in are expected
      if (event.error === 'aborted') {
        return;
      }

      if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
        transitionTo(VS.ERROR, 'Microphone access was denied. Please allow microphone in browser settings.');
        return;
      }

      // Other errors: attempt restart if session is still active
      if (voiceState.sessionActive) {
        scheduleRecognitionRestart(1000);
      }
    };

    // ── onend — recognition stopped (happens after no-speech, aborted, etc.) ──
    rec.onend = () => {
      console.info('[VoiceMode] Recognition ended. sessionActive=', voiceState.sessionActive);

      if (!voiceState.sessionActive) return;

      // Auto-restart if we're still listening or if we were speaking and finished
      if (
        voiceState.current === VS.LISTENING ||
        voiceState.current === VS.SPEAKING ||
        voiceState.current === VS.IDLE
      ) {
        scheduleRecognitionRestart(200);
      }
    };

    return rec;
  }

  function scheduleRecognitionRestart(delayMs = 300) {
    if (voiceState.restartPending || !voiceState.sessionActive) return;
    voiceState.restartPending = true;

    setTimeout(() => {
      voiceState.restartPending = false;
      if (!voiceState.sessionActive) return;

      try {
        if (voiceState.recognition) {
          try { voiceState.recognition.stop(); } catch (_) {}
        }
        voiceState.recognition = createRecognition();
        voiceState.recognition.start();
        console.info('[VoiceMode] Recognition restarted');
      } catch (err) {
        console.warn('[VoiceMode] Restart failed:', err);
        if (voiceState.sessionActive) {
          scheduleRecognitionRestart(1500);
        }
      }
    }, delayMs);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Barge-In Handler
  // ═══════════════════════════════════════════════════════════════════════
  function handleBargeIn() {
    if (voiceState.bargeInCooldown) return;
    voiceState.bargeInCooldown = true;
    voiceState.bargeInFired = true;

    console.info('[VoiceMode] Barge-in detected — stopping LEO audio');

    // 1. Stop local TTS playback immediately
    const deps = window.__LEO_VOICE_DEPS__;
    if (deps?.stopCurrentAudio) deps.stopCurrentAudio();

    // 2. Signal server-side barge-in
    const callId = getActiveCallId();
    if (callId) {
      fetch(`/api/v1/demo/calls/${callId}/barge-in`, { method: 'POST' }).catch(() => {});
    }

    // 3. Transition to INTERRUPTED then quickly to LISTENING
    transitionTo(VS.INTERRUPTED, 'Capturing your message…');

    // Cooldown prevents echo from speaker triggering barge-in again
    setTimeout(() => {
      voiceState.bargeInCooldown = false;
      voiceState.bargeInFired = false;
    }, 1800);

    // Clear any pending "return to listening" timer
    if (voiceState.speakingEndTimer) {
      clearTimeout(voiceState.speakingEndTimer);
      voiceState.speakingEndTimer = null;
    }

    // Short delay then move to LISTENING so the new utterance gets picked up
    setTimeout(() => {
      if (voiceState.current === VS.INTERRUPTED && voiceState.sessionActive) {
        transitionTo(VS.LISTENING);
      }
    }, 300);
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Process User Utterance → Backend → TTS
  // ═══════════════════════════════════════════════════════════════════════
  async function processUserUtterance(text) {
    if (!text || !text.trim()) return;

    const callId = getActiveCallId();
    if (!callId) {
      transitionTo(VS.ERROR, 'No active call session. Start a call from the dashboard first.');
      return;
    }

    // Stop any current audio
    const deps = window.__LEO_VOICE_DEPS__;
    if (deps?.stopCurrentAudio) deps.stopCurrentAudio();

    transitionTo(VS.PROCESSING);

    const playToken = 'vm_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
    voiceState.activePlayToken = playToken;

    // Also update the dashboard input and send via form (for transcript recording)
    const dashState = getDashState();
    const dashEl = window.__LEO_DASHBOARD_ELS__;

    try {
      // Use the live-stream pipeline (ultra-low-latency LLM→TTS)
      transitionTo(VS.SPEAKING, 'LEO is generating and speaking…');

      // Start listening to live stream
      const streamSuccess = await playVoiceLiveStream(callId, text, playToken);

      if (!streamSuccess) {
        // Fallback: POST to /message and wait for SSE turn
        await fetch(`/api/v1/demo/calls/${callId}/message`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text }),
        });
      }

    } catch (err) {
      console.error('[VoiceMode] Process utterance error:', err);
      if (voiceState.activePlayToken === playToken) {
        transitionTo(VS.ERROR, `Error: ${err.message}. Tap Retry.`);
        addVoiceBubble('error', '⚠️ ' + err.message);
      }
      return;
    }

    // After TTS finishes, return to LISTENING
    if (voiceState.activePlayToken === playToken && voiceState.sessionActive) {
      returnToListening();
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Live-Stream TTS Player (reuses existing framed-PCM protocol)
  // ═══════════════════════════════════════════════════════════════════════
  async function playVoiceLiveStream(callId, messageText, playToken) {
    let audioCtx;
    try {
      const deps = window.__LEO_VOICE_DEPS__;
      audioCtx = deps?.getOrCreateAudioContext?.() ||
        new (window.AudioContext || window.webkitAudioContext)();
    } catch (e) {
      return false;
    }

    const liveStart = performance.now();
    let accumulatedText = '';
    let firstAudioPlayed = false;
    let sampleRate = 24000;
    let buffer = new Uint8Array(0);
    let activeStreamSources = [];
    let nextChunkPlayTime = audioCtx.currentTime;
    let streamReader = null;

    // Register cleanup with shared stopCurrentAudio
    const deps = window.__LEO_VOICE_DEPS__;
    if (deps) {
      deps._vmStreamSources = activeStreamSources;
      deps._vmStreamReader = { cancel: () => { if (streamReader) { try { streamReader.cancel(); } catch (_) {} } } };
    }

    try {
      const resp = await fetch(`/api/v1/demo/calls/${callId}/live-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageText }),
      });

      if (!resp.ok) {
        console.warn('[VoiceMode] live-stream HTTP', resp.status);
        return false;
      }

      streamReader = resp.body.getReader();

      function appendToBuffer(chunk) {
        const merged = new Uint8Array(buffer.length + chunk.length);
        merged.set(buffer, 0);
        merged.set(chunk, buffer.length);
        buffer = merged;
      }

      while (true) {
        // Check if this play token is still current (barge-in may have changed it)
        if (voiceState.activePlayToken !== playToken) {
          try { streamReader.cancel(); } catch (_) {}
          activeStreamSources.forEach(s => { try { s.stop(); } catch (_) {} });
          return true; // graceful cancellation
        }

        const { done, value } = await streamReader.read();
        if (done) break;

        appendToBuffer(value);

        // Parse framed protocol
        while (buffer.length >= 5) {
          const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
          const frameLen = view.getUint32(0, false);
          if (buffer.length < 4 + frameLen) break;

          const frameType = buffer[4];
          const payload = buffer.slice(5, 4 + frameLen);
          buffer = buffer.slice(4 + frameLen);

          if (frameType === 0x03) {
            // TEXT frame — show incrementally
            const sentence = new TextDecoder().decode(payload);
            accumulatedText += (accumulatedText ? ' ' : '') + sentence;
            voiceState.lastLeoText = accumulatedText;
            updateLastBubble('assistant', accumulatedText);

          } else if (frameType === 0x01) {
            // HEADER — extract sample rate
            if (payload.length >= 8) {
              const hv = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
              sampleRate = hv.getUint32(0, false) || 24000;
            }

          } else if (frameType === 0x02) {
            // PCM chunk — schedule playback
            if (payload.length >= 2 && voiceState.activePlayToken === playToken) {
              const numSamples = Math.floor(payload.length / 2);
              const pcm16 = new Int16Array(payload.buffer, payload.byteOffset, numSamples);
              const audioBuf = audioCtx.createBuffer(1, numSamples, sampleRate);
              const ch = audioBuf.getChannelData(0);
              for (let i = 0; i < numSamples; i++) ch[i] = pcm16[i] / 32768.0;

              const src = audioCtx.createBufferSource();
              src.buffer = audioBuf;
              src.connect(audioCtx.destination);
              activeStreamSources.push(src);

              const now = audioCtx.currentTime;
              const start = Math.max(now + 0.02, nextChunkPlayTime);
              src.start(start);
              nextChunkPlayTime = start + audioBuf.duration;

              if (!firstAudioPlayed) {
                firstAudioPlayed = true;
                const ttfaMs = performance.now() - liveStart;
                console.info(`[VoiceMode] ⚡ First audio in ${ttfaMs.toFixed(0)}ms`);
                transitionTo(VS.SPEAKING, 'LEO is speaking… (interrupt anytime)');
                // Create assistant bubble when first audio plays
                if (!accumulatedText) addVoiceBubble('assistant', '…');
              }
            }

          } else if (frameType === 0x04) {
            // DONE
            break;

          } else if (frameType === 0x05) {
            const errMsg = new TextDecoder().decode(payload);
            console.warn('[VoiceMode] Stream error frame:', errMsg);
            return false;
          }
        }
      }

      if (firstAudioPlayed) {
        // Wait for scheduled audio to finish playing
        const remainingMs = Math.max(0, (nextChunkPlayTime - audioCtx.currentTime) * 1000);
        await new Promise(resolve => {
          voiceState.speakingEndTimer = setTimeout(resolve, remainingMs + 150);
        });
        voiceState.speakingEndTimer = null;
        return true;
      }

      return false;

    } catch (err) {
      if (err.name !== 'AbortError') {
        console.warn('[VoiceMode] Live stream error:', err);
      }
      return false;
    } finally {
      streamReader = null;
      if (deps) {
        deps._vmStreamSources = null;
        deps._vmStreamReader = null;
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Return to Listening (after TTS finishes)
  // ═══════════════════════════════════════════════════════════════════════
  function returnToListening() {
    if (!voiceState.sessionActive) return;

    // Brief cooldown so LEO's own speaker echo doesn't re-trigger barge-in
    voiceState.bargeInCooldown = true;
    setTimeout(() => { voiceState.bargeInCooldown = false; }, 1200);

    transitionTo(VS.LISTENING, 'Your turn — ask anything.');

    // Ensure recognition is running
    if (voiceState.recognition) {
      try {
        // If already running (continuous), nothing to do.
        // If it stopped (e.g., no-speech), restart it.
        voiceState.recognition.start();
      } catch (e) {
        // DOMException: recognition already started — that's fine
        if (e.name !== 'InvalidStateError') {
          scheduleRecognitionRestart(200);
        }
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Session Start / Stop
  // ═══════════════════════════════════════════════════════════════════════
  async function startVoiceSession() {
    if (voiceState.sessionActive) return;

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      alert('Speech Recognition is not supported in this browser. Please use Google Chrome or Microsoft Edge.');
      return;
    }

    // Show overlay immediately
    voiceState.sessionActive = true;
    vmEl.overlay.classList.add('active');
    vmEl.permPrompt.classList.remove('visible');
    vmEl.mainUI.style.display = 'flex';
    transitionTo(VS.IDLE, 'Requesting microphone permission…');

    // Request mic permission
    const granted = await requestMicPermission();
    if (!granted) {
      // Show permission prompt UI
      vmEl.permPrompt.classList.add('visible');
      vmEl.mainUI.style.display = 'none';
      transitionTo(VS.ERROR, 'Microphone access denied.');
      return;
    }

    // Start analyser for visual bar animation
    startOrbAnalyser();

    // Clear transcript
    vmEl.transcriptArea.innerHTML = '';

    // Start recognition
    voiceState.recognition = createRecognition();
    if (!voiceState.recognition) {
      transitionTo(VS.ERROR, 'Failed to initialise Speech Recognition.');
      return;
    }

    try {
      voiceState.recognition.start();
    } catch (err) {
      console.error('[VoiceMode] Failed to start recognition:', err);
      transitionTo(VS.ERROR, 'Could not start microphone: ' + err.message);
      return;
    }

    transitionTo(VS.LISTENING, 'Microphone is active. Speak your question.');
    console.info('[VoiceMode] Session started');
  }

  function stopVoiceSession() {
    console.info('[VoiceMode] Session stopping');

    voiceState.sessionActive = false;
    voiceState.activePlayToken = null;

    // Stop recognition
    if (voiceState.recognition) {
      try { voiceState.recognition.abort(); } catch (_) {}
      voiceState.recognition = null;
    }

    // Stop any playing audio
    const deps = window.__LEO_VOICE_DEPS__;
    if (deps?.stopCurrentAudio) deps.stopCurrentAudio();

    // Stop voice stream sources if any
    if (deps?._vmStreamSources) {
      deps._vmStreamSources.forEach(s => { try { s.stop(); } catch (_) {} });
      deps._vmStreamSources = null;
    }
    if (deps?._vmStreamReader) {
      deps._vmStreamReader.cancel();
      deps._vmStreamReader = null;
    }

    // Release mic
    releaseMicStream();

    // Stop analyser animation
    if (typeof stopOrbAnalyser === 'function') stopOrbAnalyser();

    // Clear timers
    if (voiceState.speakingEndTimer) {
      clearTimeout(voiceState.speakingEndTimer);
      voiceState.speakingEndTimer = null;
    }

    // Hide overlay
    vmEl.overlay.classList.remove('active');
    vmEl.overlay.className = 'vm-state-idle';

    voiceState.current = VS.IDLE;
    voiceState.bargeInCooldown = false;
    voiceState.bargeInFired = false;
    voiceState.restartPending = false;
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Event Listeners
  // ═══════════════════════════════════════════════════════════════════════
  function setupVoiceListeners() {
    // Activation button (in dashboard interaction bar)
    if (vmEl.activateBtn) {
      vmEl.activateBtn.addEventListener('click', () => {
        // Check active call
        const callId = getActiveCallId();
        const dashState = getDashState();
        const call = dashState?.sessions?.[callId];
        const isCallActive = call && call.status !== 'completed' && call.status !== 'failed' && call.status !== 'ringing';

        if (!callId || !isCallActive) {
          alert('Please start or select an active call session first, then click Voice Mode.');
          return;
        }
        startVoiceSession();
      });
    }

    // Close / End Session button
    if (vmEl.closeBtn) {
      vmEl.closeBtn.addEventListener('click', stopVoiceSession);
    }

    // Grant mic button (permission prompt)
    if (vmEl.grantMicBtn) {
      vmEl.grantMicBtn.addEventListener('click', async () => {
        vmEl.permPrompt.classList.remove('visible');
        vmEl.mainUI.style.display = 'flex';
        const granted = await requestMicPermission();
        if (granted) {
          voiceState.recognition = createRecognition();
          voiceState.recognition.start();
          transitionTo(VS.LISTENING, 'Microphone is active. Speak your question.');
        } else {
          vmEl.permPrompt.classList.add('visible');
          vmEl.mainUI.style.display = 'none';
          transitionTo(VS.ERROR, 'Microphone access still denied.');
        }
      });
    }

    // Retry button
    if (vmEl.retryBtn) {
      vmEl.retryBtn.addEventListener('click', () => {
        vmEl.retryBtn.classList.remove('visible');
        if (!voiceState.sessionActive) return;
        voiceState.recognition = createRecognition();
        try {
          voiceState.recognition.start();
          transitionTo(VS.LISTENING, 'Reconnected — speak your question.');
        } catch (err) {
          scheduleRecognitionRestart(500);
        }
      });
    }

    // Language selector — update recognition lang
    if (vmEl.langSelect) {
      vmEl.langSelect.addEventListener('change', () => {
        if (voiceState.recognition && voiceState.sessionActive) {
          const newLang = vmEl.langSelect.value;
          voiceState.recognition.lang = (!newLang || newLang === 'auto') ? 'en-IN' : newLang;
          // Restart to apply new lang
          try { voiceState.recognition.abort(); } catch (_) {}
          scheduleRecognitionRestart(200);
        }
      });
    }

    // Dismiss overlay by clicking the dark backdrop (outside the center content)
    if (vmEl.overlay) {
      vmEl.overlay.addEventListener('click', (e) => {
        if (e.target === vmEl.overlay) {
          stopVoiceSession();
        }
      });
    }

    // Keyboard: Escape to end session
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && voiceState.sessionActive) {
        stopVoiceSession();
      }
    });
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Bar amplitude animation driven by microphone AnalyserNode
  // (visual-only; VAD is handled by SpeechRecognition natively)
  // ═══════════════════════════════════════════════════════════════════════
  let _analyserNode = null;
  let _analyserAnimId = null;
  let _analyserData = null;

  function startOrbAnalyser() {
    if (!voiceState.micStream) return;
    try {
      const deps = window.__LEO_VOICE_DEPS__;
      const audioCtx = deps?.getOrCreateAudioContext?.() ||
        new (window.AudioContext || window.webkitAudioContext)();
      // Resume context if suspended (browser autoplay policy)
      if (audioCtx.state === 'suspended') audioCtx.resume();

      const source = audioCtx.createMediaStreamSource(voiceState.micStream);
      _analyserNode = audioCtx.createAnalyser();
      _analyserNode.fftSize = 64;
      _analyserData = new Uint8Array(_analyserNode.frequencyBinCount);
      source.connect(_analyserNode);

      const bars = vmEl.orbBars?.querySelectorAll('.vm-orb-bar');
      const NUM_BARS = bars?.length || 5;

      function drawFrame() {
        if (!voiceState.sessionActive) {
          cancelAnimationFrame(_analyserAnimId);
          // Reset bar heights
          if (bars) bars.forEach(b => { b.style.height = '6px'; });
          return;
        }
        _analyserAnimId = requestAnimationFrame(drawFrame);

        // Only drive bars manually during LISTENING/SPEAKING;
        // CSS animations handle the decorative animation during PROCESSING.
        if (voiceState.current !== VS.LISTENING && voiceState.current !== VS.SPEAKING) return;

        _analyserNode.getByteFrequencyData(_analyserData);

        if (bars) {
          for (let i = 0; i < NUM_BARS; i++) {
            const binIdx = Math.floor((i / NUM_BARS) * _analyserData.length * 0.65) + 2;
            const amplitude = _analyserData[binIdx] || 0;
            // Scale 0–255 → 5px–32px
            const height = 5 + (amplitude / 255) * 27;
            bars[i].style.height = height.toFixed(1) + 'px';
            bars[i].style.animation = 'none'; // disable CSS animation when using real data
          }
        }
      }

      drawFrame();
    } catch (e) {
      console.warn('[VoiceMode] Analyser failed (non-critical):', e);
      // Analyser is visual-only; failure is non-fatal
    }
  }

  function stopOrbAnalyser() {
    if (_analyserAnimId) {
      cancelAnimationFrame(_analyserAnimId);
      _analyserAnimId = null;
    }
    _analyserNode = null;
    _analyserData = null;
    // Re-enable CSS animations on bars
    const bars = vmEl.orbBars?.querySelectorAll('.vm-orb-bar');
    if (bars) bars.forEach(b => { b.style.animation = ''; b.style.height = ''; });
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Integration: hook into dashboard's stopCurrentAudio for barge-in
  // ═══════════════════════════════════════════════════════════════════════
  function injectDeps() {
    // Wait for __LEO_DASHBOARD__ to be registered by app.js
    const check = () => {
      if (window.__LEO_DASHBOARD__ && window.__LEO_VOICE_DEPS__) {
        setupVoiceListeners();
        console.info('[VoiceMode] Initialised and ready');
      } else {
        setTimeout(check, 100);
      }
    };
    check();
  }

  // ═══════════════════════════════════════════════════════════════════════
  // Bootstrap
  // ═══════════════════════════════════════════════════════════════════════
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectDeps);
  } else {
    injectDeps();
  }

  // Expose for debugging
  window.__LEO_VOICE_MODE__ = {
    getState: () => voiceState,
    start: startVoiceSession,
    stop: stopVoiceSession,
    transitionTo,
  };

})();
