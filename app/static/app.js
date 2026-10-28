/**
 * LEO AI Call Center Operations Dashboard - Frontend Client Logic
 * Manages 25 concurrent call sessions, real-time SSE stream, waveform canvas,
 * incoming call modal, transcript stream, human escalation queue, and call history.
 */

(function () {
  'use strict';

  // Application State
  const state = {
    sessions: {},             // Map of call_id -> DemoCall
    activeCallId: null,       // Currently selected call in monitor
    agents: [],               // List of HumanAgent
    metrics: null,            // Aggregated CallMetrics
    eventSource: null,        // SSE EventSource instance
    waveformAnimationId: null,// Canvas animation frame
    callTimerInterval: null,  // Interval for call duration timer
    incomingCountdownTimer: null, // Timer for auto-answer countdown
    audioActive: false,       // Whether audio visualization should be energetic
    // --- Audio playback ---
    audioCtx: null,           // Web AudioContext (created on first user gesture)
    currentAudioSource: null, // Currently playing AudioBufferSourceNode
    currentPlayToken: null,   // Per-call playback cancellation token (turn_id string)
    ttsEnabled: false,        // Whether TTS provider is real (not mock)
    audioSynthesizing: {},    // Map of turn_id -> boolean (in-flight synthesis)
    activeStreamSources: [],  // Currently queued AudioBufferSourceNodes for progressive streaming
    streamReader: null,       // In-flight ReadableStreamDefaultReader
    nextChunkPlayTime: 0,     // Web AudioContext timeline tracking for gapless scheduling
    // --- Polling & Lifecycle ---
    sessionPollTimer: null,   // Timer for controlled session polling
    healthPollTimer: null,    // Interval for health checking
    isFetchingSessions: false,// Overlap guard for active sessions fetch
    isFetchingHistory: false, // Overlap guard for history fetch
  };

  // DOM Elements
  const el = {
    // Top Bar
    systemHealthPill: document.getElementById('system-health-pill'),
    healthLabel: document.getElementById('health-label'),
    concurrencyLabel: document.getElementById('concurrency-label'),
    delaySelect: document.getElementById('delay-select'),
    btnSimulateCall: document.getElementById('btn-simulate-call'),
    sseIndicator: document.getElementById('sse-indicator'),
    providerName: document.getElementById('provider-name'),

    // Metrics Bar
    statTotal: document.getElementById('stat-total'),
    statActive: document.getElementById('stat-active'),
    statCompleted: document.getElementById('stat-completed'),
    statAiRate: document.getElementById('stat-ai-rate'),
    statEscalated: document.getElementById('stat-escalated'),

    // Session Queue
    sessionList: document.getElementById('session-list'),
    queueCountBadge: document.getElementById('queue-count-badge'),
    emptyQueueState: document.getElementById('empty-queue-state'),

    // Center Monitor
    monitorAvatar: document.getElementById('monitor-avatar'),
    monitorHeading: document.getElementById('monitor-heading'),
    monitorStatusPill: document.getElementById('monitor-status-pill'),
    monitorEscalationPill: document.getElementById('monitor-escalation-pill'),
    monitorCallerId: document.getElementById('monitor-caller-id'),
    monitorCallerPhone: document.getElementById('monitor-caller-phone'),
    monitorTimer: document.getElementById('monitor-timer'),
    btnMute: document.getElementById('btn-mute'),
    muteLabel: document.getElementById('mute-label'),
    btnEscalateQuick: document.getElementById('btn-escalate-quick'),
    btnHangup: document.getElementById('btn-hangup'),

    // Waveform
    waveformIndicator: document.getElementById('waveform-indicator'),
    acousticStatusLabel: document.getElementById('acoustic-status-label'),
    waveformCanvas: document.getElementById('waveform-canvas'),

    // Transcript & Input
    transcriptStream: document.getElementById('transcript-stream'),
    emptyTranscriptMessage: document.getElementById('empty-transcript-message'),
    customerMessageForm: document.getElementById('customer-message-form'),
    customerInput: document.getElementById('customer-input'),
    btnSendMessage: document.getElementById('btn-send-message'),
    btnMicrophone: document.getElementById('btn-microphone'),
    quickPrompts: document.getElementById('quick-prompts'),

    // Right Handoff
    escalationCard: document.getElementById('escalation-card'),
    escalationEmptyNotice: document.getElementById('escalation-empty-notice'),
    escalationDetails: document.getElementById('escalation-details'),
    escReasonText: document.getElementById('esc-reason-text'),
    escNotesText: document.getElementById('esc-notes-text'),
    assignedAgentName: document.getElementById('assigned-agent-name'),
    agentList: document.getElementById('agent-list'),
    agentOnlineCount: document.getElementById('agent-online-count'),

    // History Table
    historyTableBody: document.getElementById('history-table-body'),
    historySearch: document.getElementById('history-search'),
    historyFilterStatus: document.getElementById('history-filter-status'),
    btnRefreshHistory: document.getElementById('btn-refresh-history'),

    // Modals
    incomingModal: document.getElementById('incoming-modal'),
    modalCallerName: document.getElementById('modal-caller-name'),
    modalCallerPhone: document.getElementById('modal-caller-phone'),
    modalCallId: document.getElementById('modal-call-id'),
    modalCountdown: document.getElementById('modal-countdown'),
    modalProgress: document.getElementById('modal-progress'),
    btnModalAnswer: document.getElementById('btn-modal-answer'),
    btnModalReject: document.getElementById('btn-modal-reject'),

    // Drawer Modal
    drawerModal: document.getElementById('drawer-modal'),
    drawerTitle: document.getElementById('drawer-title'),
    drawerSubtitle: document.getElementById('drawer-subtitle'),
    drawerBody: document.getElementById('drawer-body'),
    btnCloseDrawer: document.getElementById('btn-close-drawer'),
  };

  // ============================================================================
  // Initialisation
  // ============================================================================
  async function init() {
    setupEventListeners();
    initWaveform();
    startCallTimer();
    connectSSE();
    await checkHealth();
    await fetchTTSConfig();
    await fetchAgents();
    await fetchMetrics();
    await loadActiveSessions();
    await loadHistory();

    // Start controlled periodic health probe & session sync
    state.healthPollTimer = setInterval(checkHealth, 15000);
    scheduleNextSessionPoll();
  }


  // ============================================================================
  // Event Listeners Setup
  // ============================================================================
  function setupEventListeners() {
    // Simulate Incoming Call
    el.btnSimulateCall.addEventListener('click', onSimulateCallClick);

    // Call Actions
    el.btnMute.addEventListener('click', onMuteToggle);
    el.btnEscalateQuick.addEventListener('click', onQuickEscalate);
    el.btnHangup.addEventListener('click', onHangupClick);

    // Incoming Call Modal Actions
    el.btnModalAnswer.addEventListener('click', onModalAnswer);
    el.btnModalReject.addEventListener('click', onModalReject);

    // Send Customer Turn
    el.customerMessageForm.addEventListener('submit', onCustomerMessageSubmit);

    // Microphone speech recognition
    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!el.btnMicrophone) {
      console.error('[LEO] Microphone button not found in HTML.');
    } else if (!SpeechRecognition) {
      el.btnMicrophone.addEventListener('click', () => {
        alert('Speech recognition supported nahi hai. Google Chrome use karo.');
      });
    } else {
      const recognition = new SpeechRecognition();

      recognition.lang = 'en-IN';
      recognition.continuous = false;
      recognition.interimResults = false;

      recognition.onstart = () => {
        el.btnMicrophone.textContent = '🎙️ Listening...';
        console.info('[LEO Microphone] Recognition started');
        // Immediate local stop and server-side barge-in
        stopCurrentAudio();
        if (state.activeCallId) {
          fetch(`/api/v1/demo/calls/${state.activeCallId}/barge-in`, { method: 'POST' }).catch(() => {});
        }
      };

      recognition.onspeechstart = () => {
        // Customer speech onset detected: cancel any ongoing speech immediately
        stopCurrentAudio();
        if (state.activeCallId) {
          fetch(`/api/v1/demo/calls/${state.activeCallId}/barge-in`, { method: 'POST' }).catch(() => {});
        }
      };

      recognition.onend = () => {
        el.btnMicrophone.textContent = '🎙️ Speak';
        console.info('[LEO Microphone] Recognition ended');
      };

      recognition.onresult = (event) => {
        const spokenText = event.results[0][0].transcript;
        el.customerInput.value = spokenText;

        el.customerMessageForm.dispatchEvent(
          new Event('submit', {
            bubbles: true,
            cancelable: true
          })
        );
      };

      recognition.onerror = (event) => {
        console.error('[LEO Microphone]', event.error);
        if (event.error !== 'no-speech') {
          alert('Microphone error: ' + event.error);
        }
      };

      el.btnMicrophone.addEventListener('click', async () => {
        if (!state.activeCallId || el.customerInput.disabled) {
          alert('Pehle ek active call start karo.');
          return;
        }

        // Stop any current audio immediately upon clicking speak
        stopCurrentAudio();
        if (state.activeCallId) {
          fetch(`/api/v1/demo/calls/${state.activeCallId}/barge-in`, { method: 'POST' }).catch(() => {});
        }

        try {
          // Dynamic language selection from dropdown or active call
          const langSelect = document.getElementById('select-speech-lang');
          if (langSelect && langSelect.value !== 'auto') {
            recognition.lang = langSelect.value;
          } else {
            const activeCall = state.sessions[state.activeCallId];
            recognition.lang = activeCall?.metadata?.language_locale || 'en-IN';
          }

          // Browser microphone permission check
          await navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
              stream.getTracks().forEach(track => track.stop());
            });

          recognition.start();
        } catch (error) {
          console.error('[LEO Microphone] Start failed:', error);

          if (error.name === 'NotAllowedError') {
            alert('Microphone permission blocked hai. Chrome site settings mein microphone Allow karo.');
          } else {
            alert('Microphone start nahi hua: ' + error.message);
          }
        }
      });
    }

    // Quick Prompts
    el.quickPrompts.addEventListener('click', (e) => {
      const chip = e.target.closest('.chip');
      if (chip && chip.dataset.text && !el.customerInput.disabled) {
        el.customerInput.value = chip.dataset.text;
        el.customerMessageForm.dispatchEvent(new Event('submit'));
      }
    });

    // History controls
    el.btnRefreshHistory.addEventListener('click', () => loadHistory());
    el.historyFilterStatus.addEventListener('change', () => loadHistory());
    el.historySearch.addEventListener('input', debounce(() => loadHistory(), 350));

    // Drawer close
    el.btnCloseDrawer.addEventListener('click', () => {
      el.drawerModal.classList.add('hidden');
    });
    el.drawerModal.addEventListener('click', (e) => {
      if (e.target === el.drawerModal) {
        el.drawerModal.classList.add('hidden');
      }
    });
  }

  // ============================================================================
  // Server-Sent Events (SSE) Stream
  // ============================================================================
  function connectSSE() {
    if (state.eventSource) {
      state.eventSource.close();
    }

    el.sseIndicator.textContent = 'SSE Connecting...';
    el.sseIndicator.className = 'sse-pill connecting';

    const evtSource = new EventSource('/api/v1/demo/events');
    state.eventSource = evtSource;

    evtSource.addEventListener('open', () => {
      el.sseIndicator.textContent = 'SSE Live Gateway Active';
      el.sseIndicator.className = 'sse-pill connected';

      // Resync sessions when SSE connects or reconnects
      loadActiveSessions();
      scheduleNextSessionPoll();
    });

    evtSource.addEventListener('ping', () => {
      // keepalive ping
    });

    evtSource.addEventListener('sessions_snapshot', (e) => {
      const calls = JSON.parse(e.data);

      state.sessions = Object.fromEntries(
        calls.map(call => [call.call_id, call])
      );

      renderSessionList();

      if (
        state.activeCallId &&
        state.sessions[state.activeCallId]
      ) {
        updateMonitorUI();
        renderTranscriptUI();
        updateEscalationUI();
      } else if (calls.length > 0) {
        selectCall(calls[0].call_id);
      } else {
        state.activeCallId = null;
        updateMonitorUI();
        renderTranscriptUI();
        updateEscalationUI();
      }
    });


    evtSource.addEventListener('call_ringing', (e) => {
      const call = JSON.parse(e.data);
      handleCallRinging(call);
    });

    evtSource.addEventListener('call_state_changed', (e) => {
      const data = JSON.parse(e.data);
      handleCallStateChanged(data);
    });

    evtSource.addEventListener('transcript_turn', (e) => {
      const data = JSON.parse(e.data);
      handleTranscriptTurn(data);
    });

    evtSource.addEventListener('call_escalated', (e) => {
      const data = JSON.parse(e.data);
      handleCallEscalated(data);
    });

    evtSource.addEventListener('agent_assigned', (e) => {
      const data = JSON.parse(e.data);
      handleAgentAssigned(data);
    });

    evtSource.addEventListener('call_muted', (e) => {
      const data = JSON.parse(e.data);
      if (state.sessions[data.call_id]) {
        state.sessions[data.call_id].is_muted = data.is_muted;
        if (state.activeCallId === data.call_id) {
          updateMonitorUI();
        }
      }
    });

    evtSource.onerror = () => {
      el.sseIndicator.textContent = 'SSE Reconnecting...';
      el.sseIndicator.className = 'sse-pill connecting';
      scheduleNextSessionPoll();
    };
  }

  // ============================================================================
  // Incoming Call & State Event Handlers
  // ============================================================================
  function handleCallRinging(call) {
    state.sessions[call.call_id] = call;
    renderSessionList();
    debouncedFetchMetrics();
    // If no active call is selected, or this is a fresh ringing call, focus it
    if (!state.activeCallId || state.sessions[state.activeCallId]?.status === 'completed') {
      selectCall(call.call_id);
    }

    // Show modal if auto answer delay is configured
    showIncomingModal(call);
  }

  function handleCallStateChanged(data) {
    const { call_id, new_status, call } = data;
    if (call) {
      state.sessions[call_id] = call;
    } else if (state.sessions[call_id]) {
      state.sessions[call_id].status = new_status;
    }

    // Dismiss incoming modal if this call is no longer ringing
    if (new_status !== 'ringing' && el.incomingModal.dataset.callId === call_id) {
      hideIncomingModal();
    }

    renderSessionList();
    if (state.activeCallId === call_id) {
      updateMonitorUI();
    }
    debouncedFetchMetrics();
    debouncedLoadHistory();
  }

  const debouncedFetchMetrics = debounce(() => fetchMetrics(), 500);
  const debouncedLoadHistory = debounce(() => loadHistory(), 800);

  function handleTranscriptTurn(data) {
    const { call_id, turn } = data;
    const call = state.sessions[call_id];
    if (call) {
      // Prevent duplicate turn insertion
      if (!call.transcript.some(t => t.id === turn.id)) {
        call.transcript.push(turn);
      }
      if (state.activeCallId === call_id) {
        appendTranscriptTurnUI(turn);
      }
    }

    // Audio playback & timing: synthesize and play LEO audio automatically
    if (turn.role === 'leo' && state.activeCallId === call_id) {
      if (state.lastCustomerTurnTime) {
        const turnUiLatency = performance.now() - state.lastCustomerTurnTime;
        console.info(`[TIMING] LEO turn UI rendered: ${turnUiLatency.toFixed(1)}ms after customer turn (turnId=${turn.id})`);
      }
      triggerAudioActivity(3500);
      synthesizeAndPlayTurn(call_id, turn);
    } else if (turn.role === 'customer') {
      triggerAudioActivity(1800);
    }
  }

  function handleCallEscalated(data) {
    const { call_id, reason, notes } = data;
    const call = state.sessions[call_id];
    if (call) {
      call.escalated = true;
      call.escalation_reason = reason;
      call.escalation_notes = notes;
    }
    renderSessionList();
    if (state.activeCallId === call_id) {
      updateMonitorUI();
      updateEscalationUI();
    }
    debouncedFetchMetrics();
  }

  function handleAgentAssigned(data) {
    const { call_id, agent_id, agent_name } = data;
    const call = state.sessions[call_id];
    if (call) {
      call.assigned_agent_id = agent_id;
      call.assigned_agent_name = agent_name;
    }
    fetchAgents();
    renderSessionList();
    if (state.activeCallId === call_id) {
      updateMonitorUI();
      updateEscalationUI();
    }
  }

  // ============================================================================
  // Incoming Call Modal
  // ============================================================================
  function showIncomingModal(call) {
    el.incomingModal.dataset.callId = call.call_id;
    el.modalCallerName.textContent = call.caller_name;
    el.modalCallerPhone.textContent = call.caller_phone_masked;
    el.modalCallId.textContent = `CALL ID: ${call.caller_id} • ${call.call_id}`;

    const delay = call.auto_answer_delay_sec || 0;
    if (delay > 0) {
      let remaining = delay;
      el.modalCountdown.textContent = `${remaining}s`;
      el.modalProgress.style.transition = `width ${delay}s linear`;
      el.modalProgress.style.width = '0%';

      clearInterval(state.incomingCountdownTimer);
      state.incomingCountdownTimer = setInterval(() => {
        remaining -= 1;
        if (remaining <= 0) {
          clearInterval(state.incomingCountdownTimer);
          hideIncomingModal();
        } else {
          el.modalCountdown.textContent = `${remaining}s`;
        }
      }, 1000);
    } else {
      el.modalCountdown.textContent = 'Manual Answer Required';
      el.modalProgress.style.width = '100%';
    }

    el.incomingModal.classList.remove('hidden');
  }

  function hideIncomingModal() {
    clearInterval(state.incomingCountdownTimer);
    el.incomingModal.classList.add('hidden');
    el.incomingModal.dataset.callId = '';
    el.modalProgress.style.transition = 'none';
    el.modalProgress.style.width = '100%';
  }

  async function onModalAnswer() {
    const callId = el.incomingModal.dataset.callId;
    if (!callId) return;
    hideIncomingModal();
    await executeCallAction(callId, 'answer');
  }

  async function onModalReject() {
    const callId = el.incomingModal.dataset.callId;
    if (!callId) return;
    hideIncomingModal();
    await executeCallAction(callId, 'reject', 'Declined by demo operator.');
  }

  // ============================================================================
  // Call Actions API
  // ============================================================================
  async function onSimulateCallClick() {
    el.btnSimulateCall.disabled = true;
    const delay = parseInt(el.delaySelect.value, 10);
    try {
      const resp = await fetch('/api/v1/demo/calls/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          auto_answer_delay_sec: delay,
        }),
      });
      if (resp.ok) {
        const newCall = await resp.json();
        handleCallRinging(newCall);
      } else {
        const err = await resp.json();
        alert(err.detail || 'Failed to simulate incoming call.');
      }
    } catch (exc) {
      console.error('Simulate call error:', exc);
    } finally {
      el.btnSimulateCall.disabled = false;
    }
  }


  async function executeCallAction(callId, action, reason = null) {
    try {
      const resp = await fetch(`/api/v1/demo/calls/${callId}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, reason }),
      });
      if (resp.ok) {
        const updated = await resp.json();
        state.sessions[callId] = updated;
        if (state.activeCallId === callId) {
          updateMonitorUI();
        }
        renderSessionList();
      }
    } catch (exc) {
      console.error(`Execute action ${action} failed:`, exc);
    }
  }

  async function onMuteToggle() {
    const call = state.sessions[state.activeCallId];
    if (!call) return;
    const action = call.is_muted ? 'unmute' : 'mute';
    await executeCallAction(call.call_id, action);
  }

  async function onQuickEscalate() {
    const call = state.sessions[state.activeCallId];
    if (!call) return;
    try {
      const resp = await fetch(`/api/v1/demo/calls/${call.call_id}/escalate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reason: 'manual',
          notes: 'Operator manually escalated to human queue.',
        }),
      });
      if (resp.ok) {
        const updated = await resp.json();
        state.sessions[call.call_id] = updated;
        updateMonitorUI();
        updateEscalationUI();
      }
    } catch (exc) {
      console.error('Escalate failed:', exc);
    }
  }

  async function onHangupClick() {
    const call = state.sessions[state.activeCallId];
    if (!call) return;
    // Stop any ongoing LEO audio playback immediately
    stopCurrentAudio();
    await executeCallAction(call.call_id, 'end');
  }

  async function onCustomerMessageSubmit(e) {
    e.preventDefault();
    const text = el.customerInput.value.trim();
    if (!text || !state.activeCallId) return;

    // Immediately stop ongoing LEO audio and signal barge-in
    stopCurrentAudio();
    fetch(`/api/v1/demo/calls/${state.activeCallId}/barge-in`, { method: 'POST' }).catch(() => {});

    el.customerInput.value = '';
    el.customerInput.disabled = true;
    el.btnSendMessage.disabled = true;

    const turnStartTime = performance.now();
    const correlationId = 'turn_' + Math.random().toString(36).substring(2, 10);
    state.lastCustomerTurnTime = turnStartTime;
    state.lastCorrelationId = correlationId;
    console.info(`[TIMING] Customer turn submitted (correlationId=${correlationId})`);

    // Stop any stale speech before sending a new utterance
    stopCurrentAudio();

    // ── Canonical response path: POST /message ──────────────────────────────
    // 1. Records customer turn in session and database
    // 2. Invokes real LEO RAG brain
    // 3. Records LEO turn and broadcasts SSE transcript_turn
    // 4. SSE handler triggers synthesizeAndPlayTurn for progressive audio
    const callId = state.activeCallId;
    const messagePostPromise = fetch(`/api/v1/demo/calls/${callId}/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    }).then(async (resp) => {
      const httpDuration = performance.now() - turnStartTime;
      console.info(`[TIMING] /message POST done in ${httpDuration.toFixed(1)}ms (status=${resp.status})`);
      if (!resp.ok) {
        const errJson = await resp.json().catch(() => ({}));
        const errMsg = errJson.detail || `Server error (HTTP ${resp.status})`;
        console.warn('[LEO Client] Turn submission error:', errMsg);
        // Show structured error in transcript UI
        const emptyNotice = el.transcriptStream.querySelector('.empty-transcript');
        if (emptyNotice) emptyNotice.remove();
        const errBubble = document.createElement('div');
        errBubble.className = 'transcript-bubble system';
        errBubble.innerHTML = `
          <div class="bubble-meta">
            <span class="speaker-label system">System Notice</span>
            <span class="turn-timestamp">${new Date().toLocaleTimeString()}</span>
          </div>
          <div class="turn-text" style="color: #f87171;">⚠️ ${errMsg}</div>
        `;
        el.transcriptStream.appendChild(errBubble);
        el.transcriptStream.scrollTop = el.transcriptStream.scrollHeight;
      }
    }).catch((exc) => {
      console.error('Failed to send customer turn:', exc);
    });

    // Re-enable input as soon as message POST returns
    messagePostPromise.finally(() => {
      const call = state.sessions[callId];
      if (call && call.status !== 'completed' && call.status !== 'failed') {
        el.customerInput.disabled = false;
        el.btnSendMessage.disabled = false;
        el.customerInput.focus();
      }
    });

    await messagePostPromise;
  }

  // ============================================================================
  // Session Selection & UI Rendering
  // ============================================================================
  function selectCall(callId) {
    state.activeCallId = callId;
    renderSessionList();
    updateMonitorUI();
    renderTranscriptUI();
    updateEscalationUI();
  }

  function renderSessionList() {
    const calls = Object.values(state.sessions);
    const activeCalls = calls.filter(c => c.status !== 'completed' && c.status !== 'failed');
    el.queueCountBadge.textContent = activeCalls.length;
    el.concurrencyLabel.textContent = `${activeCalls.length} / 25`;

    if (calls.length === 0) {
      el.sessionList.innerHTML = '';
      el.sessionList.appendChild(el.emptyQueueState);
      return;
    }

    el.sessionList.innerHTML = '';
    // Sort active first, then newest
    calls.sort((a, b) => {
      const aDone = a.status === 'completed' || a.status === 'failed';
      const bDone = b.status === 'completed' || b.status === 'failed';
      if (aDone !== bDone) return aDone ? 1 : -1;
      return new Date(b.created_at) - new Date(a.created_at);
    });

    calls.forEach((call) => {
      const item = document.createElement('div');
      item.className = `session-item ${call.call_id === state.activeCallId ? 'active-selected' : ''}`;
      item.role = 'tab';
      item.tabIndex = 0;
      item.innerHTML = `
        <div class="session-header-row">
          <span class="session-caller-name" title="${escapeHtml(call.caller_name)}">${escapeHtml(call.caller_name)}</span>
          <span class="status-pill ${call.status}">${call.status}</span>
        </div>
        <div class="session-sub-row">
          <span class="session-id-tag">${call.caller_id}</span>
          <span>${formatTimer(call.duration_seconds || 0)}</span>
        </div>
      `;
      item.addEventListener('click', () => selectCall(call.call_id));
      item.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          selectCall(call.call_id);
        }
      });
      el.sessionList.appendChild(item);
    });
  }

  function updateMonitorUI() {
    const call = state.sessions[state.activeCallId];
    if (!call) {
      el.monitorAvatar.textContent = '--';
      el.monitorHeading.textContent = 'No Call Selected';
      el.monitorStatusPill.className = 'status-pill idle';
      el.monitorStatusPill.textContent = 'IDLE';
      el.monitorEscalationPill.classList.add('hidden');
      el.monitorCallerId.textContent = 'ID: --';
      el.monitorCallerPhone.textContent = '+1 (***) ***-****';
      el.monitorTimer.textContent = '00:00';
      el.btnMute.disabled = true;
      el.btnEscalateQuick.disabled = true;
      el.btnHangup.disabled = true;
      el.customerInput.disabled = true;
      el.btnSendMessage.disabled = true;
      return;
    }

    const isFinished = call.status === 'completed' || call.status === 'failed';
    el.monitorAvatar.textContent = call.caller_name.substring(0, 2).toUpperCase();
    el.monitorHeading.textContent = call.caller_name;
    el.monitorStatusPill.className = `status-pill ${call.status}`;
    el.monitorStatusPill.textContent = call.status;

    if (call.escalated) {
      el.monitorEscalationPill.classList.remove('hidden');
    } else {
      el.monitorEscalationPill.classList.add('hidden');
    }

    el.monitorCallerId.textContent = `ID: ${call.caller_id} (${call.call_id})`;
    el.monitorCallerPhone.textContent = call.caller_phone_masked;
    el.monitorTimer.textContent = formatTimer(call.duration_seconds || 0);

    // Mute button label
    el.muteLabel.textContent = call.is_muted ? 'Unmute' : 'Mute';

    // Controls enablement
    el.btnMute.disabled = isFinished || call.status === 'ringing';
    el.btnEscalateQuick.disabled = isFinished || call.escalated || call.status === 'ringing';
    el.btnHangup.disabled = isFinished;
    el.customerInput.disabled = isFinished || call.status === 'ringing';
    el.btnSendMessage.disabled = isFinished || call.status === 'ringing';

    // Acoustic carrier label
    if (call.status === 'speaking') {
      el.acousticStatusLabel.textContent = 'Voice Activity: LEO Synthesizing Grounded Speech';
      el.waveformIndicator.className = 'waveform-indicator active';
    } else if (call.status === 'processing') {
      el.acousticStatusLabel.textContent = 'Voice Activity: Processing Query against pgvector';
      el.waveformIndicator.className = 'waveform-indicator active';
    } else if (call.status === 'active') {
      el.acousticStatusLabel.textContent = 'Voice Activity: Carrier Connected (Duplex Ready)';
      el.waveformIndicator.className = 'waveform-indicator active';
    } else {
      el.acousticStatusLabel.textContent = `Voice Activity: Channel ${call.status}`;
      el.waveformIndicator.className = 'waveform-indicator';
    }
  }

  function renderTranscriptUI() {
    const call = state.sessions[state.activeCallId];
    if (!call || call.transcript.length === 0) {
      el.transcriptStream.innerHTML = '';
      el.transcriptStream.appendChild(el.emptyTranscriptMessage);
      return;
    }

    el.transcriptStream.innerHTML = '';
    call.transcript.forEach((turn) => {
      appendTranscriptTurnUI(turn);
    });
    el.transcriptStream.scrollTop = el.transcriptStream.scrollHeight;
  }

  function appendTranscriptTurnUI(turn) {
    const emptyNotice = el.transcriptStream.querySelector('.empty-transcript');
    if (emptyNotice) {
      emptyNotice.remove();
    }

    const bubble = document.createElement('div');
    bubble.className = `transcript-bubble ${turn.role}`;
    bubble.dataset.turnId = turn.id;

    let sourcesHtml = '';
    if (turn.sources && turn.sources.length > 0) {
      const sourcesList = turn.sources.map(s => `
        <div class="source-item">
          <div><span class="source-name">${escapeHtml(s.source_name)}</span> • <span class="source-score">Relevance: ${(s.similarity_score * 100).toFixed(1)}%</span></div>
          <div class="source-excerpt">${escapeHtml(s.content_excerpt)}</div>
        </div>
      `).join('');

      sourcesHtml = `
        <div class="turn-sources">
          <button class="source-toggle" type="button" onclick="this.nextElementSibling.classList.toggle('hidden')">
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
            ${turn.sources.length} Grounded Knowledge Passage(s)
          </button>
          <div class="sources-list hidden">${sourcesList}</div>
        </div>
      `;
    }

    let metaHtml = '';
    if (turn.metadata && (turn.metadata.model || turn.metadata.completion_tokens)) {
      metaHtml = `
        <div class="turn-metadata">
          <span>Model: ${turn.metadata.model || 'LEO-RAG'}</span>
          <span>Tokens: ${turn.metadata.prompt_tokens || 0} in / ${turn.metadata.completion_tokens || 0} out</span>
          <span>Ver: ${turn.metadata.prompt_version || '1.0.0'}</span>
        </div>
      `;
    }

    // Audio playback button for LEO turns
    let audioHtml = '';
    if (turn.role === 'leo') {
      audioHtml = `
        <div class="turn-audio" id="audio-${escapeHtml(turn.id)}">
          <button class="btn-play-audio" id="btn-play-${escapeHtml(turn.id)}" type="button"
                  title="Play LEO speech" aria-label="Play audio" style="display:none"
                  onclick="window.playTurnAudio('${escapeHtml(turn.id)}')">&#9654; Play Audio</button>
          <span class="audio-status" id="audio-status-${escapeHtml(turn.id)}"></span>
        </div>
      `;
    }

    const roleName = turn.role === 'customer' ? 'Customer' : (turn.role === 'leo' ? 'LEO AI Voice Agent' : (turn.role === 'agent' ? 'Human Specialist' : 'System'));
    const timeFormatted = new Date(turn.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    bubble.innerHTML = `
      <div class="bubble-meta">
        <span class="speaker-label ${turn.role}">${roleName}</span>
        <span class="turn-timestamp">${timeFormatted}</span>
      </div>
      <div class="turn-text">${escapeHtml(turn.text)}</div>
      ${sourcesHtml}
      ${metaHtml}
      ${audioHtml}
    `;

    el.transcriptStream.appendChild(bubble);
    el.transcriptStream.scrollTop = el.transcriptStream.scrollHeight;
  }

  function updateEscalationUI() {
    const call = state.sessions[state.activeCallId];
    if (!call || !call.escalated) {
      el.escalationEmptyNotice.classList.remove('hidden');
      el.escalationDetails.classList.add('hidden');
      return;
    }

    el.escalationEmptyNotice.classList.add('hidden');
    el.escalationDetails.classList.remove('hidden');

    el.escReasonText.textContent = (call.escalation_reason || 'Manual').replace('_', ' ');
    el.escNotesText.textContent = call.escalation_notes || 'Escalated to human operator queue.';
    el.assignedAgentName.textContent = call.assigned_agent_name || 'Unassigned (Waiting in Queue)';

    renderAgentList();
  }

  // ============================================================================
  // Agent Pool Management
  // ============================================================================
  async function fetchAgents() {
    try {
      const resp = await fetch('/api/v1/demo/agents');
      if (resp.ok) {
        state.agents = await resp.json();
        renderAgentList();
      }
    } catch (exc) {
      console.error('Fetch agents failed:', exc);
    }
  }

  function renderAgentList() {
    const call = state.sessions[state.activeCallId];
    const online = state.agents.filter(a => a.status !== 'offline').length;
    el.agentOnlineCount.textContent = `${online} Online`;

    el.agentList.innerHTML = '';
    state.agents.forEach((agent) => {
      const card = document.createElement('div');
      card.className = 'agent-card';

      const canAssign = call && call.escalated && !call.assigned_agent_id && agent.status === 'available';

      card.innerHTML = `
        <div class="agent-info">
          <span class="agent-name">${escapeHtml(agent.name)}</span>
          <span class="agent-dept">${escapeHtml(agent.department)}</span>
          <div class="agent-meta">
            <span class="agent-status-badge ${agent.status}">${agent.status}</span>
            <span class="agent-calls-count">${agent.calls_handled_today} calls</span>
          </div>
        </div>
        ${canAssign ? `<button class="btn btn-warning btn-sm btn-assign" data-agent-id="${agent.agent_id}">Take Call</button>` : ''}
      `;

      if (canAssign) {
        card.querySelector('.btn-assign').addEventListener('click', () => {
          assignAgentToCall(call.call_id, agent.agent_id);
        });
      }

      el.agentList.appendChild(card);
    });
  }

  async function assignAgentToCall(callId, agentId) {
    try {
      const resp = await fetch(`/api/v1/demo/calls/${callId}/assign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ agent_id: agentId }),
      });
      if (resp.ok) {
        const updated = await resp.json();
        state.sessions[callId] = updated;
        updateMonitorUI();
        updateEscalationUI();
        await fetchAgents();
      }
    } catch (exc) {
      console.error('Assign agent failed:', exc);
    }
  }

  // ============================================================================
  // History & Audit Table
  // ============================================================================
  async function loadHistory() {
    if (state.isFetchingHistory) return;
    state.isFetchingHistory = true;
    try {
      const query = el.historySearch.value.trim();
      const statusFilter = el.historyFilterStatus.value;
      const params = new URLSearchParams({ limit: '50' });
      if (query) params.append('search', query);
      if (statusFilter) params.append('status', statusFilter);

      const resp = await fetch(`/api/v1/demo/calls?${params.toString()}`);
      if (resp.ok) {
        const calls = await resp.json();
        renderHistoryTable(calls);
      }
    } catch (exc) {
      console.error('Load history failed:', exc);
    } finally {
      state.isFetchingHistory = false;
    }
  }

  function renderHistoryTable(calls) {
    if (calls.length === 0) {
      el.historyTableBody.innerHTML = `
        <tr>
          <td colspan="8" class="text-center py-4">No calls matching query.</td>
        </tr>
      `;
      return;
    }

    el.historyTableBody.innerHTML = calls.map((call) => `
      <tr>
        <td class="history-id-cell">${call.caller_id}</td>
        <td><strong>${escapeHtml(call.caller_name)}</strong></td>
        <td><code>${call.caller_phone_masked}</code></td>
        <td><span class="status-pill ${call.status}">${call.status}</span></td>
        <td>${formatTimer(call.duration_seconds || 0)}</td>
        <td>${call.escalated ? '<span class="status-pill warning">Escalated</span>' : '<span class="status-pill active">AI Only</span>'}</td>
        <td>${new Date(call.created_at).toLocaleTimeString()}</td>
        <td>
          <button class="btn btn-secondary btn-sm" onclick="window.inspectTranscript('${call.call_id}')">
            Inspect
          </button>
        </td>
      </tr>
    `).join('');
  }

  // Global window function for drawer inspect
  window.inspectTranscript = function (callId) {
    const call = state.sessions[callId] || null;
    if (!call) {
      // Fetch single call detail
      fetch(`/api/v1/demo/calls/${callId}`)
        .then(r => r.json())
        .then(renderDrawer)
        .catch(console.error);
    } else {
      renderDrawer(call);
    }
  };

  function renderDrawer(call) {
    el.drawerTitle.textContent = `${call.caller_name} (${call.caller_id})`;
    el.drawerSubtitle.textContent = `Status: ${call.status} • Duration: ${formatTimer(call.duration_seconds || 0)} • Escalated: ${call.escalated ? 'Yes' : 'No'}`;

    el.drawerBody.innerHTML = call.transcript.map(turn => `
      <div class="transcript-bubble ${turn.role}">
        <div class="bubble-meta">
          <span class="speaker-label ${turn.role}">${turn.role}</span>
          <span class="turn-timestamp">${new Date(turn.timestamp).toLocaleTimeString()}</span>
        </div>
        <div class="turn-text">${escapeHtml(turn.text)}</div>
      </div>
    `).join('');

    el.drawerModal.classList.remove('hidden');
  }

  // ============================================================================
  // Initial Sync & Background Probes
  // ============================================================================
  const POLLING_CONFIG = {
    SSE_ACTIVE_SYNC_INTERVAL_MS: 15000, // Safety reconciliation when SSE is connected
    FALLBACK_INTERVAL_MS: 3000,         // Active polling when SSE is disconnected/reconnecting
  };

  function scheduleNextSessionPoll() {
    clearTimeout(state.sessionPollTimer);
    if (document.hidden) {
      return;
    }
    const isSseConnected = Boolean(state.eventSource && state.eventSource.readyState === EventSource.OPEN);
    const interval = isSseConnected
      ? POLLING_CONFIG.SSE_ACTIVE_SYNC_INTERVAL_MS
      : POLLING_CONFIG.FALLBACK_INTERVAL_MS;

    state.sessionPollTimer = setTimeout(async () => {
      await loadActiveSessions();
      scheduleNextSessionPoll();
    }, interval);
  }

  function cleanupTimersAndConnections() {
    clearTimeout(state.sessionPollTimer);
    clearInterval(state.healthPollTimer);
    clearInterval(state.callTimerInterval);
    clearInterval(state.incomingCountdownTimer);
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
  }

  async function loadActiveSessions() {
    if (state.isFetchingSessions) return;
    state.isFetchingSessions = true;
    try {
      const resp = await fetch('/api/v1/demo/calls?limit=25');
      if (resp.ok) {
        const calls = await resp.json();
        let activeChanged = false;
        calls.forEach(c => {
          const prev = state.sessions[c.call_id];
          state.sessions[c.call_id] = c;
          if (c.call_id === state.activeCallId) {
            if (!prev || prev.status !== c.status || prev.transcript.length !== c.transcript.length || prev.escalated !== c.escalated) {
              activeChanged = true;
            }
          }
        });
        renderSessionList();
        if (calls.length > 0 && !state.activeCallId) {
          selectCall(calls[0].call_id);
        } else if (activeChanged) {
          updateMonitorUI();
          renderTranscriptUI();
          updateEscalationUI();
        }
      }
    } catch (exc) {
      console.error('Load active sessions error:', exc);
    } finally {
      state.isFetchingSessions = false;
    }
  }


  async function checkHealth() {
    try {
      const resp = await fetch('/health/live');
      if (resp.ok) {
        el.healthLabel.textContent = 'Online';
        el.healthLabel.style.color = '#10b981';
      } else {
        el.healthLabel.textContent = 'Degraded';
        el.healthLabel.style.color = '#ef4444';
      }
    } catch {
      el.healthLabel.textContent = 'Offline';
      el.healthLabel.style.color = '#ef4444';
    }
  }

  async function fetchMetrics() {
    try {
      const resp = await fetch('/api/v1/demo/metrics');
      if (resp.ok) {
        const m = await resp.json();
        state.metrics = m;
        el.statTotal.textContent = m.total_calls;
        el.statActive.textContent = m.active_calls;
        el.statCompleted.textContent = m.completed_calls;
        el.statEscalated.textContent = m.escalated_calls;

        const resolvedCount = m.completed_calls;
        if (resolvedCount > 0) {
          const aiRate = Math.round((m.ai_handled_calls / resolvedCount) * 100);
          el.statAiRate.textContent = `${aiRate}%`;
        } else {
          el.statAiRate.textContent = '100%';
        }
      }
    } catch (exc) {
      console.error('Fetch metrics error:', exc);
    }
  }

  // ============================================================================
  // Waveform Canvas Audio Simulation
  // ============================================================================
  function initWaveform() {
    const canvas = el.waveformCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let step = 0;

    function render() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const width = canvas.width;
      const height = canvas.height;
      const centerY = height / 2;

      // Draw horizontal baseline
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
      ctx.lineWidth = 1;
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();

      // Dynamic sine wave
      ctx.beginPath();
      const amplitude = state.audioActive ? 14 : 2.5;
      const freq = state.audioActive ? 0.04 : 0.015;
      const speed = state.audioActive ? 0.12 : 0.03;

      ctx.strokeStyle = state.audioActive ? '#0ea5e9' : 'rgba(14, 165, 233, 0.35)';
      ctx.lineWidth = state.audioActive ? 2 : 1;

      for (let x = 0; x < width; x++) {
        const y = centerY + Math.sin(x * freq + step) * amplitude * Math.sin((x / width) * Math.PI);
        if (x === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();

      // Secondary harmonised harmonic wave
      if (state.audioActive) {
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
        ctx.lineWidth = 1.5;
        for (let x = 0; x < width; x++) {
          const y = centerY + Math.sin(x * (freq * 1.5) - step * 1.2) * (amplitude * 0.6) * Math.sin((x / width) * Math.PI);
          if (x === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }

      step += speed;
      state.waveformAnimationId = requestAnimationFrame(render);
    }

    render();
  }

  function triggerAudioActivity(durationMs) {
    state.audioActive = true;
    setTimeout(() => {
      state.audioActive = false;
    }, durationMs);
  }

  // ============================================================================
  // Audio Playback Engine
  // ============================================================================

  async function fetchTTSConfig() {
    try {
      const resp = await fetch('/api/v1/demo/tts/config');
      if (resp.ok) {
        const cfg = await resp.json();
        state.ttsEnabled = cfg.enabled === true;
        if (state.ttsEnabled) {
          console.info('[LEO Audio] TTS enabled:', cfg.provider, cfg.model, cfg.voice);
        } else {
          console.info('[LEO Audio] TTS provider is mock — no real speech will be generated.');
        }
      }
    } catch (e) {
      console.warn('[LEO Audio] Failed to fetch TTS config:', e);
    }
  }

  function getOrCreateAudioContext() {
    if (!state.audioCtx || state.audioCtx.state === 'closed') {
      state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (state.audioCtx.state === 'suspended') {
      state.audioCtx.resume();
    }
    return state.audioCtx;
  }

  function stopCurrentAudio() {
    // 1. Cancel in-flight stream reader
    if (state.streamReader) {
      try { state.streamReader.cancel(); } catch (_) { }
      state.streamReader = null;
    }
    // 2. Stop all scheduled streaming PCM nodes
    if (state.activeStreamSources && state.activeStreamSources.length > 0) {
      state.activeStreamSources.forEach(s => {
        try { s.stop(); } catch (_) { }
      });
      state.activeStreamSources = [];
    }
    // 3. Stop single buffer audio source if any
    if (state.currentAudioSource) {
      try { state.currentAudioSource.stop(); } catch (_) { }
      state.currentAudioSource = null;
    }
    state.currentPlayToken = null;
    state.audioActive = false;
    state.nextChunkPlayTime = 0;
  }

  async function playTurnAudioStream(callId, turn) {
    let audioCtx;
    try {
      audioCtx = getOrCreateAudioContext();
    } catch (e) {
      return false;
    }

    const ttsStart = performance.now();
    setAudioStatus(turn.id, '⚡ Streaming speech…', false);

    const resp = await fetch(`/api/v1/demo/calls/${callId}/turns/${turn.id}/audio-stream`, {
      method: 'POST',
    });

    if (!resp.ok) {
      throw new Error(`Stream endpoint returned HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    state.streamReader = reader;
    state.activeStreamSources = [];
    state.nextChunkPlayTime = audioCtx.currentTime;

    let sampleRate = 24000;
    let buffer = new Uint8Array(0);
    let firstAudioPlayed = false;

    function appendToBuffer(chunk) {
      const merged = new Uint8Array(buffer.length + chunk.length);
      merged.set(buffer, 0);
      merged.set(chunk, buffer.length);
      buffer = merged;
    }

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Check if turn was superseded or barge-in triggered
        if (state.currentPlayToken !== turn.id) {
          try { reader.cancel(); } catch (_) {}
          return true;
        }

        appendToBuffer(value);

        // Unpack framed protocol: [4-byte length][1-byte type][payload]
        while (buffer.length >= 5) {
          const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
          const frameLen = view.getUint32(0, false);
          if (buffer.length < 4 + frameLen) {
            break; // Wait for more bytes
          }

          const frameType = buffer[4];
          const payload = buffer.slice(5, 4 + frameLen);
          buffer = buffer.slice(4 + frameLen);

          if (frameType === 0x01) {
            // Header frame: sample_rate (uint32), channels (uint32)
            if (payload.length >= 8) {
              const hView = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
              sampleRate = hView.getUint32(0, false) || 24000;
            }
          } else if (frameType === 0x02) {
            // Raw PCM 16-bit little-endian samples
            if (payload.length >= 2 && state.currentPlayToken === turn.id) {
              const numSamples = Math.floor(payload.length / 2);
              const pcm16 = new Int16Array(payload.buffer, payload.byteOffset, numSamples);

              const audioBuf = audioCtx.createBuffer(1, numSamples, sampleRate);
              const channelData = audioBuf.getChannelData(0);
              for (let i = 0; i < numSamples; i++) {
                channelData[i] = pcm16[i] / 32768.0;
              }

              const sourceNode = audioCtx.createBufferSource();
              sourceNode.buffer = audioBuf;
              sourceNode.connect(audioCtx.destination);
              state.activeStreamSources.push(sourceNode);

              const now = audioCtx.currentTime;
              const scheduledStart = Math.max(now + 0.02, state.nextChunkPlayTime);
              sourceNode.start(scheduledStart);
              state.nextChunkPlayTime = scheduledStart + audioBuf.duration;

              if (!firstAudioPlayed) {
                firstAudioPlayed = true;
                const ttfaMs = performance.now() - ttsStart;
                console.info(`[TIMING] ⚡ FIRST AUDIBLE RESPONSE in ${ttfaMs.toFixed(1)}ms! (turnId=${turn.id})`);
                triggerAudioActivity(4000);
                setAudioStatus(turn.id, '🔊 Speaking…', false);
              }
            }
          } else if (frameType === 0x04) {
            // Done frame
            break;
          } else if (frameType === 0x05) {
            const errMsg = new TextDecoder().decode(payload);
            throw new Error(errMsg || "Streaming error");
          }
        }
      }

      if (firstAudioPlayed) {
        const remainingMs = Math.max(0, (state.nextChunkPlayTime - audioCtx.currentTime) * 1000);
        setTimeout(() => {
          if (state.currentPlayToken === turn.id) {
            setAudioStatus(turn.id, '✓ Played', false);
            state.audioActive = false;
          }
        }, remainingMs + 100);
        return true;
      }
      return false;
    } finally {
      state.streamReader = null;
    }
  }

  async function synthesizeAndPlayTurnWav(callId, turn) {
    const ttsStart = performance.now();
    const resp = await fetch(`/api/v1/demo/calls/${callId}/turns/${turn.id}/audio`, {
      method: 'POST',
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      const msg = err.detail || `HTTP ${resp.status}`;
      setAudioStatus(turn.id, `⚠️ Audio failed: ${msg}`, true, turn.id, callId);
      return;
    }

    if (state.currentPlayToken !== turn.id) {
      setAudioStatus(turn.id, '', false);
      return;
    }

    const firstChunkMs = performance.now() - ttsStart;
    const arrayBuffer = await resp.arrayBuffer();

    let audioCtx;
    try {
      audioCtx = getOrCreateAudioContext();
    } catch (e) {
      setAudioStatus(turn.id, '⚠️ Web Audio API not available', true, turn.id, callId);
      return;
    }

    const decodeStart = performance.now();
    let audioBuffer;
    try {
      audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
    } catch (decodeErr) {
      console.error('[LEO Audio] Failed to decode WAV:', decodeErr);
      setAudioStatus(turn.id, '⚠️ Failed to decode audio', true, turn.id, callId);
      return;
    }
    const decodeMs = performance.now() - decodeStart;

    if (state.currentPlayToken !== turn.id) {
      setAudioStatus(turn.id, '', false);
      return;
    }

    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.destination);
    state.currentAudioSource = source;

    const durationMs = Math.round(audioBuffer.duration * 1000);
    triggerAudioActivity(durationMs + 300);
    setAudioStatus(turn.id, '🔊 Playing…', false);

    source.onended = () => {
      if (state.currentPlayToken === turn.id) {
        state.currentAudioSource = null;
        state.currentPlayToken = null;
        state.audioActive = false;
      }
      setAudioStatus(turn.id, '✓ Played', false);
    };

    try {
      source.start(0);
      const playbackStartMs = performance.now() - ttsStart;
      console.info(`[TIMING] Audio playback started in ${playbackStartMs.toFixed(1)}ms (audioFetch=${firstChunkMs.toFixed(1)}ms, decode=${decodeMs.toFixed(1)}ms, turnId=${turn.id})`);
    } catch (playErr) {
      console.warn('[LEO Audio] Autoplay blocked:', playErr);
      setAudioStatus(turn.id, '', true, turn.id, callId);
      state.currentAudioSource = null;
      state.currentPlayToken = null;
      state.audioActive = false;
    }
  }

  /**
   * Ultra-low-latency live-stream: immediately opens LLM→TTS pipeline.
   * Returns true if audio was successfully streamed, false on error/skip.
   */
  async function playLiveStream(callId, messageText, playToken) {
    let audioCtx;
    try {
      audioCtx = getOrCreateAudioContext();
    } catch (e) {
      return false;
    }

    const liveStart = performance.now();
    // Create a temporary LEO turn bubble to show live transcript
    const liveTurnId = 'live_' + playToken;
    let liveBubble = null;
    let liveBubbleText = null;
    let accumulatedText = '';
    let turnIdFromServer = null;

    function getOrCreateLiveBubble() {
      if (liveBubble) return liveBubble;
      // Remove empty transcript placeholder
      const emptyNotice = el.transcriptStream.querySelector('.empty-transcript');
      if (emptyNotice) emptyNotice.remove();

      liveBubble = document.createElement('div');
      liveBubble.className = 'transcript-bubble leo';
      liveBubble.dataset.turnId = liveTurnId;
      liveBubble.innerHTML = `
        <div class="bubble-meta">
          <span class="speaker-label leo">LEO AI Voice Agent</span>
          <span class="turn-timestamp">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
        </div>
        <div class="turn-text" id="live-text-${liveTurnId}"></div>
        <div class="turn-audio" id="audio-${liveTurnId}">
          <span class="audio-status" id="audio-status-${liveTurnId}">⚡ Streaming speech…</span>
        </div>
      `;
      el.transcriptStream.appendChild(liveBubble);
      el.transcriptStream.scrollTop = el.transcriptStream.scrollHeight;
      liveBubbleText = document.getElementById('live-text-' + liveTurnId);
      return liveBubble;
    }

    const resp = await fetch(`/api/v1/demo/calls/${callId}/live-stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: messageText }),
    });

    if (!resp.ok) {
      console.warn('[LEO Live] live-stream endpoint returned HTTP', resp.status);
      return false;
    }

    const reader = resp.body.getReader();
    state.streamReader = reader;
    state.activeStreamSources = [];
    state.nextChunkPlayTime = audioCtx.currentTime;

    let sampleRate = 24000;
    let buffer = new Uint8Array(0);
    let firstAudioPlayed = false;
    let headerReceived = false;

    function appendToBuffer(chunk) {
      const merged = new Uint8Array(buffer.length + chunk.length);
      merged.set(buffer, 0);
      merged.set(chunk, buffer.length);
      buffer = merged;
    }

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Barge-in: stop if token was superseded
        if (state.currentPlayToken !== playToken) {
          try { reader.cancel(); } catch (_) {}
          return true;
        }

        appendToBuffer(value);

        // Parse framed protocol: [4-byte length][1-byte type][payload]
        while (buffer.length >= 5) {
          const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
          const frameLen = view.getUint32(0, false);
          if (buffer.length < 4 + frameLen) break;

          const frameType = buffer[4];
          const payload = buffer.slice(5, 4 + frameLen);
          buffer = buffer.slice(4 + frameLen);

          if (frameType === 0x03) {
            // TEXT frame: incremental transcript sentence
            const sentence = new TextDecoder().decode(payload);
            accumulatedText += (accumulatedText ? ' ' : '') + sentence;
            getOrCreateLiveBubble();
            if (liveBubbleText) {
              liveBubbleText.textContent = accumulatedText;
              el.transcriptStream.scrollTop = el.transcriptStream.scrollHeight;
            }
            const textMs = performance.now() - liveStart;
            console.info(`[TIMING] ⚡ FIRST TEXT SENTENCE in ${textMs.toFixed(1)}ms`);

          } else if (frameType === 0x01) {
            // HEADER frame: sample_rate + channels
            if (!headerReceived && payload.length >= 8) {
              const hView = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
              sampleRate = hView.getUint32(0, false) || 24000;
              headerReceived = true;
            }

          } else if (frameType === 0x02) {
            // PCM frame: schedule audio playback
            if (payload.length >= 2 && state.currentPlayToken === playToken) {
              const numSamples = Math.floor(payload.length / 2);
              const pcm16 = new Int16Array(payload.buffer, payload.byteOffset, numSamples);

              const audioBuf = audioCtx.createBuffer(1, numSamples, sampleRate);
              const channelData = audioBuf.getChannelData(0);
              for (let i = 0; i < numSamples; i++) {
                channelData[i] = pcm16[i] / 32768.0;
              }

              const sourceNode = audioCtx.createBufferSource();
              sourceNode.buffer = audioBuf;
              sourceNode.connect(audioCtx.destination);
              state.activeStreamSources.push(sourceNode);

              const now = audioCtx.currentTime;
              const scheduledStart = Math.max(now + 0.02, state.nextChunkPlayTime);
              sourceNode.start(scheduledStart);
              state.nextChunkPlayTime = scheduledStart + audioBuf.duration;

              if (!firstAudioPlayed) {
                firstAudioPlayed = true;
                const ttfaMs = performance.now() - liveStart;
                console.info(`[TIMING] ⚡⚡ FIRST AUDIBLE LIVE RESPONSE in ${ttfaMs.toFixed(1)}ms! (callId=${callId})`);
                triggerAudioActivity(4000);
                getOrCreateLiveBubble();
                const statusEl = document.getElementById('audio-status-' + liveTurnId);
                if (statusEl) statusEl.textContent = '🔊 Speaking…';
              }
            }

          } else if (frameType === 0x04) {
            // DONE frame
            break;

          } else if (frameType === 0x05) {
            // ERROR frame
            const errMsg = new TextDecoder().decode(payload);
            console.warn('[LEO Live] Stream error frame:', errMsg);
            break;
          }
        }
      }

      if (firstAudioPlayed) {
        const remainingMs = Math.max(0, (state.nextChunkPlayTime - audioCtx.currentTime) * 1000);
        setTimeout(() => {
          if (state.currentPlayToken === playToken) {
            state.audioActive = false;
          }
          const statusEl = document.getElementById('audio-status-' + liveTurnId);
          if (statusEl) statusEl.textContent = '✓ Played';
        }, remainingMs + 100);
        return true;
      }

      return false;
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.warn('[LEO Live] Stream read error:', err);
      }
      return false;
    } finally {
      state.streamReader = null;
    }
  }

  async function synthesizeAndPlayTurn(callId, turn) {
    if (!state.ttsEnabled) return;          // Skip for mock provider
    if (!turn.text || !turn.text.trim()) return;
    if (state.audioSynthesizing[turn.id]) return; // Already in-flight

    // If a live-stream is already playing for the active call, skip duplicate synthesis
    // (the live-stream pre-empted the SSE-triggered path)
    if (state.activeStreamSources && state.activeStreamSources.length > 0) {
      // Live stream is in progress — ensure SSE turn bubble matches but skip TTS
      state.audioSynthesizing[turn.id] = true;
      delete state.audioSynthesizing[turn.id];
      return;
    }

    state.audioSynthesizing[turn.id] = true;
    setAudioStatus(turn.id, '⚡ synthesizing…', false);

    // Stop any currently playing LEO speech (new turn preempts previous)
    stopCurrentAudio();
    state.currentPlayToken = turn.id;

    try {
      // 1. Try progressive streaming first (audio begins in ~1.1s)
      try {
        const streamed = await playTurnAudioStream(callId, turn);
        if (streamed) return;
      } catch (streamErr) {
        console.warn('[LEO Audio] Progressive stream failed, falling back to full WAV:', streamErr);
      }

      // 2. Safe Fallback: non-streaming WAV
      if (state.currentPlayToken !== turn.id) return;
      await synthesizeAndPlayTurnWav(callId, turn);

    } catch (err) {
      console.error('[LEO Audio] synthesis error:', err);
      setAudioStatus(turn.id, `⚠️ Error: ${err.message}`, true, turn.id, callId);
    } finally {
      delete state.audioSynthesizing[turn.id];
    }
  }

  // Expose playTurnAudio globally for the Play button onclick attribute
  window.playTurnAudio = async function (turnId) {
    const call = state.sessions[state.activeCallId];
    if (!call) return;
    const turn = call.transcript.find(t => t.id === turnId);
    if (!turn) return;
    stopCurrentAudio();
    delete state.audioSynthesizing[turnId];
    await synthesizeAndPlayTurn(state.activeCallId, turn);
  };

  function setAudioStatus(turnId, message, showPlayBtn, playTurnId, callId) {
    const statusEl = document.getElementById(`audio-status-${turnId}`);
    const playBtn = document.getElementById(`btn-play-${turnId}`);
    if (statusEl) statusEl.textContent = message || '';
    if (playBtn) {
      playBtn.style.display = showPlayBtn ? 'inline-flex' : 'none';
    }
  }

  function startCallTimer() {
    clearInterval(state.callTimerInterval);
    state.callTimerInterval = setInterval(() => {
      Object.values(state.sessions).forEach((call) => {
        if (call.status === 'active' || call.status === 'processing' || call.status === 'speaking' || call.status === 'transferring') {
          call.duration_seconds = (call.duration_seconds || 0) + 1;
        }
      });
      const current = state.sessions[state.activeCallId];
      if (current) {
        el.monitorTimer.textContent = formatTimer(current.duration_seconds || 0);
      }
    }, 1000);
  }

  // ============================================================================
  // Helpers
  // ============================================================================
  function formatTimer(totalSeconds) {
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Lifecycle & tab visibility listeners
  window.addEventListener('beforeunload', cleanupTimersAndConnections);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      loadActiveSessions();
      scheduleNextSessionPoll();
    } else {
      clearTimeout(state.sessionPollTimer);
    }
  });

  // Expose dashboard handle globally for testing/observability
  window.__LEO_DASHBOARD__ = {
    getState: () => state,
    scheduleNextSessionPoll,
    loadActiveSessions,
    loadHistory,
    cleanup: cleanupTimersAndConnections,
  };

  // Run on DOM Ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
