#!/usr/bin/env python3
"""Backend patch for LEO. Run from the repo root (D:\\leo-ai-backend):

    python apply_backend_patch.py

Patches app/ai/orchestrator.py and app/voice/gemini_voice.py.
All-or-nothing: if ANY anchor does not match exactly once, nothing is written.
Backups are saved as <file>.bak.
"""
import shutil, sys, pathlib

ORCH = "app/ai/orchestrator.py"
VOICE = "app/voice/gemini_voice.py"

ORCH_OLD = '''    t_gen_start = time.perf_counter()
    ttft_ms: float | None = None

    # Try streaming to capture TTFT and stream output if provider supports it
    if hasattr(llm, "generate_stream"):
        try:
            chunks: list[str] = []
            async for chunk in llm.generate_stream(gen_request):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_gen_start) * 1000.0
                chunks.append(chunk)

            if chunks:
                full_content = "".join(chunks)
                gen_response = GenerationResponse(
                    content=full_content,
                    model=getattr(llm, "_model", getattr(gen_request, "model", "default")),
                    prompt_tokens=0,
                    completion_tokens=len(full_content.split()),
                    finish_reason="stop",
                    provider=getattr(llm, "provider_name", "unknown"),
                )
            else:
                gen_response = await llm.generate(gen_request)
        except Exception:
            gen_response = await llm.generate(gen_request)
    else:
        gen_response = await llm.generate(gen_request)
'''

ORCH_NEW = '''    t_gen_start = time.perf_counter()
    ttft_ms: float | None = None  # not measured: generation is one non-streaming call

    # Exactly ONE provider call per turn.
    # Previously this block streamed the reply, hit a NameError (GenerationResponse was
    # never imported), swallowed it with `except Exception`, and silently called
    # generate() a second time: double latency, double quota, and a second full
    # deadline on failures. The provider owns retry/backoff under a single total
    # deadline (llm_timeout_seconds); errors now propagate to the caller untouched.
    gen_response = await llm.generate(gen_request)
'''

VOICE_OLD = '''        for model_name in candidates:
            try:
                async with asyncio.timeout(min(self._timeout_seconds, 15.0)):
'''

VOICE_NEW = '''        # One total deadline across ALL candidate models (previously each candidate got
        # its own 15s window, so 4 models could burn ~60s). A single attempt may use up
        # to _TTS_ATTEMPT_CAP_S so a slow-but-working synthesis is not cut off at 15s.
        _TTS_ATTEMPT_CAP_S = 25.0
        tts_deadline = time.perf_counter() + float(self._timeout_seconds)
        for model_name in candidates:
            remaining = tts_deadline - time.perf_counter()
            if remaining <= 1.0:
                logger.warning("tts_deadline_exhausted", last_model=model_name)
                break
            try:
                async with asyncio.timeout(min(remaining, _TTS_ATTEMPT_CAP_S)):
'''

EDITS = {ORCH: [(ORCH_OLD, ORCH_NEW)], VOICE: [(VOICE_OLD, VOICE_NEW)]}

def main():
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(".")
    new_src, problems, crlf = {}, [], {}
    for rel, edits in EDITS.items():
        p = root / rel
        if not p.exists():
            problems.append(f"{rel}: file not found"); continue
        crlf[rel] = b"\r\n" in p.read_bytes()   # preserve Windows line endings
        src = p.read_text(encoding="utf-8")
        for old, new in edits:
            n = src.count(old)
            if n != 1:
                problems.append(f"{rel}: expected 1 match, found {n} for: {old.strip().splitlines()[0][:60]!r}")
            else:
                src = src.replace(old, new)
        new_src[rel] = src
    if problems:
        print("NOT APPLIED (your files differ from the versions this patch targets):")
        print("\n".join("  " + x for x in problems)); sys.exit(1)
    for rel, src in new_src.items():
        p = root / rel
        shutil.copyfile(p, str(p) + ".bak")
        p.write_text(src, encoding="utf-8", newline="\r\n" if crlf[rel] else "\n")
        print(f"patched {rel} (backup: {rel}.bak)")

if __name__ == "__main__":
    main()