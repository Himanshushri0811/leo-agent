"""
Multilingual Detection, Matching, and Session Language Context Policy for LEO Voice Agent.

Supports:
- Automatic script and vocabulary identification (Hindi, English, Hinglish, Tamil, Telugu, Kannada, Malayalam).
- Context retention across multi-turn sessions (no session leaks, hysteresis against brief borrowed terms).
- Code-switching and technical loanword preservation.
- Language capability matrix for verified TTS and ASR mappings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class SupportedLanguage(StrEnum):
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hinglish"
    TAMIL = "ta"
    TELUGU = "te"
    KANNADA = "kn"
    MALAYALAM = "ml"


# Verified locale tags for ASR & TTS providers
LANGUAGE_LOCALE_MAP: dict[SupportedLanguage, str] = {
    SupportedLanguage.ENGLISH: "en-IN",
    SupportedLanguage.HINDI: "hi-IN",
    SupportedLanguage.HINGLISH: "en-IN",  # Or hi-IN depending on model
    SupportedLanguage.TAMIL: "ta-IN",
    SupportedLanguage.TELUGU: "te-IN",
    SupportedLanguage.KANNADA: "kn-IN",
    SupportedLanguage.MALAYALAM: "ml-IN",
}

# Unicode script ranges for Indian languages
_SCRIPT_PATTERNS = {
    SupportedLanguage.HINDI: re.compile(r"[\u0900-\u097F]"),
    SupportedLanguage.TAMIL: re.compile(r"[\u0B80-\u0BFF]"),
    SupportedLanguage.TELUGU: re.compile(r"[\u0C00-\u0C7F]"),
    SupportedLanguage.KANNADA: re.compile(r"[\u0C80-\u0CFF]"),
    SupportedLanguage.MALAYALAM: re.compile(r"[\u0D00-\u0D7F]"),
}

# High-frequency Hinglish marker words (romanized Hindi function words)
_HINGLISH_MARKERS = {
    "namaste", "kya", "kaise", "kaun", "kaunsa", "kaunsi", "kaunse", "kab", "kyun",
    "kahan", "kidhar", "kripya", "aap", "hum", "mera", "meri", "mere", "apka", "aapka",
    "aapki", "aapke", "chahiye", "chahta", "chahti", "chahte", "hota", "hoti", "hote",
    "hoga", "hogi", "hain", "hai", "tha", "thi", "the", "kar", "karta", "karti", "karte",
    "karein", "karo", "karna", "karne", "kiya", "kiye", "dena", "dijiye", "bataiye",
    "batao", "bataye", "batana", "shukriya", "dhanyawad", "sahayata", "madad", "sakte",
    "sakti", "sakta", "sakenge", "nahin", "nahi", "accha", "theek", "bahut", "thoda",
    "kuch", "sab", "yeh", "woh", "mujhe", "tumhe", "apne", "baare", "mein", "par",
    "se", "si", "sa", "ko", "ka", "ki", "ke", "aur", "ya", "lekin", "bhi", "toh",
    "kis", "kisko", "kiske", "kiski", "paas", "saath", "samajh"
}

# Common borrowed enterprise words that should NOT cause a language switch
_BORROWED_TECHNICAL_TERMS = {
    "cloud", "security", "support", "services", "sla", "database", "incident",
    "devops", "backup", "architecture", "consulting", "server", "network",
    "api", "login", "password", "agent", "ticket", "okay", "yes", "no", "thanks", "thank",
    "sure", "provide", "offer", "help"
}


@dataclass
class LanguageDetectionResult:
    detected_language: SupportedLanguage
    locale_code: str
    confidence: float
    is_code_switched: bool
    script: str  # "devanagari", "latin", "tamil", "telugu", "kannada", "malayalam"
    reason: str


class LanguageDetector:
    """Robust, deterministic language identification tailored for Indian voice agents."""

    @classmethod
    def detect_utterance(
        cls,
        text: str,
        session_language: SupportedLanguage | None = None,
    ) -> LanguageDetectionResult:
        clean = (text or "").strip()
        if not clean:
            default_lang = session_language or SupportedLanguage.ENGLISH
            return LanguageDetectionResult(
                detected_language=default_lang,
                locale_code=LANGUAGE_LOCALE_MAP[default_lang],
                confidence=0.5,
                is_code_switched=False,
                script="latin",
                reason="empty_input_fallback",
            )

        # 1. Check native Indian scripts
        for lang, pattern in _SCRIPT_PATTERNS.items():
            matches = pattern.findall(clean)
            if len(matches) >= 2 or (len(matches) == 1 and len(clean) < 6):
                script_name = lang.value
                return LanguageDetectionResult(
                    detected_language=lang,
                    locale_code=LANGUAGE_LOCALE_MAP[lang],
                    confidence=0.95,
                    is_code_switched=bool(re.search(r"[a-zA-Z]", clean)),
                    script=script_name,
                    reason=f"native_script_{script_name}",
                )

        # 2. Latin script text: check Hinglish vs English
        # Normalize hyphens and slashes to spaces so compound tokens like 'kaun-kaun' split
        normalized_clean = re.sub(r"[-/]", " ", clean)
        words = [w.strip(".,!?:;\"'()[]{}").lower() for w in normalized_clean.split()]
        words = [w for w in words if w]

        hinglish_count = sum(1 for w in words if w in _HINGLISH_MARKERS)
        borrowed_count = sum(1 for w in words if w in _BORROWED_TECHNICAL_TERMS)
        total_words = len(words)

        # If previous session language was a regional language or Hinglish and current input is short or pure loanwords/acknowledgments
        if session_language is not None and session_language != SupportedLanguage.ENGLISH:
            short_ack = {"ok", "okay", "yes", "yeah", "yep", "sure", "thanks", "thank", "theek", "accha", "ha", "haan"}
            if total_words <= 3 and all(w in short_ack or w in _BORROWED_TECHNICAL_TERMS for w in words):
                return LanguageDetectionResult(
                    detected_language=session_language,
                    locale_code=LANGUAGE_LOCALE_MAP[session_language],
                    confidence=0.85,
                    is_code_switched=True,
                    script="latin" if session_language == SupportedLanguage.HINGLISH else session_language.value,
                    reason="session_continuity_acknowledgment",
                )
            if session_language in (SupportedLanguage.HINDI, SupportedLanguage.HINGLISH):
                if total_words <= 3 and (borrowed_count > 0 or hinglish_count > 0):
                    return LanguageDetectionResult(
                        detected_language=session_language,
                        locale_code=LANGUAGE_LOCALE_MAP[session_language],
                        confidence=0.85,
                        is_code_switched=True,
                        script="latin",
                        reason="session_continuity_loanwords",
                    )

        # Hinglish detection: at least one marker word and meaningful fraction of sentence
        if hinglish_count >= 1:
            ratio = hinglish_count / max(1, (total_words - borrowed_count))
            if ratio >= 0.2 or hinglish_count >= 2:
                return LanguageDetectionResult(
                    detected_language=SupportedLanguage.HINGLISH,
                    locale_code=LANGUAGE_LOCALE_MAP[SupportedLanguage.HINGLISH],
                    confidence=min(0.95, 0.6 + 0.15 * hinglish_count),
                    is_code_switched=True,
                    script="latin",
                    reason="hinglish_markers_detected",
                )

        # Default to English
        return LanguageDetectionResult(
            detected_language=SupportedLanguage.ENGLISH,
            locale_code=LANGUAGE_LOCALE_MAP[SupportedLanguage.ENGLISH],
            confidence=0.90,
            is_code_switched=False,
            script="latin",
            reason="english_latin_grammar",
        )


def resolve_response_language(
    user_text: str,
    session_metadata: dict[str, Any],
) -> tuple[SupportedLanguage, str]:
    """
    Authoritative per-session language resolution.
    Returns (SupportedLanguage, locale_code) and updates session_metadata.
    """
    prev_lang_str = session_metadata.get("detected_language")
    prev_lang: SupportedLanguage | None = None
    if prev_lang_str:
        try:
            prev_lang = SupportedLanguage(prev_lang_str)
        except ValueError:
            prev_lang = None

    result = LanguageDetector.detect_utterance(user_text, session_language=prev_lang)
    session_metadata["detected_language"] = result.detected_language.value
    session_metadata["language_locale"] = result.locale_code
    session_metadata["language_script"] = result.script
    session_metadata["is_code_switched"] = result.is_code_switched

    logger.info(
        "language_resolved_for_turn",
        detected_language=result.detected_language.value,
        locale=result.locale_code,
        confidence=result.confidence,
        reason=result.reason,
    )

    return result.detected_language, result.locale_code
