"""Unit tests for automatic language matching, script detection, and session context."""

from app.voice.language import (
    LanguageDetector,
    SupportedLanguage,
    resolve_response_language,
)


def test_detect_hindi_devanagari():
    res = LanguageDetector.detect_utterance("नमस्ते, मुझे बिटमैक्स ग्रुप की सेवाओं के बारे में जानना है।")
    assert res.detected_language == SupportedLanguage.HINDI
    assert res.locale_code == "hi-IN"
    assert res.script == "hi"


def test_detect_hinglish():
    res = LanguageDetector.detect_utterance("Global Projects mein project services ke liye kya options hain?")
    assert res.detected_language == SupportedLanguage.HINGLISH
    assert res.locale_code == "en-IN"
    assert res.is_code_switched is True


def test_detect_english():
    res = LanguageDetector.detect_utterance("What core project management services does Global Projects offer?")
    assert res.detected_language == SupportedLanguage.ENGLISH
    assert res.locale_code == "en-IN"


def test_detect_south_indian_scripts():
    # Tamil (CRITICAL: preserve existing Tamil detection)
    ta = LanguageDetector.detect_utterance("வணக்கம், நீங்கள் என்ன சேவைகளை வழங்குகிறீர்கள்?")
    assert ta.detected_language == SupportedLanguage.TAMIL
    assert ta.locale_code == "ta-IN"

    # Telugu
    te = LanguageDetector.detect_utterance("నమస్కారం, మీరు ఏ సేవలను అందిస్తారు?")
    assert te.detected_language == SupportedLanguage.TELUGU
    assert te.locale_code == "te-IN"


def test_detect_bengali():
    bn = LanguageDetector.detect_utterance("নমস্কার, গ্লোবাল প্রজেক্টস কী ধরণের সেবা প্রদান করে?")
    assert bn.detected_language == SupportedLanguage.BENGALI
    assert bn.locale_code == "bn-IN"


def test_session_language_retention_tamil():
    # Caller speaks Tamil
    session_meta = {"detected_language": "ta", "language_locale": "ta-IN"}
    # Caller sends short acknowledgment loanword
    lang, locale = resolve_response_language("okay thanks", session_meta)
    assert lang == SupportedLanguage.TAMIL
    assert locale == "ta-IN"


def test_session_language_retention_on_borrowed_loanwords():
    # User was speaking Hindi/Hinglish
    session_meta = {"detected_language": "hinglish", "language_locale": "en-IN"}

    # User replies with short borrowed technical word
    lang, locale = resolve_response_language("Okay cloud SLA kya hai?", session_meta)
    assert lang == SupportedLanguage.HINGLISH
    assert session_meta["detected_language"] == "hinglish"

    # Short loanword confirmation should retain Hindi/Hinglish session
    lang2, _ = resolve_response_language("Okay thanks", session_meta)
    assert lang2 == SupportedLanguage.HINGLISH
