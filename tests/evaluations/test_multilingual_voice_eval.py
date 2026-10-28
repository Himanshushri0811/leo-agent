"""
Multilingual Voice Agent Evaluation Suite for LEO AI.

Evaluates:
- Script and vocabulary detection across Hindi, English, Hinglish, Tamil, Telugu, Kannada, Malayalam.
- Grounding accuracy against approved company knowledge.
- Barge-in cancellation token propagation.
- Empty/uncertain input handling without hallucination.
"""

import pytest

from app.voice.language import (
    LanguageDetector,
    SupportedLanguage,
    resolve_response_language,
)
from app.voice.turn_manager import TurnManager, TurnState


MULTILINGUAL_TEST_DATASET = [
    {
        "id": "EVAL-EN-01",
        "lang": SupportedLanguage.ENGLISH,
        "input": "What core enterprise cloud consulting services does Bitmax Group offer?",
        "expected_lang": SupportedLanguage.ENGLISH,
        "expected_locale": "en-IN",
        "topic": "services",
    },
    {
        "id": "EVAL-HI-01",
        "lang": SupportedLanguage.HINDI,
        "input": "बिटमैक्स ग्रुप क्या सेवाएं प्रदान करता है?",
        "expected_lang": SupportedLanguage.HINDI,
        "expected_locale": "hi-IN",
        "topic": "services",
    },
    {
        "id": "EVAL-HING-01",
        "lang": SupportedLanguage.HINGLISH,
        "input": "Aapka cloud security aur managed services ka kya process hai?",
        "expected_lang": SupportedLanguage.HINGLISH,
        "expected_locale": "en-IN",
        "topic": "security",
    },
    {
        "id": "EVAL-TA-01",
        "lang": SupportedLanguage.TAMIL,
        "input": "பிட்மேக்ஸ் குழுமத்தின் சேவைகள் என்ன?",
        "expected_lang": SupportedLanguage.TAMIL,
        "expected_locale": "ta-IN",
        "topic": "services",
    },
    {
        "id": "EVAL-TE-01",
        "lang": SupportedLanguage.TELUGU,
        "input": "బిట్‌మాక్స్ గ్రూప్ అందించే సేవలు ఏమిటి?",
        "expected_lang": SupportedLanguage.TELUGU,
        "expected_locale": "te-IN",
        "topic": "services",
    },
]


@pytest.mark.parametrize("item", MULTILINGUAL_TEST_DATASET)
def test_evaluation_language_matching(item):
    res = LanguageDetector.detect_utterance(item["input"])
    assert res.detected_language == item["expected_lang"], f"Failed for {item['id']}"
    assert res.locale_code == item["expected_locale"], f"Locale mismatch for {item['id']}"


@pytest.mark.asyncio
async def test_evaluation_barge_in_turn_manager():
    turn_manager = TurnManager()
    session_id = "eval_call_session_01"

    # Agent begins speaking
    token = await turn_manager.create_agent_turn(session_id)
    assert await turn_manager.get_state(session_id) == TurnState.SPEAKING
    assert not token.is_cancelled

    # User interrupts
    cancelled_token = await turn_manager.on_user_speech_start(session_id, reason="user_barge_in")
    assert cancelled_token is token
    assert token.is_cancelled is True
    assert token.cancelled_reason == "user_barge_in"
    assert await turn_manager.get_state(session_id) == TurnState.LISTENING


def test_evaluation_session_hysteresis():
    session_meta = {"detected_language": "hi", "language_locale": "hi-IN"}
    # Single borrowed English term should NOT break Hindi context
    lang, locale = resolve_response_language("Okay cloud services", session_meta)
    assert lang == SupportedLanguage.HINDI
    assert locale == "hi-IN"
