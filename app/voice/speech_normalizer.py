"""
Spoken text normalization and prosody enhancement for LEO Voice Pipeline.

Pre-processes text before Text-to-Speech (TTS) synthesis to ensure natural,
human-like speech delivery:
- Strips markdown formatting, asterisks, hash headers, and raw symbols.
- Normalizes currency symbols to spoken words (₹, Rs. -> rupees).
- Expands acronyms and abbreviations (BPO -> B.P.O., KPI -> K.P.I., QA -> Q.A.).
- Converts technical notations (16x6 -> 16 by 6, 24x7 -> 24 by 7, >= -> at least).
- Formats project codes for clear pronunciation (RO/567772/2026 -> R.O. 567772 of 2026).
- Preserves native Indic scripts (Tamil, Hindi, Telugu, Bengali, Kannada, Malayalam) verbatim.
"""

from __future__ import annotations

import re


# Common acronyms to format with dot separation so TTS pronounces individual letters clearly
_ACRONYMS = {
    r"\bBPO\b": "B.P.O.",
    r"\bKPO\b": "K.P.O.",
    r"\bLPO\b": "L.P.O.",
    r"\bKPIs\b": "K.P.I.s",
    r"\bKPI\b": "K.P.I.",
    r"\bQA\b": "Q.A.",
    r"\bTL\b": "Team Leader",
    r"\bSLA\b": "S.L.A.",
    r"\bSLAs\b": "S.L.A.s",
    r"\bNDA\b": "N.D.A.",
    r"\bKYC\b": "K.Y.C.",
    r"\bTAT\b": "turnaround time",
    r"\bAHT\b": "average handling time",
    r"\bFCR\b": "first call resolution",
    r"\bCSAT\b": "customer satisfaction",
    r"\bSOW\b": "scope of work",
    r"\bCRM\b": "C.R.M.",
    r"\bSSD\b": "S.S.D.",
    r"\bRAM\b": "R.A.M.",
    r"\bVPN\b": "V.P.N.",
    r"\bFHD\b": "F.H.D.",
    r"\bUSB\b": "U.S.B.",
    r"\bIST\b": "I.S.T.",
    r"\bWPM\b": "words per minute",
    r"\bIT\b": "I.T.",
    r"\bMbps\b": "megabits per second",
    r"\bGB\b": "gigabytes",
}


def _replace_currency(m: re.Match) -> str:
    amount = m.group(1).strip()
    unit = m.group(2)
    if unit:
        return f" {amount} {unit} rupees "
    return f" {amount} rupees "


def normalize_text_for_speech(text: str, language_code: str = "en-IN") -> str:
    """
    Transforms written assistant response text into natural, spoken-ready dialogue.
    Cleans formatting and expands pronunciation without altering core meaning.
    """
    if not text or not text.strip():
        return ""

    s = text.strip()

    # 1. Strip Markdown headers and bold/italic markers
    s = re.sub(r"#{1,6}\s*", "", s)                    # Remove ### headers
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)           # Remove **bold**
    s = re.sub(r"\*([^*]+)\*", r"\1", s)               # Remove *italic*
    s = re.sub(r"__([^_]+)__", r"\1", s)               # Remove __bold__
    s = re.sub(r"_([^_]+)_", r"\1", s)                 # Remove _italic_
    s = re.sub(r"`([^`]+)`", r"\1", s)                 # Remove `code`
    s = re.sub(r"^\s*[-*•]\s+", "", s, flags=re.MULTILINE)  # Remove list bullets
    s = re.sub(r"^\s*\d+\.\s+", "", s, flags=re.MULTILINE)  # Remove numbered lists

    # 2. Currency symbol normalization
    s = re.sub(r"₹\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(lakh|crore)?", _replace_currency, s, flags=re.IGNORECASE)
    s = re.sub(r"Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(lakh|crore)?", _replace_currency, s, flags=re.IGNORECASE)
    s = re.sub(r"₹", " rupees ", s)

    # 3. Ratio / Operating model expressions: 16x6, 24x7, 16×6, 24×7
    s = re.sub(r"\b(\d{1,2})\s*[x×]\s*(\d{1,2})\b", r"\1 by \2", s)

    # 4. Math & KPI comparison symbols
    s = re.sub(r"(?:<=|≤)\s*(\d+)", r"under \1", s)
    s = re.sub(r"(?:>=|≥)\s*(\d+)", r"at least \1", s)
    s = re.sub(r"(\d+)%", r"\1 percent", s)

    # 5. Project codes normalization for spoken clarity
    s = re.sub(r"\bRO/567772/2026\b", "R.O. 567772 of 2026", s)
    s = re.sub(r"\bNTX/65772/2026\b", "NTX 65772 of 2026", s)
    s = re.sub(r"\b452/TYC/2026\b", "452 TYC 2026", s)
    s = re.sub(r"\b552662/RO\b", "552662 RO", s)
    s = re.sub(r"\b8282/26/NTX\b", "8282 26 NTX", s)
    s = re.sub(r"\b4326727/26\b", "4326727 26", s)

    # 6. Acronyms expansion in English / Hinglish contexts
    # Only replace Latin-based acronyms if text contains Latin characters
    if re.search(r"[a-zA-Z]", s):
        for pattern, replacement in _ACRONYMS.items():
            s = re.sub(pattern, replacement, s)

    # 7. Hardware & technical specifications
    s = re.sub(r"(?<!Core\s)\bi5\b", "Core i5", s, flags=re.IGNORECASE)
    s = re.sub(r"10th\s*Gen\+?", "tenth generation or higher", s, flags=re.IGNORECASE)

    # 8. Clean up whitespace and punctuation
    s = re.sub(r"[:;]\s*", ", ", s)               # Colons/semicolons become pause commas
    # Space after punctuation except commas inside numbers like 68,000
    s = re.sub(r"(?<!\d),|,(?!\d)", ", ", s)
    s = re.sub(r"\s*([!?।])\s*", r"\1 ", s)       # Space after other sentence punctuation
    s = re.sub(r"\s+", " ", s).strip()            # Collapse multi-spaces
    s = re.sub(r"([.!?।])\s*\1+", r"\1", s)        # Collapse repeated punctuation

    return s
