"""
guardrails.py

The safety layer of the bot.

Before any user message reaches the LLM, it passes through this check.
If the message matches a sensitive pattern (self-harm, abuse disclosure,
immediate danger), we DO NOT send it to the LLM at all. Instead, we
return a fixed, pre-written safe response.

Why intercept BEFORE the LLM, rather than just prompting the LLM to
"be careful" with sensitive topics?
- It's deterministic: a fixed response can't be talked out of itself,
  hallucinate, or accidentally say something harmful.
- It's auditable: every guardrail trigger can be logged for the
  organisation to review, regardless of which LLM is powering generation.
- It's fast and free: no API call needed for these cases.

Two-layer approach:
1. Regex patterns — fast, free, deterministic. Catches explicit language.
2. Zero-shot ML classifier — catches paraphrases and indirect disclosures
   that no keyword list will ever cover ("I don't want to be here anymore").
   Only runs if regex finds nothing, so cost is minimal.
"""

import re
from dataclasses import dataclass
from typing import Optional
from transformers import pipeline


# --- ML Classifier (loads once at startup) ---
print("Loading safety classifier... (first run may take a moment)")
_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

_ML_CANDIDATE_LABELS = [
    "self-harm or suicide",
    "sexual abuse or domestic violence",
    "immediate danger or threat",
    "reproductive health question",
    "general conversation",
]

_ML_UNSAFE_LABELS = {
    "self-harm or suicide",
    "sexual abuse or domestic violence",
    "immediate danger or threat",
}

_ML_CONFIDENCE_THRESHOLD = 0.60


@dataclass
class GuardrailResult:
    triggered: bool
    category: Optional[str] = None
    matched_pattern: Optional[str] = None
    safe_response: Optional[str] = None


# Each category maps to:
# - a list of regex patterns (case-insensitive)
# - a fixed safe response shown to the user instead of an LLM-generated answer
GUARDRAIL_CATEGORIES = {
    "self_harm": {
        "patterns": [
            r"\bkill (myself|himself|herself|themselves)\b",
            r"\bend my life\b",
            r"\bsuicid\w*\b",
            r"\bself[\s-]?harm\w*\b",
            r"\bhurt(ing)? myself\b",
            r"\bcut(ting)? myself\b",
            r"\bdon'?t want to (live|be alive)\b",
            r"\bno (reason|point) (in|to) (living|liv(e|ing))\b",
            r"\bwant(ing)? to die\b",
            r"\bbetter off dead\b",
        ],
        "safe_response": (
            "Thank you for trusting me with something this important. I'm not able "
            "to support this myself, but you don't have to go through this alone. "
            "Please reach out right now to a trusted adult, a school counselor, a "
            "health worker, or a local crisis helpline — they can give you real "
            "support today. If you feel you might act on these feelings, please "
            "contact emergency services or go to the nearest hospital immediately."
        ),
    },
    "abuse_disclosure": {
        "patterns": [
            r"\b(rape|raped|raping)\b",
            r"\bsexual(ly)?\s+abus\w*\b",
            r"\bmolest\w*\b",
            r"\btouch(ed|ing)\s+me\b.{0,30}\b(wrong|bad|private|uncomfortable|scared)\b",
            r"\bforced me to\b",
            r"\bsomeone (touched|hurt|abused) me\b",
            r"\b(he|she|they)\s+(touched|forced|hurt|abused)\s+me\b",
        ],
        "safe_response": (
            "Thank you for telling me this — it takes courage, and what happened is "
            "not your fault. I'm not able to handle this myself, but please tell a "
            "trusted adult, a school counselor, a health worker, or a local child "
            "protection helpline as soon as you safely can, so you can get real "
            "support and stay safe."
        ),
    },
    "immediate_danger": {
        "patterns": [
            r"\bnot safe at home\b",
            r"\b(he|she|they)'?(ll| will)?\s*(hurt|kill) me\b",
            r"\bhurt me\b.{0,30}\bif\b",
            r"\bI'?m (scared|afraid) (he|she|they) (will|is going to|are going to)\b",
            r"\btrapped\b.{0,30}\b(home|house|him|her|them)\b",
            r"\bhe (won'?t|will not) let me leave\b",
        ],
        "safe_response": (
            "Your safety matters right now. I'm not able to help with this myself, "
            "but please reach out immediately to a trusted adult, local authorities, "
            "or an emergency helpline in your area if you are in danger."
        ),
    },
}

# Map ML labels back to the right safe response
_ML_LABEL_TO_CATEGORY = {
    "self-harm or suicide": "self_harm",
    "sexual abuse or domestic violence": "abuse_disclosure",
    "immediate danger or threat": "immediate_danger",
}


def check_guardrail(text: str) -> GuardrailResult:
    """
    Two-layer safety check:
    1. Regex patterns — fast, free, deterministic.
    2. Zero-shot ML classifier — catches indirect/paraphrased disclosures.

    Returns the first match found. If nothing triggers, returns
    GuardrailResult(triggered=False).
    """
    # --- Layer 1: Regex ---
    for category, config in GUARDRAIL_CATEGORIES.items():
        for pattern in config["patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                return GuardrailResult(
                    triggered=True,
                    category=category,
                    matched_pattern=pattern,
                    safe_response=config["safe_response"],
                )

    # --- Layer 2: ML classifier (only if regex found nothing) ---
    ml_result = _classifier(text, _ML_CANDIDATE_LABELS)
    top_label = ml_result["labels"][0]
    top_score = ml_result["scores"][0]

    if top_label in _ML_UNSAFE_LABELS and top_score >= _ML_CONFIDENCE_THRESHOLD:
        category = _ML_LABEL_TO_CATEGORY[top_label]
        return GuardrailResult(
            triggered=True,
            category=f"ml_{category}",
            matched_pattern=f"zero-shot-classifier ({top_label}, {top_score:.2f})",
            safe_response=GUARDRAIL_CATEGORIES[category]["safe_response"],
        )

    return GuardrailResult(triggered=False)


if __name__ == "__main__":
    # Quick manual check from the command line:
    #   python guardrails.py "some message to test"
    import sys
    if len(sys.argv) < 2:
        print('Usage: python guardrails.py "message to test"')
        sys.exit(1)
    msg = " ".join(sys.argv[1:])
    result = check_guardrail(msg)
    print(f"\nMessage: {msg}")
    print(f"Triggered: {result.triggered}")
    if result.triggered:
        print(f"Category: {result.category}")
        print(f"Matched pattern: {result.matched_pattern}")
        print(f"Safe response:\n{result.safe_response}")