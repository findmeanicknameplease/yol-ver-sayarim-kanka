
"""
priors.py — Production-grade metadata priors for offline visual search

Drop-in module.
- No external dependencies
- Privacy-first
- Deterministic & fast
- Designed to cover ~90% of lay-user photo search intent

Usage:
    from priors import parse_query_intent, compute_prior_score

    intent = parse_query_intent(query)
    score = compute_prior_score(meta, intent)

Where `meta` is a dict stored per image/frame.
"""

from __future__ import annotations
from typing import Dict

# -----------------------------
# Query Intent Parsing
# -----------------------------

SELF_TERMS = {
    "me", "my", "myself", "i am", "i'm", "self"
}

FAMILY_TERMS = {
    "my daughter", "daughter",
    "my son", "son",
    "my kid", "my kids", "child", "children",
    "my wife", "wife", "spouse",
    "my husband", "husband",
    "my mom", "mother", "mum",
    "my dad", "father",
    "my grandfather", "grandfather", "grandpa",
    "my grandmother", "grandmother", "grandma",
    "my family", "family"
}

GROUP_TERMS = {
    "us", "we", "together", "friends", "group"
}

MEME_TERMS = {
    "meme", "funny", "joke", "shitpost"
}

DOCUMENT_TERMS = {
    "receipt", "invoice", "ticket", "passport", "document", "id", "booking"
}

SCENE_TERMS = {
    "park": {"park"},
    "beach": {"beach", "sea", "ocean"},
    "outdoor": {"outside", "outdoor"},
}

TIME_TERMS = {
    "recent", "yesterday", "today", "last week", "last month"
}


def parse_query_intent(query: str) -> Dict[str, bool]:
    q = query.lower()

    def contains_any(terms):
        return any(t in q for t in terms)

    intent = {
        "self": contains_any(SELF_TERMS),
        "family": contains_any(FAMILY_TERMS),
        "group": contains_any(GROUP_TERMS),
        "meme": contains_any(MEME_TERMS),
        "document": contains_any(DOCUMENT_TERMS),
        "park": "park" in q,
        "beach": contains_any(SCENE_TERMS["beach"]),
        "recent": contains_any(TIME_TERMS),
    }

    return intent


# -----------------------------
# Prior Scoring
# -----------------------------

def compute_prior_score(meta: Dict, intent: Dict[str, bool]) -> float:
    """
    Compute additive prior score.
    `meta` is expected to be a dict with keys like:
      face_count, has_selfie_cues, outdoor, greenery,
      text_density, is_meme, sharpness
    """
    score = 0.0

    face_count = meta.get("face_count", 0)
    has_selfie = meta.get("has_selfie_cues", False)

    # ---- SELF / FAMILY / GROUP ----
    if intent["self"]:
        if face_count >= 1:
            score += 0.10
        if has_selfie:
            score += 0.07
        if face_count == 0:
            score -= 0.15

    if intent["family"]:
        if face_count >= 2:
            score += 0.12

    if intent["group"]:
        if face_count >= 2:
            score += 0.10

    # ---- SCENES ----
    if intent["park"]:
        if meta.get("outdoor"):
            score += 0.05
        if meta.get("greenery"):
            score += 0.05

    if intent["beach"]:
        if meta.get("water"):
            score += 0.08

    # ---- DOCUMENTS ----
    if intent["document"]:
        if meta.get("text_density", 0) > 0.10:
            score += 0.12

    # ---- MEMES ----
    if intent["meme"]:
        if meta.get("is_meme"):
            score += 0.15
        if meta.get("text_density", 0) < 0.05:
            score -= 0.10

    # ---- QUALITY ----
    if meta.get("sharpness", 1.0) < 0.3:
        score -= 0.05

    return score
