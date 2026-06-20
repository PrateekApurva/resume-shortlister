"""
Structured feature extraction: converts raw candidate JSON → flat feature dict.

This is what gets stored in features.parquet — one row per candidate,
~15 columns, used directly in Phase 3 scoring with no further parsing.
"""

from datetime import date

from src.features.honeypot import detect_honeypot
from src.features.skills import compute_skill_score
from src.scoring.availability import compute_availability
from src.scoring.career import compute_career_quality

# Country names that count as India-based (handles both API and display formats)
_INDIA_NAMES = {"india", "in", "bharat"}

_EDU_TIER_SCORE = {
    "tier_1": 1.00,   # IIT, IISc
    "tier_2": 0.70,   # NIT, SRM, VIT, BITS
    "tier_3": 0.50,   # other private/state colleges
    "tier_4": 0.30,   # unknown / not listed
}

_RELEVANT_FIELDS = {
    "computer science", "computer engineering", "electrical engineering",
    "electronics", "information technology", "mathematics", "statistics",
    "artificial intelligence", "machine learning", "data science",
    "software engineering",
}


def extract_features(candidate: dict, jd_skill_weights: dict, today: date | None = None) -> dict:
    """
    Returns a flat dict of features for one candidate.
    All numeric values are Python floats or ints — safe to store in parquet.
    """
    if today is None:
        today = date.today()

    profile  = candidate["profile"]
    signals  = candidate["redrob_signals"]
    career   = candidate["career_history"]
    edu      = candidate["education"]

    # ── Honeypot ────────────────────────────────────────────────────────────────
    is_hp  = detect_honeypot(candidate)
    hp_mul = 0.0 if is_hp else 1.0

    # ── Location ────────────────────────────────────────────────────────────────
    country = profile.get("country", "").lower().strip()
    is_india = country in _INDIA_NAMES
    willing  = bool(signals.get("willing_to_relocate", False))
    loc_score = _location_score(is_india, willing)

    # ── Career quality ───────────────────────────────────────────────────────────
    career_quality = compute_career_quality(candidate)

    # ── Education ────────────────────────────────────────────────────────────────
    edu_tier, edu_relevant = _education_features(edu)
    edu_score = _EDU_TIER_SCORE.get(edu_tier, 0.30)

    # ── Skills ───────────────────────────────────────────────────────────────────
    skills_fit = compute_skill_score(candidate, jd_skill_weights)

    # ── Availability ─────────────────────────────────────────────────────────────
    avail = compute_availability(signals, today=today)

    return {
        "candidate_id":         candidate["candidate_id"],
        # Experience
        "years_of_experience":  float(profile.get("years_of_experience", 0)),
        # Location
        "is_india_based":       is_india,
        "willing_to_relocate":  willing,
        "location_score":       loc_score,
        # Career
        "career_quality_score": career_quality,
        # Education
        "highest_edu_tier":     edu_tier,
        "edu_relevant_field":   edu_relevant,
        "education_score":      edu_score,
        # Skills
        "skills_fit_score":     skills_fit,
        # Availability
        "availability_score":   avail,
        # Honeypot
        "is_honeypot":          is_hp,
        "honeypot_multiplier":  hp_mul,
    }


def _location_score(is_india: bool, willing_to_relocate: bool) -> float:
    if is_india and willing_to_relocate:
        return 1.00
    if is_india:
        return 0.70   # in India but not explicitly willing to relocate
    if willing_to_relocate:
        return 0.30   # outside India but open to relocating
    return 0.00       # outside India AND not willing


def _education_features(education: list[dict]) -> tuple[str, bool]:
    """
    Returns (highest_tier, edu_relevant_field).
    highest_tier: "tier_1" > "tier_2" > "tier_3" > "tier_4"
    edu_relevant_field: True if any degree is in a CS/ML/Math related field
    """
    tier_rank = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1}
    best_tier = "tier_4"
    best_rank = 1
    is_relevant = False

    for e in education:
        tier = e.get("tier", "tier_4")
        rank = tier_rank.get(tier, 1)
        if rank > best_rank:
            best_rank = rank
            best_tier = tier

        field = e.get("field_of_study", "").lower()
        if any(rel in field for rel in _RELEVANT_FIELDS):
            is_relevant = True

    return best_tier, is_relevant
