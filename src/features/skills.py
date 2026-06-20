"""
Computes skills_fit_score from a candidate's skills section.

Skills are low-trust (anyone can add them in one click). We counteract this by:
  1. Weighting each skill by its JD relevance (from jd_skill_weights.json)
  2. Blending self-claimed proficiency with platform assessment score (60/40)
     — platform tests are much harder to fake than a skill tag
  3. Applying a duration factor (0.5 to 1.0) — longer usage = more credibility

Negative JD weights (e.g. speech recognition for a retrieval role) reduce the score.
Final score is always in [0.0, 1.0].
"""

PROFICIENCY_WEIGHT = {
    "beginner":     0.30,
    "intermediate": 0.60,
    "advanced":     0.90,
    "expert":       1.00,
}


def compute_skill_score(candidate: dict, jd_skill_weights: dict[str, float]) -> float:
    """
    jd_skill_weights: {skill_name_lowercase: weight}
      e.g. {"faiss": 1.0, "embeddings": 1.0, "speech recognition": -0.2}
    """
    assessments = candidate["redrob_signals"].get("skill_assessment_scores", {})

    total_weighted = 0.0
    max_possible   = 0.0

    for skill in candidate["skills"]:
        name      = skill["name"].lower()
        relevance = jd_skill_weights.get(name, 0.0)

        if relevance == 0.0:
            continue   # skill not in JD weight table — skip entirely

        proficiency = PROFICIENCY_WEIGHT.get(skill.get("proficiency", "beginner"), 0.30)

        # If the platform tested this skill, blend: 40% self-claim + 60% platform score.
        # assessments keys may use original casing — check case-insensitively.
        platform_score = _get_assessment(assessments, skill["name"])
        if platform_score is not None:
            proficiency = 0.40 * proficiency + 0.60 * platform_score

        # Duration factor: 0 months → 0.5, 60+ months → 1.0
        duration = skill.get("duration_months", 0)
        duration_factor = 0.5 + 0.5 * min(duration, 60) / 60

        contribution = relevance * proficiency * duration_factor
        total_weighted += contribution
        max_possible   += abs(relevance)   # negative weights still count toward denominator

    if max_possible == 0.0:
        return 0.0

    return max(0.0, min(total_weighted / max_possible, 1.0))


def _get_assessment(assessments: dict, skill_name: str) -> float | None:
    """Look up platform assessment score case-insensitively. Returns 0–1 or None."""
    for key, score in assessments.items():
        if key.lower() == skill_name.lower():
            return float(score) / 100.0
    return None
