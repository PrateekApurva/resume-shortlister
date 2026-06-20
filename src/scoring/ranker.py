"""
Final scoring formula shared by Phase 2 (re-scoring) and Phase 3 (ranking).

final_score = (
    0.25 × work_evidence_score   ← LLM-extracted: what they actually built
  + 0.20 × semantic_similarity   ← FAISS cosine distance: career narrative vs JD
  + 0.20 × career_quality_score  ← product co. history + experience band
  + 0.15 × skills_fit_score      ← JD-weighted skill match, trust-adjusted
  + 0.12 × availability_score    ← behavioural signals: can we actually hire them?
  + 0.08 × location_score        ← India + willing to relocate to Pune/Noida
) × honeypot_multiplier          ← 0.0 kills impossible profiles entirely
"""


def compute_work_evidence_score(evidence: dict, jd_signals: dict) -> float:
    """
    Converts LLM-extracted boolean signals into a 0–1 score.

    Uses signal weights from jd_signals.json — generated dynamically from the JD
    in Phase 0, so this works for any JD without hardcoded signal names.
    """
    if not evidence:
        return 0.0

    score = 0.0
    for signal in jd_signals["signals"]:
        key    = signal["key"]
        weight = signal["weight"]
        value  = evidence.get(key, False)
        if isinstance(value, bool):
            score += weight * (1.0 if value else 0.0)

    # Consulting-only background is a JD red flag — multiplicative penalty
    # so it reduces the score proportionally rather than subtracting a fixed amount
    if evidence.get("company_type") == "consulting":
        score *= 0.6

    return min(score, 1.0)


def compute_final_score(feat: dict, sem_sim: float, work_score: float) -> float:
    """
    Compute the final weighted score for one candidate.

    feat      — feature dict (or pandas Series) from features.parquet
    sem_sim   — cosine similarity from FAISS search (already 0–1 for normalised vectors)
    work_score — output of compute_work_evidence_score()
    """
    score = (
        0.25 * work_score
      + 0.20 * sem_sim
      + 0.20 * float(feat["career_quality_score"])
      + 0.15 * float(feat["skills_fit_score"])
      + 0.12 * float(feat["availability_score"])
      + 0.08 * float(feat["location_score"])
    ) * float(feat["honeypot_multiplier"])

    return round(score, 6)
