"""
Computes career_quality_score.

Three factors multiplied together (all capped at 1.0):
  base        = 1.0 minus the fraction of career spent at consulting firms
  title_factor = slight boost for engineering/scientist titles
  exp_factor  = bell-curve penalty for too-junior or too-senior candidates

Consulting firms are explicitly listed because the JD red-flags them:
"entire career at IT consulting firms is a disqualifying pattern."
"""

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree", "mindtree",
    "ibm global services", "dxc technology", "unisys",
}

_ENGINEERING_WORDS = {"engineer", "scientist", "developer", "architect", "researcher"}


def compute_career_quality(candidate: dict) -> float:
    jobs = candidate["career_history"]
    total_months = sum(j["duration_months"] for j in jobs)
    if total_months == 0:
        return 0.0

    consulting_months = sum(
        j["duration_months"] for j in jobs
        if _is_consulting(j["company"])
    )
    consulting_ratio = consulting_months / total_months

    # Base: all product companies = 1.0, all consulting = 0.0
    base = 1.0 - consulting_ratio

    has_eng_title = any(
        any(word in j["title"].lower() for word in _ENGINEERING_WORDS)
        for j in jobs
    )
    title_factor = 1.05 if has_eng_title else 0.90

    yoe = candidate["profile"].get("years_of_experience", 0)
    exp_factor = _experience_factor(yoe)

    return min(base * title_factor * exp_factor, 1.0)


def _is_consulting(company_name: str) -> bool:
    name = company_name.lower().strip()
    return any(firm in name for firm in CONSULTING_FIRMS)


def _experience_factor(yoe: float) -> float:
    """
    Bell-curve around 5–9 years (sweet spot for Senior AI Engineer).
    Too junior (<3 yrs) or too senior (12+ yrs) gets penalised.
    """
    if yoe < 3:
        return 0.40
    elif yoe < 5:
        return 0.75
    elif yoe <= 9:
        return 1.00   # sweet spot
    elif yoe <= 12:
        return 0.85
    else:
        return 0.70   # risk of over-qualified / out-of-touch
