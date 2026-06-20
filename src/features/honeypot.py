"""
Honeypot detection: 5 rule-based checks.

The dataset contains ~80 planted "impossible" profiles to test if your system
actually reads candidates. If 2+ checks fire → honeypot_multiplier = 0.0,
which zeroes out the final score regardless of other signals.

Check 1: Expert/Advanced skill with 0 months duration
Check 2: Stated YoE vs total career months gap > 4 years
Check 3: Overlapping full-time jobs
Check 4: Impossible education sequence (PhD before Bachelor's, etc.)
Check 5: Job title vs description gross mismatch
"""

from datetime import datetime


def detect_honeypot(candidate: dict) -> bool:
    """Return True if the candidate is likely a honeypot (2+ checks fire)."""
    signals = [
        _check_expert_zero_duration(candidate["skills"]),
        _check_yoe_vs_career_months(candidate),
        _check_overlapping_jobs(candidate["career_history"]),
        _check_impossible_education(candidate["education"]),
        _check_title_description_mismatch(candidate["career_history"]),
    ]
    return sum(signals) >= 2


def honeypot_multiplier(candidate: dict) -> float:
    return 0.0 if detect_honeypot(candidate) else 1.0


# ── Check 1 ────────────────────────────────────────────────────────────────────

def _check_expert_zero_duration(skills: list[dict]) -> bool:
    """2+ skills claiming expert/advanced proficiency but 0 months of use."""
    zero_duration_experts = [
        s for s in skills
        if s.get("proficiency") in ("advanced", "expert")
        and s.get("duration_months", 1) == 0
    ]
    return len(zero_duration_experts) >= 2


# ── Check 2 ────────────────────────────────────────────────────────────────────

def _check_yoe_vs_career_months(candidate: dict) -> bool:
    """Stated years_of_experience vs sum of career job durations differ by > 4 years."""
    total_months = sum(j.get("duration_months", 0) for j in candidate["career_history"])
    stated_years = candidate["profile"].get("years_of_experience", 0)
    gap_years = abs(stated_years - total_months / 12)
    return gap_years > 4


# ── Check 3 ────────────────────────────────────────────────────────────────────

def _check_overlapping_jobs(career: list[dict]) -> bool:
    """Two jobs with overlapping date ranges (both non-current = both have end dates)."""
    dated_jobs = []
    for job in career:
        start = _parse_date(job.get("start_date"))
        end   = _parse_date(job.get("end_date"))   # None if current job
        if start and end:
            dated_jobs.append((start, end))

    # Check all pairs for overlap
    for i in range(len(dated_jobs)):
        for j in range(i + 1, len(dated_jobs)):
            s1, e1 = dated_jobs[i]
            s2, e2 = dated_jobs[j]
            # Overlap: one starts before the other ends, and vice versa
            # A gap of 0 days (same-day transitions) is NOT an overlap
            if s1 < e2 and s2 < e1:
                return True
    return False


# ── Check 4 ────────────────────────────────────────────────────────────────────

def _check_impossible_education(education: list[dict]) -> bool:
    """
    A higher degree (PhD, Masters) cannot end BEFORE a lower degree (Bachelor's)
    ends — you need the lower degree to be awarded first.

    Example of the impossible case:
      PhD:    2013–2016  (rank 3)
      B.Tech: 2015–2019  (rank 1)
      → PhD ended 2016 while B.Tech still ongoing until 2019 → impossible
    """
    degree_rank = {
        "ph.d": 3, "phd": 3, "doctorate": 3,
        "m.tech": 2, "m.e.": 2, "mtech": 2, "m.s.": 2, "ms": 2,
        "mba": 2, "m.sc": 2, "msc": 2, "m.a.": 2, "ma": 2,
        "b.tech": 1, "be": 1, "b.e.": 1, "b.sc": 1, "bsc": 1,
        "b.a.": 1, "ba": 1, "b.com": 1,
        "diploma": 0,
    }

    def rank(deg: str) -> int:
        d = deg.lower().strip().rstrip(".")
        for key, val in degree_rank.items():
            if key in d:
                return val
        return 1

    entries = [
        (e.get("start_year"), e.get("end_year"), rank(e.get("degree", "")))
        for e in education
        if e.get("end_year")
    ]

    for i in range(len(entries)):
        for j in range(len(entries)):
            if i == j:
                continue
            _, end_i, rank_i = entries[i]
            _, end_j, rank_j = entries[j]
            # Higher degree (rank_i) must end AFTER lower degree (rank_j) ends
            if rank_i > rank_j and end_i < end_j:
                return True

    return False


# ── Check 5 ────────────────────────────────────────────────────────────────────

# Keywords that strongly suggest a domain — if title and description domains clash, it's a mismatch
_DOMAIN_KEYWORDS = {
    "mechanical": {"solidworks", "cad", "ansys", "fea", "creo", "machining", "tooling", "dfm"},
    "marketing":  {"seo", "content writing", "brand", "editorial", "campaign", "copywriting", "content creation"},
    "finance":    {"accounts payable", "balance sheet", "audit", "bookkeeping", "tax", "ledger"},
    "design":     {"figma", "adobe", "typography", "ui/ux", "visual design", "logo", "photoshop"},
    "sales":      {"cold calling", "pipeline", "quota", "crm", "salesforce", "deal closing"},
    "operations": {"fulfillment", "logistics", "inventory", "procurement", "supply chain", "warehousing"},
}

_TITLE_DOMAIN = {
    "mechanical":        "mechanical",
    "marketing":         "marketing",
    "accountant":        "finance",
    "graphic designer":  "design",
    "sales":             "sales",
    "operations manager": "operations",
    "content writer":    "marketing",
}


def _check_title_description_mismatch(career: list[dict]) -> bool:
    """Title claims one domain; description describes completely different work on 2+ jobs."""
    mismatches = 0
    for job in career:
        title = job.get("title", "").lower()
        desc  = job.get("description", "").lower()
        if not desc:
            continue

        # Find the title's domain — try longest match first to catch "operations manager"
        # before a shorter fragment like "operations" accidentally matches
        title_domain = None
        for title_fragment in sorted(_TITLE_DOMAIN, key=len, reverse=True):
            if title_fragment in title:
                title_domain = _TITLE_DOMAIN[title_fragment]
                break

        if title_domain is None:
            continue

        other_domains = [d for d in _DOMAIN_KEYWORDS if d != title_domain]
        for other in other_domains:
            hits = sum(1 for kw in _DOMAIN_KEYWORDS[other] if kw in desc)
            if hits >= 2:
                mismatches += 1
                break

    return mismatches >= 2


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_date(date_str: str | None) -> datetime | None:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None
