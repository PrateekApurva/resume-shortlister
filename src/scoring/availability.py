"""
Computes availability_score from redrob_signals.

Measures how likely a candidate is to respond, interview, and actually join.
All inputs come from platform-observed behaviour (highest trust tier).

Formula:
  0.30 × recency          (how recently they were active — stale profiles waste recruiter time)
  0.25 × open_to_work     (explicit signal they want a job right now)
  0.20 × response_rate    (do they actually reply to recruiters?)
  0.15 × notice_period    (shorter notice = faster hire)
  0.10 × interview_compl  (do they follow through on interviews?)
"""

from datetime import date


def compute_availability(signals: dict, today: date | None = None) -> float:
    if today is None:
        today = date.today()

    last_active = _parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = (today - last_active).days
        # Linear decay to 0 over 180 days. Active today = 1.0; inactive 6+ months = 0.0.
        recency = max(0.0, 1.0 - days_inactive / 180)
    else:
        recency = 0.5  # unknown — assume average

    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.6

    response_rate = _safe_float(signals.get("recruiter_response_rate"), default=0.5)

    notice_days = signals.get("notice_period_days", 90)
    # 0 days notice = 1.0 (can join immediately); 180+ days = 0.0
    notice_score = max(0.0, 1.0 - notice_days / 180)

    # offer_acceptance_rate and interview_completion_rate can be -1 (not set)
    interview = _safe_float(signals.get("interview_completion_rate"), default=0.5)

    return (
        0.30 * recency
      + 0.25 * open_to_work
      + 0.20 * response_rate
      + 0.15 * notice_score
      + 0.10 * interview
    )


def _parse_date(date_str: str | None) -> date | None:
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None


def _safe_float(value, default: float = 0.5) -> float:
    """Return default if value is None, -1 (sentinel for 'not set'), or non-numeric."""
    try:
        v = float(value)
        return default if v < 0 else v
    except (TypeError, ValueError):
        return default
