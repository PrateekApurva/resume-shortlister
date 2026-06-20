"""
Phase 2 Step 4: Reasoning generation for top 200 candidates.

Generates a 1-2 sentence recruiter-style summary for each candidate explaining
why they ranked where they did. These become the "reasoning" column in submission.csv.

Only runs on the top 200 (not 2,000) because:
- Phase 3 re-runs the same scoring formula and always picks its top 100 from
  within Phase 2's top 200, so every final candidate already has reasoning ready
- Generating 200 instead of 2,000 keeps cost to ~$0.60 instead of ~$6
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

MODEL       = "gpt-4o-mini"
MAX_WORKERS = 10   # reasoning is longer output — fewer parallel workers than evidence

_SYSTEM_PROMPT = """\
You are a senior technical recruiter writing shortlist notes for a hiring manager.
Write factual, specific, honest 1-2 sentence summaries. Reference real facts from
the profile. Acknowledge genuine concerns. Do NOT hallucinate anything not shown below.
"""


def _top_relevant_skills(candidate: dict, jd_skill_weights: dict, n: int = 3) -> list[str]:
    """Return the N candidate skills with highest JD relevance weight."""
    scored = []
    for skill in candidate["skills"]:
        weight = jd_skill_weights.get(skill["name"].lower(), 0.0)
        if weight > 0:
            scored.append((skill["name"], weight))
    scored.sort(key=lambda x: -x[1])
    return [s[0] for s in scored[:n]]


def generate_single(
    candidate: dict,
    rank_in_200: int,
    score: float,
    evidence: dict,
    feat,                    # pandas Series or dict from features.parquet
    jd_skill_weights: dict,
    client: OpenAI,
) -> str:
    """Generate a 1-2 sentence reasoning string for one candidate."""
    top_skills = _top_relevant_skills(candidate, jd_skill_weights)
    work_signals = [k for k, v in evidence.items() if v is True and k not in
                    ("company_type", "red_flags", "confidence")]
    career_path = " → ".join(j["company"] for j in candidate["career_history"])

    prompt = f"""Write a 1-2 sentence recruiter note for this candidate's position in a
Senior AI Engineer shortlist. Be specific. Reference facts from their profile.
Acknowledge concerns honestly if present.

Rank: #{rank_in_200} of 200 shortlisted  |  Score: {score:.3f}

Profile facts:
- Name:          {candidate['profile'].get('anonymized_name', 'Unknown')}
- Current role:  {candidate['profile']['current_title']} at {candidate['profile']['current_company']}
- Career path:   {career_path}
- Company type:  {evidence.get('company_type', 'unknown')}
- Work signals:  {work_signals if work_signals else 'none confirmed'}
- Top skills:    {top_skills if top_skills else 'none matching JD'}
- Location:      {candidate['profile'].get('location', 'unknown')} | Relocate: {feat['willing_to_relocate']}
- Last active:   {int(feat.get('days_since_last_active', 0)) if 'days_since_last_active' in (feat.keys() if hasattr(feat, 'keys') else feat.index) else 'unknown'} days ago
- Response rate: {candidate['redrob_signals']['recruiter_response_rate']}
- Red flags:     {evidence.get('red_flags', [])}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=120,   # 1-2 sentences — keep it tight
        temperature=0.4,  # slight creativity for natural language, still factual
    )

    return response.choices[0].message.content.strip()


def generate_batch(
    top_200: list[tuple[str, float]],   # [(candidate_id, score), ...]
    candidates_lookup: dict[str, dict],
    features_df,                         # pandas DataFrame indexed by candidate_id
    work_evidence: dict[str, dict],
    jd_skill_weights: dict,
) -> dict[str, str]:
    """
    Generate reasoning for all top-200 candidates in parallel.
    Returns {candidate_id: reasoning_text}.
    """
    client = OpenAI()
    results = {}

    def process_one(rank_in_200: int, cid: str, score: float):
        candidate = candidates_lookup[cid]
        evidence  = work_evidence.get(cid, {})
        feat      = features_df.loc[cid]
        text      = generate_single(candidate, rank_in_200, score, evidence, feat,
                                    jd_skill_weights, client)
        return cid, text

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one, rank, cid, score): cid
            for rank, (cid, score) in enumerate(top_200, start=1)
        }

        with tqdm(total=len(top_200), desc="  LLM reasoning", unit="cand") as pbar:
            for future in as_completed(futures):
                cid, text = future.result()
                results[cid] = text
                pbar.update(1)

    return results
