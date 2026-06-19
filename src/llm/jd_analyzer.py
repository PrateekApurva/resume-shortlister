
"""
Phase 0 LLM call: sends the full JD text to GPT-4o-mini and gets back
three structured artifacts in one API call.
"""

import json
from openai import OpenAI

# JSON schema for the structured response.
# Using response_format=json_schema forces GPT to return valid JSON that
# matches this shape exactly — no parsing errors, no missing keys.
_JD_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "work_evidence_signals": {
            "type": "object",
            "properties": {
                "signals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "key":      {"type": "string"},
                            "question": {"type": "string"},
                            "weight":   {"type": "number"},
                        },
                        "required": ["key", "question", "weight"],
                        "additionalProperties": False,
                    },
                },
                "red_flag_patterns":      {"type": "array", "items": {"type": "string"}},
                "preferred_company_types": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["signals", "red_flag_patterns", "preferred_company_types"],
            "additionalProperties": False,
        },
        "skill_relevance_weights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill":  {"type": "string"},
                    "weight": {"type": "number"},
                },
                "required": ["skill", "weight"],
                "additionalProperties": False,
            },
        },
        "focused_query": {"type": "string"},
    },
    "required": ["work_evidence_signals", "skill_relevance_weights", "focused_query"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """\
You are a senior technical recruiter with deep expertise in AI/ML engineering roles.
Your job is to analyse a job description and extract three structured outputs that
will power an automated candidate ranking system.

Be precise and technical. The system relies entirely on your analysis — errors here
propagate through all downstream scoring.
"""

_USER_PROMPT_TEMPLATE = """\
Read this job description carefully.

JD TEXT:
{jd_text}

Produce three outputs as a single JSON object:

1. work_evidence_signals
   6-8 signals that reveal whether a candidate's WORK HISTORY (not their skills list)
   demonstrates genuine fit for this role. Each signal must be:
   - Answerable as true/false from reading career history descriptions
   - Specific enough that a recruiter reading a job description entry can decide clearly
   - Focused on what was actually BUILT or SHIPPED, not just mentioned
   Weights across all signals must sum to exactly 1.0.

   Also include:
   - red_flag_patterns: career patterns that disqualify a candidate (list of strings)
   - preferred_company_types: company types that indicate strong fit (list of strings)

2. skill_relevance_weights
   For every skill that might appear in an AI/ML candidate profile, assign a weight:
     1.0   = required, core to the role
     0.5-0.9 = useful, nice to have
     0.0   = irrelevant (omit these — don't include 0.0 weights)
     negative = wrong domain (penalise: e.g. -0.2 for speech recognition in an NLP ranking role)
   Use lowercase skill names. Include at least 20 skills.
   Return as an array of objects: [{{"skill": "faiss", "weight": 1.0}}, ...]

3. focused_query
   A 3-5 sentence prose description of what this role ACTUALLY needs.
   This text will be embedded and compared against candidate career narratives.
   Emphasise the type of work done and systems built — not keywords.
   Make it sound like a senior recruiter's internal summary note.
"""


def analyze_jd(jd_text: str) -> tuple[dict, dict, str]:
    """
    Sends the JD text to GPT-4o-mini and returns three artifacts:
      - jd_signals (dict): work evidence signal schema
      - jd_skill_weights (dict): skill name → relevance weight
      - focused_query (str): clean prose query for embedding

    Returns a tuple: (jd_signals, jd_skill_weights, focused_query)
    """
    client = OpenAI()  # reads OPENAI_API_KEY from environment automatically

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT_TEMPLATE.format(jd_text=jd_text)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "jd_analysis",
                "schema": _JD_ANALYSIS_SCHEMA,
                "strict": True,
            },
        },
        temperature=0.2,  # low temperature = more deterministic, less creative variation
    )

    raw = json.loads(response.choices[0].message.content)

    jd_signals = raw["work_evidence_signals"]
    # LLM returns [{skill, weight}, ...] — convert to {skill: weight} dict for easy lookup
    jd_skill_weights = {item["skill"]: item["weight"] for item in raw["skill_relevance_weights"]}
    focused_query = raw["focused_query"]

    _validate_signals(jd_signals)

    return jd_signals, jd_skill_weights, focused_query


def _validate_signals(jd_signals: dict) -> None:
    """Sanity-check that signal weights sum to ~1.0."""
    total = sum(s["weight"] for s in jd_signals["signals"])
    if not (0.95 <= total <= 1.05):
        raise ValueError(
            f"Signal weights sum to {total:.3f} — must be ~1.0. "
            "Check the LLM response or re-run analyze_jd."
        )
