"""
Phase 2 Step 2: Work evidence extraction via LLM.

For each of the top 2,000 candidates, sends their career history to GPT-4o-mini
and asks it to answer the JD-derived signal questions as true/false.

Key design decisions:
- ThreadPoolExecutor for concurrency: 2,000 sequential calls ≈ 30 min;
  20 parallel workers brings it to ~2–3 min (I/O-bound, not CPU-bound)
- Checkpoint support: saves progress every CHECKPOINT_EVERY calls so a crash
  mid-run doesn't lose everything
- Dynamic JSON schema: built from jd_signals.json at runtime so it works
  for any JD without code changes
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

CHECKPOINT_EVERY = 200   # save work_evidence.json to disk every N completions
MAX_WORKERS      = 20    # concurrent API calls — stay well within rate limits
MODEL            = "gpt-4o-mini"

_SYSTEM_PROMPT = """\
You are an expert technical recruiter evaluating candidates for an AI engineering role.
Read the candidate's career history carefully and answer each signal question truthfully.
Base your answers ONLY on what is explicitly described in the work history provided.
Do NOT infer from the skills section. If the evidence is absent or unclear, answer false.
"""


def build_evidence_schema(signal_keys: list[str]) -> dict:
    """
    Build a strict JSON schema dynamically from the signal keys in jd_signals.json.
    All boolean signal fields + company_type + red_flags + confidence.
    """
    properties = {key: {"type": "boolean"} for key in signal_keys}
    properties["company_type"] = {
        "type": "string",
        "enum": ["product", "consulting", "mixed"],
    }
    properties["red_flags"]  = {"type": "array", "items": {"type": "string"}}
    properties["confidence"] = {"type": "number"}

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def _build_prompt(candidate: dict, jd_signals: dict) -> str:
    career_text = "\n\n".join(
        f"Role: {j['title']} at {j['company']} ({j.get('duration_months', 0)} months)\n"
        f"Description: {j.get('description', '(none)')}"
        for j in candidate["career_history"]
    )

    signal_lines = "\n".join(
        f"- {s['key']}: {s['question']}"
        for s in jd_signals["signals"]
    )

    red_flag_lines = "\n".join(f"- {r}" for r in jd_signals.get("red_flag_patterns", []))

    return f"""Candidate career history (no skills section — read only what they actually did):

{career_text}

Answer each signal (true/false based only on the career history above):
{signal_lines}

Also answer:
- company_type: "product" | "consulting" | "mixed"
  (consulting = TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, or similar)
- red_flags: which of these apply to this candidate (list only those that apply):
{red_flag_lines}
- confidence: 0.0–1.0 how confident are you given the text provided
"""


def extract_single(
    candidate: dict,
    jd_signals: dict,
    schema: dict,
    client: OpenAI,
) -> dict:
    """Extract work evidence for one candidate. Returns the evidence dict."""
    prompt = _build_prompt(candidate, jd_signals)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name":   "work_evidence",
                "schema": schema,
                "strict": True,
            },
        },
        temperature=0.1,   # very low — we want consistent, deterministic judgements
    )

    return json.loads(response.choices[0].message.content)


def extract_batch(
    candidates: dict[str, dict],        # {candidate_id: full candidate dict}
    jd_signals: dict,
    checkpoint_path: str | None = None,  # path to save progress
    existing: dict | None = None,        # already-extracted evidence (for resuming)
) -> dict[str, dict]:
    """
    Extract work evidence for all candidates in parallel.

    Returns {candidate_id: evidence_dict} for all candidates.
    If checkpoint_path is given, saves progress every CHECKPOINT_EVERY completions.
    If existing is given, skips candidates already in it (resume support).
    """
    client   = OpenAI()
    signal_keys = [s["key"] for s in jd_signals["signals"]]
    schema   = build_evidence_schema(signal_keys)

    results  = dict(existing or {})   # start from checkpoint if available
    lock     = threading.Lock()       # protects shared results dict
    done_count = 0

    # Only process candidates not already in results
    todo = {cid: c for cid, c in candidates.items() if cid not in results}
    print(f"  Extracting evidence for {len(todo):,} candidates "
          f"({len(results):,} already cached)...")

    def process_one(cid: str, candidate: dict):
        try:
            evidence = extract_single(candidate, jd_signals, schema, client)
            return cid, evidence, None
        except Exception as e:
            return cid, None, str(e)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one, cid, c): cid
            for cid, c in todo.items()
        }

        with tqdm(total=len(todo), desc="  LLM evidence", unit="cand") as pbar:
            for future in as_completed(futures):
                cid, evidence, error = future.result()

                with lock:
                    if evidence is not None:
                        results[cid] = evidence
                    else:
                        # On failure, store a safe empty-evidence dict so
                        # the scoring formula still works (all signals = False)
                        results[cid] = {k: False for k in signal_keys}
                        results[cid].update({
                            "company_type": "mixed",
                            "red_flags": [f"extraction_error: {error}"],
                            "confidence": 0.0,
                        })
                    done_count += 1

                    # Checkpoint: save partial results to disk periodically
                    if checkpoint_path and done_count % CHECKPOINT_EVERY == 0:
                        with open(checkpoint_path, "w") as f:
                            json.dump(results, f)

                pbar.update(1)

    # Final save
    if checkpoint_path:
        with open(checkpoint_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
