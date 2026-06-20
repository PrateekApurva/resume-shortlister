"""
Phase 2: LLM Enrichment

Step 1 — FAISS search: embed JD query → find top 2,000 candidates
Step 2 — LLM evidence: extract work signals for each of those 2,000 (~$6, ~3 min)
Step 3 — Re-score all 2,000 with full formula → take top 200
Step 4 — LLM reasoning: generate 1-2 sentence note for each of top 200 (~$0.60)

Outputs:
    artifacts/work_evidence.json   — LLM-extracted signals for top 2,000
    artifacts/reasoning.json       — recruiter reasoning text for top 200

Usage:
    python enrich.py               # full run
    python enrich.py --sample      # use sample_candidates.json, k=50
    python enrich.py --force       # rerun even if artifacts exist
    python enrich.py --skip-evidence  # skip Step 2 (reuse existing work_evidence.json)
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.data.loader import iter_candidates
from src.features.embedder import encode_single
from src.llm.evidence import extract_batch
from src.llm.reasoning import generate_batch
from src.scoring.ranker import compute_final_score, compute_work_evidence_score

load_dotenv()

ROOT          = Path(__file__).parent
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR      = ROOT / "India_runs_data_and_ai_challenge"

FULL_CANDIDATES   = DATA_DIR / "candidates.json"
SAMPLE_CANDIDATES = DATA_DIR / "sample_candidates.json"

FAISS_K        = 2000   # candidates retrieved by semantic search
REASONING_TOP  = 200    # candidates that get LLM reasoning


def load_artifacts() -> dict:
    required = [
        "jd_signals.json", "jd_skill_weights.json", "jd_focused_query.txt",
        "features.parquet", "candidates.faiss", "candidate_ids.json",
    ]
    for name in required:
        if not (ARTIFACTS_DIR / name).exists():
            print(f"ERROR: Missing artifact: {ARTIFACTS_DIR / name}")
            print("Run analyze_jd.py then precompute.py first.")
            sys.exit(1)

    return {
        "jd_signals":       json.loads((ARTIFACTS_DIR / "jd_signals.json").read_text()),
        "jd_skill_weights": json.loads((ARTIFACTS_DIR / "jd_skill_weights.json").read_text()),
        "jd_focused_query": (ARTIFACTS_DIR / "jd_focused_query.txt").read_text().strip(),
        "features_df":      pd.read_parquet(ARTIFACTS_DIR / "features.parquet"),
        "faiss_index":      faiss.read_index(str(ARTIFACTS_DIR / "candidates.faiss")),
        "candidate_ids":    json.loads((ARTIFACTS_DIR / "candidate_ids.json").read_text()),
    }


def faiss_search(index, jd_query: str, candidate_ids: list, k: int) -> tuple[list, dict]:
    """Embed JD query and search FAISS. Returns (ordered id list, {id: similarity})."""
    print(f"\nStep 1: FAISS search (k={k})...")
    jd_vec = encode_single(jd_query)                    # shape (1, 384), L2-normalised
    distances, indices = index.search(jd_vec, k)        # distances = cosine similarities

    top_ids  = [candidate_ids[i] for i in indices[0]]
    sim_map  = {candidate_ids[i]: float(distances[0][j])
                for j, i in enumerate(indices[0])}

    print(f"  Top similarity: {distances[0][0]:.4f}  |  "
          f"Bottom of top-{k}: {distances[0][-1]:.4f}")
    return top_ids, sim_map


def load_candidates_for_ids(
    candidates_path: Path,
    needed_ids: set,
) -> dict[str, dict]:
    """
    Stream through candidates file and collect only the candidates we need.
    Memory-efficient: never loads all 100K into RAM.
    """
    print(f"\n  Loading {len(needed_ids):,} candidate profiles from disk...")
    result = {}
    for candidate in tqdm(iter_candidates(candidates_path), desc="  Scanning", unit="cand"):
        cid = candidate["candidate_id"]
        if cid in needed_ids:
            result[cid] = candidate
        if len(result) == len(needed_ids):
            break   # found everyone — no need to read further
    print(f"  Loaded {len(result):,} candidates.")
    return result


def rescore(
    top_ids: list,
    sim_map: dict,
    features_df: pd.DataFrame,
    work_evidence: dict,
    jd_signals: dict,
) -> dict[str, float]:
    """Score all top-K candidates with the full formula."""
    scores = {}
    for cid in top_ids:
        if cid not in features_df.index:
            continue
        feat       = features_df.loc[cid]
        evidence   = work_evidence.get(cid, {})
        sem_sim    = sim_map[cid]
        work_score = compute_work_evidence_score(evidence, jd_signals)
        scores[cid] = compute_final_score(feat, sem_sim, work_score)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Phase 2: LLM enrichment")
    parser.add_argument("--sample", action="store_true",
                        help="Use sample_candidates.json and k=50")
    parser.add_argument("--k", type=int, default=None,
                        help="Override FAISS k (e.g. --k 10 for a cheap test run)")
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing artifacts")
    parser.add_argument("--skip-evidence", action="store_true",
                        help="Reuse existing work_evidence.json (skip Step 2)")
    args = parser.parse_args()

    evidence_path  = ARTIFACTS_DIR / "work_evidence.json"
    reasoning_path = ARTIFACTS_DIR / "reasoning.json"

    if evidence_path.exists() and reasoning_path.exists() and not args.force:
        print("Phase 2 artifacts already exist. Use --force to rerun.")
        return

    # ── Resolve candidates path ─────────────────────────────────────────────────
    if args.candidates:
        candidates_path = args.candidates
    elif args.sample:
        candidates_path = SAMPLE_CANDIDATES
    else:
        candidates_path = FULL_CANDIDATES

    if args.k:
        k = args.k
    elif args.sample:
        k = 50
    else:
        k = FAISS_K

    # ── Load Phase 0 + 1 artifacts ──────────────────────────────────────────────
    print("Loading artifacts...")
    arts = load_artifacts()
    jd_signals       = arts["jd_signals"]
    jd_skill_weights = arts["jd_skill_weights"]
    features_df      = arts["features_df"]

    # ── Step 1: FAISS search ────────────────────────────────────────────────────
    top_ids, sim_map = faiss_search(
        arts["faiss_index"], arts["jd_focused_query"], arts["candidate_ids"], k=k
    )

    # ── Step 2: Load career histories for top-K ─────────────────────────────────
    candidates_lookup = load_candidates_for_ids(candidates_path, set(top_ids))

    # ── Step 3: LLM work evidence extraction ───────────────────────────────────
    if args.skip_evidence and evidence_path.exists():
        print("\nStep 2: Skipping evidence extraction (--skip-evidence). Loading cached...")
        work_evidence = json.loads(evidence_path.read_text())
    else:
        print(f"\nStep 2: Extracting work evidence for {len(top_ids):,} candidates...")
        existing = {}
        if evidence_path.exists() and not args.force:
            existing = json.loads(evidence_path.read_text())
            print(f"  Resuming from checkpoint: {len(existing):,} already done.")

        work_evidence = extract_batch(
            candidates_lookup,
            jd_signals,
            checkpoint_path=str(evidence_path),
            existing=existing,
        )

    print(f"  Work evidence collected for {len(work_evidence):,} candidates.")

    # ── Step 4: Re-score all top-K → take top 200 ──────────────────────────────
    print(f"\nStep 3: Re-scoring {len(top_ids):,} candidates with full formula...")
    scores = rescore(top_ids, sim_map, features_df, work_evidence, jd_signals)

    reasoning_n = min(REASONING_TOP, len(scores))
    top_200 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:reasoning_n]
    print(f"  Top score: {top_200[0][1]:.4f}  |  "
          f"Score at rank {REASONING_TOP}: {top_200[-1][1]:.4f}")
    print(f"  Top 10 candidates: {[cid for cid, _ in top_200[:10]]}")

    # ── Step 5: LLM reasoning for top 200 ──────────────────────────────────────
    print(f"\nStep 4: Generating reasoning for top {reasoning_n} candidates...")
    reasoning = generate_batch(
        top_200,
        candidates_lookup,
        features_df,
        work_evidence,
        jd_skill_weights,
    )

    # ── Save artifacts ──────────────────────────────────────────────────────────
    evidence_path.write_text(json.dumps(work_evidence, indent=2))
    reasoning_path.write_text(json.dumps(reasoning, indent=2))

    print(f"\n=== Phase 2 complete ===")
    print(f"  work_evidence.json — {len(work_evidence):,} candidates")
    print(f"  reasoning.json     — {len(reasoning):,} candidates")
    print(f"\nTop 5 candidates by score:")
    for rank, (cid, score) in enumerate(top_200[:5], 1):
        r = reasoning.get(cid, "")[:80]
        print(f"  #{rank}  {cid}  score={score:.4f}  \"{r}...\"")


if __name__ == "__main__":
    main()
