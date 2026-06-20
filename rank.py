"""
Phase 3: Final ranking (≤5 min, CPU only, zero API calls)

Loads all pre-computed artifacts from disk, scores the top-2000 candidates
retrieved by FAISS, and writes the top 100 to submission.csv.

This is the script the judges actually run. It must:
- Make zero API calls (no internet)
- Use CPU only
- Finish in under 5 minutes
- Output exactly 100 rows, scores non-increasing

Usage:
    python rank.py              # → submission.csv
    python rank.py --out foo.csv  # custom output path
    python rank.py --k 5000       # search wider FAISS pool (still ≤5 min)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import faiss
import pandas as pd

from src.features.embedder import encode_single
from src.scoring.ranker import compute_final_score, compute_work_evidence_score

ROOT          = Path(__file__).parent
ARTIFACTS_DIR = ROOT / "artifacts"

TOP_N  = 100   # hard constraint: exactly 100 rows
FAISS_K = 2000  # candidates to retrieve from FAISS before re-scoring


def load_artifacts() -> dict:
    """Load all 7 pre-computed artifact files. Exits with a clear message if any are missing."""
    required = [
        "jd_signals.json",
        "jd_focused_query.txt",
        "features.parquet",
        "candidates.faiss",
        "candidate_ids.json",
        "work_evidence.json",
        "reasoning.json",
    ]
    for name in required:
        if not (ARTIFACTS_DIR / name).exists():
            print(f"ERROR: Missing artifact: {ARTIFACTS_DIR / name}")
            print("Run in order: analyze_jd.py → precompute.py → enrich.py")
            sys.exit(1)

    return {
        "jd_signals":       json.loads((ARTIFACTS_DIR / "jd_signals.json").read_text()),
        "jd_focused_query": (ARTIFACTS_DIR / "jd_focused_query.txt").read_text().strip(),
        "features_df":      pd.read_parquet(ARTIFACTS_DIR / "features.parquet"),
        "faiss_index":      faiss.read_index(str(ARTIFACTS_DIR / "candidates.faiss")),
        "candidate_ids":    json.loads((ARTIFACTS_DIR / "candidate_ids.json").read_text()),
        "work_evidence":    json.loads((ARTIFACTS_DIR / "work_evidence.json").read_text()),
        "reasoning":        json.loads((ARTIFACTS_DIR / "reasoning.json").read_text()),
    }


def main():
    t_start = time.time()

    parser = argparse.ArgumentParser(description="Phase 3: Final ranking")
    parser.add_argument("--out", type=Path, default=ROOT / "submission.csv",
                        help="Output CSV path (default: submission.csv)")
    parser.add_argument("--k", type=int, default=FAISS_K,
                        help=f"FAISS search pool size (default: {FAISS_K})")
    args = parser.parse_args()

    print("Phase 3: Final ranking")
    print("=" * 45)

    # ── Step 1: Load artifacts ───────────────────────────────────────────────
    print("\nStep 1: Loading artifacts...")
    arts          = load_artifacts()
    features_df   = arts["features_df"]
    jd_signals    = arts["jd_signals"]
    work_evidence = arts["work_evidence"]
    reasoning     = arts["reasoning"]
    candidate_ids = arts["candidate_ids"]

    t1 = time.time()
    print(f"  Features:      {len(features_df):,} candidates")
    print(f"  Work evidence: {len(work_evidence):,} candidates  (Phase 2 top-2000)")
    print(f"  Reasoning:     {len(reasoning):,} candidates  (Phase 2 top-200)")
    print(f"  Loaded in {t1 - t_start:.1f}s")

    # ── Step 2: Embed JD query + FAISS search ───────────────────────────────
    print(f"\nStep 2: Embedding JD query + FAISS search (k={args.k})...")
    jd_vec = encode_single(arts["jd_focused_query"])          # (1, 384) L2-normalised
    distances, indices = arts["faiss_index"].search(jd_vec, args.k)

    # Map FAISS integer positions back to candidate IDs
    top_ids = [candidate_ids[int(i)] for i in indices[0]]
    sim_map = {candidate_ids[int(i)]: float(distances[0][j])
               for j, i in enumerate(indices[0])}

    t2 = time.time()
    print(f"  Top cosine similarity:    {distances[0][0]:.4f}")
    print(f"  Bottom of pool (rank {args.k}): {distances[0][-1]:.4f}")
    print(f"  Search done in {t2 - t1:.2f}s")

    # ── Step 3: Score all top-K with the full formula ────────────────────────
    # This is pure arithmetic — no LLM, no network calls
    print(f"\nStep 3: Scoring {len(top_ids):,} candidates (full formula, no API)...")
    scores = {}
    missing_features = 0
    for cid in top_ids:
        if cid not in features_df.index:
            missing_features += 1
            continue
        feat       = features_df.loc[cid]
        evidence   = work_evidence.get(cid, {})
        sem_sim    = sim_map[cid]
        work_score = compute_work_evidence_score(evidence, jd_signals)
        scores[cid] = compute_final_score(feat, sem_sim, work_score)

    if missing_features:
        print(f"  WARNING: {missing_features} candidates had no features row (skipped)")

    t3 = time.time()
    print(f"  Scored {len(scores):,} candidates in {t3 - t2:.2f}s")

    # ── Step 4: Sort → take top 100 ─────────────────────────────────────────
    print(f"\nStep 4: Selecting top {TOP_N}...")
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N]

    if len(ranked) < TOP_N:
        print(f"ERROR: Only {len(ranked)} candidates scored — need exactly {TOP_N}")
        sys.exit(1)

    # Verify scores are non-increasing (hard constraint)
    for i in range(1, len(ranked)):
        if ranked[i][1] > ranked[i-1][1]:
            print(f"ERROR: Score increased at rank {i+1} — constraint violated")
            sys.exit(1)

    honeypot_count = sum(
        1 for cid, _ in ranked
        if float(features_df.loc[cid]["honeypot_multiplier"]) == 0.0
    )
    honeypot_pct = honeypot_count / TOP_N * 100
    print(f"  Top score:         {ranked[0][1]:.6f}")
    print(f"  Score at rank 100: {ranked[-1][1]:.6f}")
    print(f"  Honeypots in top 100: {honeypot_count} ({honeypot_pct:.1f}%)")
    if honeypot_pct > 10:
        print(f"  WARNING: Honeypot rate {honeypot_pct:.1f}% exceeds 10% limit!")

    # ── Step 5: Attach reasoning + write CSV ────────────────────────────────
    print(f"\nStep 5: Writing {args.out}...")
    rows = []
    missing_reasoning = 0
    for rank, (cid, score) in enumerate(ranked, start=1):
        reason = reasoning.get(cid, "")
        if not reason:
            # Fallback for any candidate not in Phase 2's top-200
            # (shouldn't happen, but safe to handle)
            feat = features_df.loc[cid]
            reason = (
                f"Score {score:.3f}: career quality "
                f"{float(feat['career_quality_score']):.2f}, "
                f"skills fit {float(feat['skills_fit_score']):.2f}, "
                f"availability {float(feat['availability_score']):.2f}."
            )
            missing_reasoning += 1
        rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        score,
            "reasoning":    reason,
        })

    if missing_reasoning:
        print(f"  NOTE: {missing_reasoning} candidates used fallback reasoning "
              f"(not in Phase 2 top-200)")

    df_out = pd.DataFrame(rows, columns=["candidate_id", "rank", "score", "reasoning"])
    df_out.to_csv(args.out, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    t_end   = time.time()
    elapsed = t_end - t_start

    print(f"\n{'=' * 45}")
    print(f"Phase 3 complete in {elapsed:.1f}s  "
          f"(budget: 300s, used {elapsed/300*100:.1f}%)")
    print(f"  Output:  {args.out}")
    print(f"  Rows:    {len(df_out)}")
    print(f"\nTop 5 candidates:")
    for _, row in df_out.head(5).iterrows():
        print(f"  #{int(row['rank']):>3}  {row['candidate_id']}  "
              f"score={row['score']:.6f}")
        print(f"       {str(row['reasoning'])[:90]}...")

    if elapsed > 300:
        print(f"\nERROR: Took {elapsed:.1f}s — exceeds 300s limit!")
        sys.exit(1)


if __name__ == "__main__":
    main()
