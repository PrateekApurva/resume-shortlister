"""
Local NDCG@10 / NDCG@50 evaluator.

Since the hackathon judges hold the true relevance labels, this script builds
a "silver standard" ground truth from our Phase 2 work_evidence.json — the LLM
already told us which candidates actually demonstrated the JD signals. We use
that to approximate true relevance and measure our submission's ranking quality.

NDCG (Normalized Discounted Cumulative Gain) — why it's the right metric:
  A ranking is "good" if the best candidates appear earliest. NDCG penalises
  you more for putting a great candidate at rank 20 than at rank 5. Scores
  range 0–1; 1.0 = perfect ordering.

Usage:
    python evaluation/evaluate.py                            # default paths
    python evaluation/evaluate.py --submission my.csv        # custom CSV
    python evaluation/evaluate.py --ground-truth labels.json # custom labels
    python evaluation/evaluate.py --dump-labels              # export silver labels
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT          = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_SUB   = ROOT / "submission.csv"


# ── Silver label generation ──────────────────────────────────────────────────

def _signal_count(evidence: dict, signal_keys: list[str]) -> int:
    """Count how many JD boolean signals fired for this candidate."""
    return sum(1 for k in signal_keys if evidence.get(k) is True)


def build_silver_labels(
    work_evidence: dict,
    jd_signals: dict,
    features_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Convert Phase 2 work_evidence into 0–3 graded relevance scores.

    Grading rubric (chosen to spread candidates across all four tiers):
      3 — 3+ signals fired AND product/mixed company  (strong technical fit)
      2 — 2+ signals fired OR 1 signal at product co  (partial fit)
      1 — 1 signal fired at consulting co             (weak fit)
      0 — no signals fired OR honeypot                (not relevant)

    Consulting-only company gets a one-tier penalty because the JD explicitly
    asks for product company experience.
    """
    signal_keys = [s["key"] for s in jd_signals["signals"]]
    labels = {}

    for cid, evidence in work_evidence.items():
        # Honeypot: multiplier = 0, automatically irrelevant
        if cid in features_df.index:
            if float(features_df.loc[cid]["honeypot_multiplier"]) == 0.0:
                labels[cid] = 0.0
                continue

        n_signals    = _signal_count(evidence, signal_keys)
        company_type = evidence.get("company_type", "mixed")
        is_consulting = company_type == "consulting"

        if n_signals >= 3 and not is_consulting:
            relevance = 3.0
        elif n_signals >= 2 or (n_signals >= 1 and not is_consulting):
            relevance = 2.0
        elif n_signals >= 1:   # consulting with 1 signal
            relevance = 1.0
        else:
            relevance = 0.0

        labels[cid] = relevance

    return labels


# ── NDCG computation ─────────────────────────────────────────────────────────

def dcg_at_k(ranked_relevances: list[float], k: int) -> float:
    """
    Discounted Cumulative Gain at cutoff k.
    Formula: sum(rel_i / log2(i+1)) for i = 1..k
    log2(2)=1, so rank 1 has no discount. Rank 2 is discounted by log2(3)≈1.58, etc.
    """
    gains = ranked_relevances[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(gains))
    # note: i starts at 0, so denominator is log2(i+2) = log2(1+1), log2(2+1), ...


def ndcg_at_k(
    ranked_ids: list[str],
    relevance_map: dict[str, float],
    k: int,
) -> float:
    """
    NDCG at cutoff k.

    ranked_ids    — candidate IDs in submission order (rank 1 first)
    relevance_map — {candidate_id: relevance_score}  (0–3 scale)
    """
    # Actual DCG: pick up relevance for each candidate in our ranked order
    ranked_rels = [relevance_map.get(cid, 0.0) for cid in ranked_ids]
    actual_dcg  = dcg_at_k(ranked_rels, k)

    # Ideal DCG: sort all known candidates by relevance, best-first
    ideal_rels = sorted(relevance_map.values(), reverse=True)
    ideal_dcg  = dcg_at_k(ideal_rels, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compute NDCG@10 / NDCG@50 for a submission CSV"
    )
    parser.add_argument("--submission",   type=Path, default=DEFAULT_SUB,
                        help="Submission CSV to evaluate (default: submission.csv)")
    parser.add_argument("--ground-truth", type=Path, default=None,
                        help="Custom ground-truth JSON {candidate_id: relevance 0-3}. "
                             "If omitted, silver labels are generated from work_evidence.json.")
    parser.add_argument("--dump-labels",  action="store_true",
                        help="Write silver labels to evaluation/silver_labels.json and exit")
    args = parser.parse_args()

    # ── Load artifacts ───────────────────────────────────────────────────────
    required = ["jd_signals.json", "work_evidence.json", "features.parquet"]
    for name in required:
        if not (ARTIFACTS_DIR / name).exists():
            print(f"ERROR: Missing {ARTIFACTS_DIR / name}")
            print("Run enrich.py first.")
            sys.exit(1)

    jd_signals    = json.loads((ARTIFACTS_DIR / "jd_signals.json").read_text())
    work_evidence = json.loads((ARTIFACTS_DIR / "work_evidence.json").read_text())
    features_df   = pd.read_parquet(ARTIFACTS_DIR / "features.parquet")

    # ── Build or load ground truth ───────────────────────────────────────────
    if args.ground_truth:
        print(f"Loading custom ground truth from {args.ground_truth}...")
        relevance_map = json.loads(args.ground_truth.read_text())
        print(f"  {len(relevance_map):,} labelled candidates")
    else:
        print("Generating silver labels from Phase 2 work evidence...")
        relevance_map = build_silver_labels(work_evidence, jd_signals, features_df)
        print(f"  {len(relevance_map):,} candidates labelled")

        # Show distribution
        from collections import Counter
        dist = Counter(relevance_map.values())
        for grade in sorted(dist, reverse=True):
            print(f"    Relevance {int(grade)}: {dist[grade]:>5} candidates")

    if args.dump_labels:
        out = Path(__file__).parent / "silver_labels.json"
        out.write_text(json.dumps(relevance_map, indent=2))
        print(f"\nSilver labels written to {out}")
        return

    # ── Load submission ──────────────────────────────────────────────────────
    if not args.submission.exists():
        print(f"ERROR: Submission not found: {args.submission}")
        print("Run rank.py first.")
        sys.exit(1)

    sub_df = pd.read_csv(args.submission)
    sub_df = sub_df.sort_values("rank")                  # ensure rank order
    ranked_ids = sub_df["candidate_id"].tolist()         # rank 1 first

    print(f"\nSubmission: {args.submission}")
    print(f"  {len(ranked_ids)} candidates ranked")

    # ── Compute NDCG ────────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("Results")
    print("=" * 45)

    for k in [10, 50, 100]:
        score = ndcg_at_k(ranked_ids, relevance_map, k)
        bar   = "█" * int(score * 20)
        print(f"  NDCG@{k:<3}  {score:.4f}  {bar}")

    # ── Per-rank breakdown ───────────────────────────────────────────────────
    print("\nTop 10 candidates in submission vs silver label:")
    print(f"  {'Rank':<6} {'Candidate ID':<16} {'Our Score':<12} {'Silver Relevance'}")
    print(f"  {'-'*6} {'-'*16} {'-'*12} {'-'*16}")
    for _, row in sub_df.head(10).iterrows():
        cid       = row["candidate_id"]
        our_score = row["score"]
        silver    = relevance_map.get(cid, "?")
        label     = {3.0: "HIGH (3)", 2.0: "medium (2)",
                     1.0: "low (1)",  0.0: "none (0)"}.get(silver, "unknown")
        print(f"  #{int(row['rank']):<5} {cid:<16} {our_score:<12.6f} {label}")

    # ── Missed high-relevance candidates ────────────────────────────────────
    top_ids_set = set(ranked_ids[:50])
    missed_high = [
        cid for cid, rel in relevance_map.items()
        if rel == 3.0 and cid not in top_ids_set
    ]
    if missed_high:
        print(f"\nHigh-relevance candidates NOT in our top 50: {len(missed_high)}")
        for cid in missed_high[:5]:
            print(f"  {cid}")
        if len(missed_high) > 5:
            print(f"  ... and {len(missed_high) - 5} more")
    else:
        print("\nAll high-relevance candidates captured in top 50.")


if __name__ == "__main__":
    main()
