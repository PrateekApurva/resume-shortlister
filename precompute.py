"""
Phase 1: Offline Preprocessing

Processes all candidates and builds:
  artifacts/features.parquet     — 100K rows × feature columns
  artifacts/candidates.faiss     — FAISS index of work narrative embeddings
  artifacts/candidate_ids.json   — maps FAISS index position → candidate_id

Run once. Takes ~45 min on the full 100K dataset.
For dev/testing, use --sample to run on the 50-candidate sample file.

Usage:
    python precompute.py                           # full 100K
    python precompute.py --sample                  # 50-candidate sample
    python precompute.py --candidates path/to/f    # custom file
    python precompute.py --force                   # overwrite existing artifacts
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from src.data.loader import iter_candidates, load_candidates
from src.features.embedder import build_work_narrative, encode_texts
from src.features.extractor import extract_features

load_dotenv()

ROOT          = Path(__file__).parent
ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR      = ROOT / "India_runs_data_and_ai_challenge"

FULL_CANDIDATES   = DATA_DIR / "candidates.json"
SAMPLE_CANDIDATES = DATA_DIR / "sample_candidates.json"


def artifacts_exist() -> bool:
    return (
        (ARTIFACTS_DIR / "features.parquet").exists()
        and (ARTIFACTS_DIR / "candidates.faiss").exists()
        and (ARTIFACTS_DIR / "candidate_ids.json").exists()
    )


def load_phase0_artifacts() -> dict:
    """Load the 3 artifacts produced by analyze_jd.py (Phase 0)."""
    required = ["jd_signals.json", "jd_skill_weights.json", "jd_focused_query.txt"]
    for name in required:
        if not (ARTIFACTS_DIR / name).exists():
            print(f"ERROR: Missing Phase 0 artifact: {ARTIFACTS_DIR / name}")
            print("Run analyze_jd.py first.")
            sys.exit(1)

    return {
        "jd_signals":      json.loads((ARTIFACTS_DIR / "jd_signals.json").read_text()),
        "jd_skill_weights": json.loads((ARTIFACTS_DIR / "jd_skill_weights.json").read_text()),
        "jd_focused_query": (ARTIFACTS_DIR / "jd_focused_query.txt").read_text().strip(),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Precompute features and FAISS index")
    parser.add_argument("--sample", action="store_true",
                        help="Use sample_candidates.json (50 candidates) instead of full dataset")
    parser.add_argument("--candidates", type=Path, default=None,
                        help="Override path to candidates file")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing artifacts")
    args = parser.parse_args()

    if artifacts_exist() and not args.force:
        print("Phase 1 artifacts already exist. Use --force to rebuild.")
        return

    # ── Load Phase 0 artifacts ──────────────────────────────────────────────────
    phase0 = load_phase0_artifacts()
    jd_skill_weights = phase0["jd_skill_weights"]
    print(f"Loaded Phase 0 artifacts. JD skill weights: {len(jd_skill_weights)} skills.")

    # ── Resolve input path ──────────────────────────────────────────────────────
    if args.candidates:
        candidates_path = args.candidates
    elif args.sample:
        candidates_path = SAMPLE_CANDIDATES
    else:
        candidates_path = FULL_CANDIDATES

    if not candidates_path.exists():
        print(f"ERROR: Candidates file not found: {candidates_path}")
        sys.exit(1)

    print(f"Input: {candidates_path}")

    today = date.today()

    # ── Step 1–3: Feature extraction (per candidate) ────────────────────────────
    # We stream through candidates one by one so RAM usage stays flat
    # (full 100K JSON in memory = ~4GB; streaming keeps it < 500MB)
    print("\nStep 1–3: Extracting features and building work narratives...")
    all_features  = []
    all_narratives = []
    all_ids       = []

    for candidate in tqdm(iter_candidates(candidates_path), desc="Candidates", unit="cand"):
        # Steps 1–3 in ARCHITECTURE.md: honeypot + features + skill score
        features = extract_features(candidate, jd_skill_weights, today=today)
        all_features.append(features)
        all_ids.append(candidate["candidate_id"])

        # Step 4: Build work narrative (text for embedding)
        narrative = build_work_narrative(candidate)
        all_narratives.append(narrative)

    n = len(all_features)
    print(f"Processed {n} candidates.")
    honeypots = sum(1 for f in all_features if f["is_honeypot"])
    print(f"Honeypots detected: {honeypots} ({100*honeypots/n:.1f}%)")

    # ── Step 5: Batch embed all work narratives ──────────────────────────────────
    # encode_texts handles L2 normalisation so inner product = cosine similarity
    print(f"\nStep 5: Embedding {n} work narratives (this takes a few minutes)...")
    embeddings = encode_texts(all_narratives, batch_size=64, show_progress=True)
    # Shape: (n, 384), dtype float32, already L2-normalised
    print(f"Embeddings shape: {embeddings.shape}")

    # ── Step 6: Build FAISS index ────────────────────────────────────────────────
    print("\nStep 6: Building FAISS index...")
    dim   = embeddings.shape[1]         # 384
    index = faiss.IndexFlatIP(dim)      # Flat = exact search; IP = inner product (= cosine on normalised)
    index.add(embeddings)
    print(f"FAISS index: {index.ntotal} vectors, dim={dim}")

    # ── Save all artifacts ───────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    features_path    = ARTIFACTS_DIR / "features.parquet"
    faiss_path       = ARTIFACTS_DIR / "candidates.faiss"
    ids_path         = ARTIFACTS_DIR / "candidate_ids.json"

    features_df = pd.DataFrame(all_features)
    features_df = features_df.set_index("candidate_id")
    features_df.to_parquet(features_path)
    print(f"Saved: {features_path}  ({features_df.shape[0]} rows × {features_df.shape[1]} cols)")

    faiss.write_index(index, str(faiss_path))
    print(f"Saved: {faiss_path}")

    ids_path.write_text(json.dumps(all_ids))
    print(f"Saved: {ids_path}  ({len(all_ids)} candidate IDs)")

    print("\nPhase 1 complete. All artifacts saved to artifacts/")


if __name__ == "__main__":
    main()
