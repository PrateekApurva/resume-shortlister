"""
Phase 0: JD Analysis

Reads job_description.docx, makes ONE LLM call to GPT-4o-mini,
and writes 3 artifacts to artifacts/:
  - jd_signals.json       (work evidence signal schema)
  - jd_skill_weights.json (skill name → relevance weight)
  - jd_focused_query.txt  (clean prose query for embedding)

Usage:
    python analyze_jd.py
    python analyze_jd.py --force     # overwrite existing artifacts
    python analyze_jd.py --jd path/to/other.docx
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from docx import Document

load_dotenv()  # loads OPENAI_API_KEY (and anything else) from .env into os.environ

from src.llm.jd_analyzer import analyze_jd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
JD_PATH_DEFAULT = ROOT / "India_runs_data_and_ai_challenge" / "job_description.docx"
ARTIFACTS_DIR = ROOT / "artifacts"


def read_docx(path: Path) -> str:
    """Extract all paragraph text from a .docx file as a single string."""
    doc = Document(str(path))
    # Each paragraph is a block of text. Join with newlines, skip empty ones.
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def artifacts_exist() -> bool:
    return (
        (ARTIFACTS_DIR / "jd_signals.json").exists()
        and (ARTIFACTS_DIR / "jd_skill_weights.json").exists()
        and (ARTIFACTS_DIR / "jd_focused_query.txt").exists()
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Analyse job description")
    parser.add_argument("--jd", type=Path, default=JD_PATH_DEFAULT,
                        help="Path to job description .docx file")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing artifacts")
    args = parser.parse_args()

    # Guard: skip if artifacts already exist (expensive LLM call)
    if artifacts_exist() and not args.force:
        print("Artifacts already exist. Use --force to regenerate.")
        print(f"  {ARTIFACTS_DIR}/jd_signals.json")
        print(f"  {ARTIFACTS_DIR}/jd_skill_weights.json")
        print(f"  {ARTIFACTS_DIR}/jd_focused_query.txt")
        return

    if not args.jd.exists():
        print(f"ERROR: JD file not found: {args.jd}", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in environment.", file=sys.stderr)
        sys.exit(1)

    # ── Step 1: Read the JD ────────────────────────────────────────────────────
    print(f"Reading JD from: {args.jd}")
    jd_text = read_docx(args.jd)
    print(f"JD length: {len(jd_text)} characters, {len(jd_text.split())} words")

    # ── Step 2: One LLM call → 3 outputs ──────────────────────────────────────
    print("\nCalling GPT-4o-mini (one API call)...")
    jd_signals, jd_skill_weights, focused_query = analyze_jd(jd_text)

    # ── Step 3: Write artifacts ────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    signals_path = ARTIFACTS_DIR / "jd_signals.json"
    weights_path = ARTIFACTS_DIR / "jd_skill_weights.json"
    query_path   = ARTIFACTS_DIR / "jd_focused_query.txt"

    signals_path.write_text(json.dumps(jd_signals, indent=2))
    weights_path.write_text(json.dumps(jd_skill_weights, indent=2))
    query_path.write_text(focused_query)

    # ── Step 4: Print a summary so you can sanity-check the output ────────────
    print("\n=== DONE ===")
    print(f"\nWork evidence signals ({len(jd_signals['signals'])} total):")
    for s in jd_signals["signals"]:
        print(f"  [{s['weight']:.2f}] {s['key']}: {s['question']}")

    weight_total = sum(s["weight"] for s in jd_signals["signals"])
    print(f"  Weight total: {weight_total:.3f} (should be ~1.000)")

    print(f"\nSkill weights ({len(jd_skill_weights)} skills):")
    for skill, w in sorted(jd_skill_weights.items(), key=lambda x: -x[1])[:10]:
        print(f"  {w:+.1f}  {skill}")
    if len(jd_skill_weights) > 10:
        print(f"  ... and {len(jd_skill_weights) - 10} more")

    print(f"\nFocused query:\n{focused_query}")

    print(f"\nArtifacts written to {ARTIFACTS_DIR}/")
    print(f"  {signals_path.name}")
    print(f"  {weights_path.name}")
    print(f"  {query_path.name}")


if __name__ == "__main__":
    main()
