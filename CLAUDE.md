# CLAUDE.md — Resume Shotlister

## What This Project Is

Redrob hackathon: build an AI system that ranks 100,000 candidates for a
Senior AI Engineer job description. Output the top 100 as a CSV with scores
and reasoning. Deadline: 2026-07-02.

**Read ARCHITECTURE.md fully before writing any code.** It contains the complete
design, all decisions, scoring formula, and three worked examples with actual numbers.

---

## User Context

Junior Gen AI engineer (1 year experience). Building this to win/place in a
hackathon and put it on their resume. Wants to learn while building.

**How to work with this user:**
- Explain every decision — why this function, why this approach, why this library
- Don't just write code, teach as you go
- When running commands, explain what the command does and why
- When a new concept appears (FAISS, parquet, cosine similarity etc.), explain it briefly
- Keep explanations concise — one clear paragraph, not an essay

---

## Current Status

Architecture: **COMPLETE** (see ARCHITECTURE.md)
Code: **NOT STARTED**

**Build in this exact order** — each phase depends on the previous:

1. `setup` — requirements.txt, folder structure, git init
2. `analyze_jd.py` — Phase 0: one LLM call → 3 artifact files
3. `precompute.py` — Phase 1: 100K feature extraction + FAISS index
4. `enrich.py` — Phase 2: LLM work evidence + reasoning for top 2K/200
5. `rank.py` — Phase 3: final ranking in ≤5 min, no internet
6. `evaluation/evaluate.py` — measure NDCG@10/50 locally
7. Streamlit demo — HuggingFace Spaces (required by hackathon rules)

---

## Tech Stack

```
sentence-transformers   all-MiniLM-L6-v2 model for embeddings
faiss-cpu               vector similarity search
openai                  LLM calls (user has OpenAI API key)
pandas                  data processing + parquet files
numpy                   numerical operations
tqdm                    progress bars for 100K processing loops
python-docx             read .docx JD file
streamlit               demo app for HuggingFace Spaces
```

---

## Hard Constraints (never violate these)

These come from the hackathon rules. Breaking any = disqualification:

- `rank.py` must finish in **≤ 5 minutes** wall-clock on CPU
- `rank.py` must make **zero API calls** (no internet at rank time)
- `rank.py` must use **CPU only** (no GPU)
- Output CSV must have **exactly 100 rows**, ranks 1–100, scores non-increasing
- Top 100 must have **≤ 10% honeypot candidates** (our detection handles this)

---

## Key Files and Folders

```
ARCHITECTURE.md                          complete design document (read first)
CLAUDE.md                                this file

India_runs_data_and_ai_challenge/
├── candidates.json                      100K candidates, NDJSON format, 487MB
├── job_description.docx                 the JD to rank candidates for
├── sample_candidates.json               50 sample candidates (use for dev/testing)
├── validate_submission.py               official validator (run before submitting)
├── submission_metadata_template.yaml    fill this before submitting
└── sample_submission.csv                format reference only

src/
├── data/loader.py                       load NDJSON candidates efficiently
├── features/
│   ├── honeypot.py                      5-check honeypot detection
│   ├── extractor.py                     structured feature extraction
│   ├── skills.py                        skill score computation
│   └── embedder.py                      sentence-transformers wrapper
├── scoring/
│   ├── availability.py                  behavioral signals → availability_score
│   ├── career.py                        career_quality_score
│   └── ranker.py                        final scoring formula
└── llm/
    ├── jd_analyzer.py                   Phase 0 LLM call
    ├── evidence.py                      Phase 2 work evidence extraction
    └── reasoning.py                     Phase 2 reasoning generation

artifacts/                               pre-computed files (git-committed after running)
├── jd_signals.json                      Phase 0 → used in Phase 2, 3
├── jd_skill_weights.json                Phase 0 → used in Phase 1
├── jd_focused_query.txt                 Phase 0 → used in Phase 1, 2, 3
├── features.parquet                     Phase 1 → used in Phase 2, 3
├── candidates.faiss                     Phase 1 → used in Phase 2, 3
├── candidate_ids.json                   Phase 1 → used in Phase 2, 3
├── work_evidence.json                   Phase 2 → used in Phase 3
└── reasoning.json                       Phase 2 → used in Phase 3

evaluation/evaluate.py                   compute NDCG@10/50 locally
```

---

## Scoring Formula (the core logic)

```
final_score = (
    0.25 × work_evidence_score     ← LLM-extracted: did they actually build X?
  + 0.20 × semantic_similarity     ← FAISS: does career narrative match JD?
  + 0.20 × career_quality_score    ← product co. history + experience band
  + 0.15 × skills_fit_score        ← skill match weighted by platform assessments
  + 0.12 × availability_score      ← behavioral signals: are they hirable now?
  + 0.08 × location_score          ← India / willing to relocate to Pune/Noida
) × honeypot_multiplier            ← 0.0 kills impossible profiles
```

Sources:
- `work_evidence_score` → computed from `work_evidence.json` + `jd_signals.json`
- `semantic_similarity` → cosine distance from FAISS search
- all other scores → pre-computed in Phase 1, stored in `features.parquet`

---

## Where to Use Claude Code Features

### TaskCreate — use at the start of each coding session
Create tasks for each module you're about to build. Mark them complete as you go.
This keeps you on track across a long session.
```
Example: TaskCreate "Build src/features/honeypot.py"
```

### /code-review skill — use after completing each module
After writing a complete file, run `/code-review` to catch bugs, logic errors,
and simplification opportunities before moving to the next file.

### Explore agent — use when searching large files
When you need to find where something is defined or how a pattern is used across files:
```
Example: Agent(subagent_type="Explore", prompt="Find where candidate_ids is used")
```

### mcp__ide__getDiagnostics — use after writing code
Check for Python type errors and syntax issues immediately after writing a file.

---

## Development Tips

**Always test on sample_candidates.json first (50 candidates), not the full 100K.**
The full dataset is 487MB. Every script should have a `--sample` flag or accept a
small input for fast iteration.

**Run the official validator before every submission:**
```bash
python India_runs_data_and_ai_challenge/validate_submission.py --submission submission.csv
```

**The FAISS index and embeddings take ~45 min to build. Build once, reuse always.**
Never delete `artifacts/candidates.faiss` or `artifacts/features.parquet` unless
you intentionally want to rebuild from scratch.

**Commit artifacts to git after each phase completes.**
This means the next person (or judge) can run just `rank.py` without rebuilding everything.
Exception: `candidates.json` (487MB) — do not commit, too large.

---

## OpenAI API Usage

User has an OpenAI API key. Use `gpt-4o-mini` for all LLM calls:
- Phase 0: 1 call (JD analysis)
- Phase 2 Step 2: ~2,000 calls (work evidence extraction) → ~$6
- Phase 2 Step 4: ~200 calls (reasoning generation) → ~$0.60
- Total cost: ~$7

Always use structured outputs (response_format with JSON schema) for Phase 2
work evidence extraction — this prevents the LLM from returning malformed JSON.
