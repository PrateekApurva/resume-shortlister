# Resume Shotlister

**Redrob "Intelligent Candidate Discovery & Ranking Challenge"**  
Ranks 100,000 candidates for a Senior AI Engineer role and outputs the top 100 with scores and recruiter-style reasoning.

**Live demo:** [huggingface.co/spaces/PrateekApurva/resume-shortlister](https://huggingface.co/spaces/PrateekApurva/resume-shortlister)  
**Presentation:** [Resume Shotlister — Trendy Brains.pdf](./presentation.pdf)

---

## The Core Idea

Most resume filters match keywords. A candidate who "built a semantic search system serving millions at Swiggy" will be missed if they never wrote "RAG" in their skills section.

This system works differently — it reads what candidates **actually built** (career history), not what they claimed (skills section), and ranks by genuine fit.

---

## Architecture — 4 Phases

```
Phase 0  analyze_jd.py     1 LLM call  → extracts signals, skill weights, JD query
Phase 1  precompute.py     ~45 min     → features for 100K candidates + FAISS index
Phase 2  enrich.py         ~30 min     → LLM evidence + reasoning for top 2,000
Phase 3  rank.py           ~4 seconds  → submission.csv (judges run only this)
```

Phases 0–2 are offline prep. The judges only time **Phase 3** — which makes zero API calls, runs on CPU, and finishes in under 5 seconds.

---

## Scoring Formula

```
final_score = (
    0.25 × work_evidence_score    ← LLM: did they actually build X?
  + 0.20 × semantic_similarity    ← FAISS: does career narrative match JD?
  + 0.20 × career_quality_score   ← product company history + experience band
  + 0.15 × skills_fit_score       ← JD-weighted skill match + platform assessments
  + 0.12 × availability_score     ← behavioral signals: are they hirable now?
  + 0.08 × location_score         ← India / willing to relocate to Pune/Noida
) × honeypot_multiplier           ← 0.0 kills impossible profiles entirely
```

**Why embed career history, not the skills section?**  
Skills sections are trivially gamed — anyone can add "RAG, LangChain, GPT-4." Career descriptions are harder to fake and show what was actually built.

---

## No GPU — Designed for CPU from the Start

This system was built with the constraint that judges run `rank.py` on a standard CPU machine with no internet. Every component was chosen with this in mind:

| Component | Choice | Why CPU-friendly |
|-----------|--------|-----------------|
| Embedding model | `all-MiniLM-L6-v2` | 384-dim (not 1536-dim), fast on CPU, no GPU needed |
| Vector search | `faiss-cpu` (IndexFlatIP) | Exact search on 100K × 384 vectors in ~50ms on CPU |
| Scoring | Pure arithmetic (pandas/numpy) | No model inference at rank time |
| LLM calls | GPT-4o-mini, offline only | Zero API calls during Phase 3 |

**Phase 3 (`rank.py`) makes zero network requests and uses zero GPU. It loads pre-built files from disk and does arithmetic — that's why it finishes in 4 seconds.**

The expensive work (LLM calls, embedding 100K narratives) happens once in Phases 0–2, which have no time limit. The judges never see that cost.

---

## Results

| Metric | Score |
|--------|-------|
| NDCG@10 | **1.0000** |
| NDCG@50 | **0.8527** |
| NDCG@100 | **0.6379** |
| Honeypots in top 100 | **0 (0%)** |
| Phase 3 runtime | **4.1s on CPU / 300s budget (1.4% used)** |

Top 5 candidates:

| Rank | Candidate | Score |
|------|-----------|-------|
| 1 | CAND_0011687 | 0.8631 |
| 2 | CAND_0005260 | 0.8420 |
| 3 | CAND_0028793 | 0.8376 |
| 4 | CAND_0008425 | 0.8365 |
| 5 | CAND_0015578 | 0.8343 |

---

## Setup

**Requirements:** Python 3.11+, OpenAI API key

```bash
git clone https://github.com/PrateekApurva/resume-shortlister.git
cd resume-shortlister

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # add your OPENAI_API_KEY
```

Place the hackathon data bundle at `India_runs_data_and_ai_challenge/`.

---

## Running

### Quick start (reproduce submission.csv)

If you already have the pre-built artifacts in `artifacts/`:

```bash
python rank.py
```

Takes ~4 seconds. Outputs `submission.csv`.

### Full pipeline (build from scratch)

```bash
# Phase 0 — analyse job description (~1 min, 1 LLM call)
python analyze_jd.py

# Phase 1 — extract features + build FAISS index (~45 min, no API)
python precompute.py

# Phase 2 — LLM evidence + reasoning for top 2,000 (~30 min, ~$6.60)
python enrich.py

# Phase 3 — final ranking (4 seconds, no API)
python rank.py
```

### Validate submission

```bash
python India_runs_data_and_ai_challenge/validate_submission.py submission.csv
```

### Evaluate NDCG locally

```bash
python evaluation/evaluate.py
```

---

## Project Structure

```
analyze_jd.py              Phase 0: JD → signal schema + skill weights + query
precompute.py              Phase 1: 100K feature extraction + FAISS index
enrich.py                  Phase 2: LLM work evidence + reasoning for top 2K
rank.py                    Phase 3: final ranking (judges run this)
evaluation/evaluate.py     Local NDCG@10/50 scorer

src/
├── data/loader.py         NDJSON + JSON array candidate loader
├── features/
│   ├── honeypot.py        5-check honeypot detection
│   ├── extractor.py       Structured feature extraction per candidate
│   ├── skills.py          JD-weighted skill score
│   └── embedder.py        sentence-transformers wrapper (all-MiniLM-L6-v2)
├── scoring/
│   ├── availability.py    Behavioral signals → availability score
│   ├── career.py          Product company history + experience band
│   └── ranker.py          Final scoring formula (shared by Phase 2 + 3)
└── llm/
    ├── jd_analyzer.py     Phase 0 LLM call (structured output)
    ├── evidence.py        Phase 2 work evidence extraction (20 parallel workers)
    └── reasoning.py       Phase 2 reasoning generation (10 parallel workers)

artifacts/                 Pre-computed files (committed except candidates.faiss)
├── jd_signals.json        6 JD-derived boolean signals with weights
├── jd_skill_weights.json  21 skills with relevance weights
├── jd_focused_query.txt   Prose JD query for embedding
├── features.parquet       100K × 12 feature table (1.3 MB)
├── candidate_ids.json     FAISS position → candidate ID mapping
├── work_evidence.json     LLM signals for top 2,000 candidates
└── reasoning.json         Recruiter notes for top 200 candidates
```

> `artifacts/candidates.faiss` (146 MB) is excluded from git.  
> Rebuild with: `python precompute.py`

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `sentence-transformers` | all-MiniLM-L6-v2 — 384-dim career narrative embeddings |
| `faiss-cpu` | Exact cosine similarity search over 100K vectors |
| `openai` | GPT-4o-mini for JD analysis + work evidence extraction |
| `pandas` + `pyarrow` | Feature storage in parquet format |
| `python-docx` | Read .docx job description |
| `tqdm` | Progress bars for 100K processing loops |

---

## Honeypot Detection

Five rule-based checks eliminate fake/impossible profiles before scoring:

1. Experience years declared > actual career history duration
2. Overlapping full-time jobs at different companies
3. Claimed skills that contradict career history domain
4. Degree end dates in wrong chronological order
5. Job titles that don't match the company's domain

Any candidate triggering 2+ checks gets `honeypot_multiplier = 0.0` → final score = 0.

---

## Cost & Time Summary

| Phase | Time | Cost |
|-------|------|------|
| Phase 0 | < 1 min | ~$0.01 |
| Phase 1 | ~45 min | $0 |
| Phase 2 | ~30 min | ~$6.60 |
| Phase 3 | 4 seconds | $0 |
