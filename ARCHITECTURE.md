# Resume Shotlister — Complete Architecture & Pipeline Design

> **Project:** Redrob "Intelligent Candidate Discovery & Ranking Challenge"
> **Goal:** Rank 100,000 candidates for a Senior AI Engineer JD, output top 100
> **Hard constraint:** Ranking step ≤ 5 minutes, CPU only, no internet, ≤ 16 GB RAM

---

## Current Status (Session Handoff)

| Item | Status |
|---|---|
| Architecture design | COMPLETE — read this file fully before writing any code |
| Project setup | NOT STARTED |
| Phase 0 — analyze_jd.py | NOT STARTED |
| Phase 1 — precompute.py | NOT STARTED |
| Phase 2 — enrich.py | NOT STARTED |
| Phase 3 — rank.py | NOT STARTED |
| Evaluation | NOT STARTED |
| Streamlit demo | NOT STARTED |

**Next action:** Set up the project (git, folders, requirements.txt), then build Phase 0.

---

## Table of Contents

1. [The Problem in Plain Language](#1-the-problem-in-plain-language)
2. [What the Data Looks Like](#2-what-the-data-looks-like)
3. [System Overview — Four Phases](#3-system-overview--four-phases)
4. [Phase 0 — JD Analysis (analyze_jd.py)](#4-phase-0--jd-analysis)
5. [Phase 1 — Offline Preprocessing (precompute.py)](#5-phase-1--offline-preprocessing)
6. [Phase 2 — LLM Enrichment (enrich.py)](#6-phase-2--llm-enrichment)
7. [Phase 3 — Online Ranking (rank.py)](#7-phase-3--online-ranking)
8. [The Scoring Formula — Fully Explained](#8-the-scoring-formula--fully-explained)
9. [Tech Terms Glossary (Beginner-Friendly)](#9-tech-terms-glossary)
10. [Three Candidate Walkthroughs](#10-three-candidate-walkthroughs)
    - [Candidate A: Ela Singh — Strong Fit](#candidate-a-ela-singh--cand_0000031)
    - [Candidate B: Anil Bose — Honeypot Trap](#candidate-b-anil-bose--cand_0000004)
    - [Candidate C: Ira Vora — Keyword Stuffer](#candidate-c-ira-vora--cand_0000001)
11. [Final Score Comparison](#11-final-score-comparison)
12. [Artifact Reference — What Each File Contains](#12-artifact-reference)
13. [Project File Structure](#13-project-file-structure)

---

## 1. The Problem in Plain Language

Imagine you are a recruiter at a startup. You receive 100,000 resumes for one job. You have to find the 100 best people. How do you do it?

A bad recruiter uses keyword filters:
> "JD says 'RAG' → show me everyone who has 'RAG' in their skills section."

A great recruiter thinks differently:
> "This person built a semantic search system at Swiggy serving millions of users.
> Their skills section doesn't mention 'RAG' but their work clearly shows retrieval expertise.
> That's more valuable than someone who listed 'RAG, LangChain, GPT-4' in skills
> but never built anything real."

**Our system must behave like the great recruiter.**

The hackathon JD even explicitly warns us:
> *"The right answer involves reasoning about the gap between what the JD says and what
> the JD means. A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their
> profile, but if their career history shows they built a recommendation system at a product
> company, they're a fit."*

---

## 2. What the Data Looks Like

Each candidate is one JSON object (one line in the NDJSON file) with 6 blocks:

```
candidate = {
    "candidate_id": "CAND_0000031",

    "profile": {
        headline, summary,           ← self-written          [MEDIUM TRUST]
        years_of_experience,         ← declared number
        current_title,
        current_company,
        current_company_size,
        current_industry,
        location, country
    },

    "career_history": [              ← list of past jobs     [HIGH TRUST]
        {
            company, title,
            start_date, end_date,
            duration_months,
            industry, company_size,
            description              ← GOLD MINE: what they actually did
        }, ...
    ],

    "education": [                   ← degrees + tier        [HIGH TRUST]
        {
            institution, degree,
            field_of_study,
            start_year, end_year,
            tier                     ← tier_1 > tier_2 > tier_3 > tier_4
        }, ...
    ],

    "skills": [                      ← self-declared         [LOW TRUST]
        {
            name,
            proficiency,             ← beginner/intermediate/advanced/expert
            endorsements,            ← peer-validated, slightly more trusted
            duration_months          ← self-reported time using this skill
        }, ...
    ],

    "redrob_signals": {              ← platform-observed     [HIGHEST TRUST]
        profile_completeness_score,
        last_active_date,
        open_to_work_flag,
        recruiter_response_rate,
        avg_response_time_hours,
        skill_assessment_scores,     ← platform test results, overrides self-claim
        notice_period_days,
        willing_to_relocate,
        github_activity_score,
        interview_completion_rate,
        offer_acceptance_rate,
        ... 23 signals total
    }
}
```

**Trust hierarchy — a core design principle:**

| Data Block | Trust | Why |
|---|---|---|
| `redrob_signals` | Highest | Platform-observed behaviour, cannot be faked |
| `career_history.description` | High | Hard to write convincingly about work you never did |
| `education` | High | Fixed, verifiable facts |
| `profile` headline/summary | Medium | Self-written but public-facing |
| `skills` | Lowest | One-click to add, no verification |

This hierarchy is the foundation of our scoring — we weight signals in proportion to how much we trust them.

---

## 3. System Overview — Four Phases

### 3.1 Brief Overview (Read This First)

**The big picture in one sentence:**
> Understand the JD → preprocess 100K candidates offline → use LLM to judge the top 2K → rank the best 100 in under 5 minutes.

```
  job_description.txt                candidates.json (100K, 487MB)
         │                                      │
         ▼                                      │
 ┌───────────────────────────────────────────┐  │
 │  PHASE 0 · analyze_jd.py                 │  │
 │  Ask LLM once: "What matters in this JD?"│  │
 │                                           │  │
 │  OUT → jd_signals.json         (signals) │  │
 │        jd_skill_weights.json   (weights) │  │
 │        jd_focused_query.txt    (query)   │  │
 └───────────────────┬───────────────────────┘  │
                     │                          │
                     ▼                          ▼
 ┌─────────────────────────────────────────────────────────┐
 │  PHASE 1 · precompute.py                                │
 │  Process all 100K candidates once (~45 min, offline)    │
 │                                                         │
 │  For every candidate:                                   │
 │    ① Detect honeypot (5 rule checks)                   │
 │    ② Extract features (location, career, availability) │
 │    ③ Score skills (weighted by JD + platform tests)    │
 │    ④ Build work narrative (career text only, no skills)│
 │  Then for all 100K together:                            │
 │    ⑤ Embed work narratives → 100K × 384-dim vectors    │
 │    ⑥ Store vectors in FAISS index                      │
 │                                                         │
 │  OUT → features.parquet     (100K structured features) │
 │        candidates.faiss     (FAISS vector index)        │
 │        candidate_ids.json   (position → candidate_id)  │
 └───────────────────────────┬─────────────────────────────┘
                             │
                             ▼
 ┌─────────────────────────────────────────────────────────┐
 │  PHASE 2 · enrich.py                                    │
 │  LLM enrichment for top 2,000 only (~30 min, ~$6)       │
 │                                                         │
 │    ① FAISS search: embed JD → find top 2,000           │
 │    ② LLM reads career history of each → extracts       │
 │       work evidence JSON (did they actually build X?)   │
 │    ③ Re-score all 2,000 with full formula → top 200     │
 │    ④ LLM writes 1-2 sentence reasoning for top 200     │
 │                                                         │
 │  OUT → work_evidence.json   (LLM signals, 2K entries)  │
 │        reasoning.json       (text reasoning, 200 entries)│
 └───────────────────────────┬─────────────────────────────┘
                             │
                             ▼
 ┌─────────────────────────────────────────────────────────┐
 │  PHASE 3 · rank.py                                      │
 │  Produce the final CSV (≤ 5 min, CPU only, no internet) │
 │                                                         │
 │    ① Load all artifacts from disk                      │
 │    ② Embed JD → FAISS search → top 2,000 (~50ms)      │
 │    ③ Score each using the formula (no LLM, pure math)  │
 │    ④ Sort → take top 100 → attach reasoning → CSV      │
 │                                                         │
 │  OUT → submission.csv  (100 ranked candidates)          │
 └─────────────────────────────────────────────────────────┘
```

**Why offline phases at all?**
The judges time only `rank.py`. Phases 0–2 are prep work with no time limit.
By doing all heavy lifting offline (LLM calls, embedding 100K texts, FAISS build),
Phase 3 only needs to load files and do arithmetic — that's why it finishes in under 2 minutes.

---

### 3.2 Phase Summary Table

| Phase | Script | Runs | Time | Cost | Key output |
|---|---|---|---|---|---|
| 0 — JD Analysis | `analyze_jd.py` | Once | < 1 min | ~$0.01 | Signal schema, skill weights, focused query |
| 1 — Preprocessing | `precompute.py` | Once | ~45 min | $0 | FAISS index, 100K feature table |
| 2 — LLM Enrichment | `enrich.py` | Once | ~30 min | ~$6 | Work evidence + reasoning for top 2K/200 |
| 3 — Ranking | `rank.py` | Every submission | < 2 min | $0 | `submission.csv` (top 100) |

---

### 3.3 Detailed Architecture Diagram (Reference)

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    RESUME SHOTLISTER — SYSTEM ARCHITECTURE                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

 INPUTS
 ┌────────────────────────┐          ┌────────────────────────────────────────┐
 │  job_description.txt   │          │  candidates.json                       │
 │                        │          │  100,000 candidates · NDJSON · 487 MB  │
 │  Raw JD text from the  │          │  One JSON object per line              │
 │  hackathon bundle      │          │  6 blocks per candidate                │
 └───────────┬────────────┘          └──────────────────────┬─────────────────┘
             │                                              │
             │                                              │
             ▼                                              │
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  PHASE 0 · JD Analysis  (analyze_jd.py)                                   │
 │  Run once · No time limit · 1 LLM call                                    │
 ├───────────────────────────────────────────────────────────────────────────┤
 │                                                                            │
 │   job_description.txt ──────────────► [ LLM : GPT-4o-mini ]               │
 │                                              │                            │
 │                          ┌───────────────────┼──────────────────┐         │
 │                          ▼                   ▼                  ▼         │
 │              ┌─────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
 │              │ Work Evidence   │  │ Skill Relevance  │  │ Focused JD   │ │
 │              │ Signals         │  │ Weights          │  │ Query        │ │
 │              │                 │  │                  │  │              │ │
 │              │ 6-8 yes/no Qs   │  │ skill → weight   │  │ 3-5 sentence │ │
 │              │ to ask about    │  │ 1.0 = required   │  │ prose of     │ │
 │              │ career history  │  │ 0.5 = nice       │  │ what role    │ │
 │              │ weights sum=1.0 │  │ -ve = wrong fit  │  │ ACTUALLY     │ │
 │              └────────┬────────┘  └────────┬─────────┘  │ needs        │ │
 │                       │                    │            └──────┬───────┘ │
 └───────────────────────┼────────────────────┼───────────────────┼─────────┘
                         │                    │                   │
                         ▼                    ▼                   ▼
              ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
              │ jd_signals.json  │  │jd_skill_weights  │  │jd_focused_query  │
              │                  │  │.json             │  │.txt              │
              │ signal schema +  │  │                  │  │                  │
              │ weights for LLM  │  │ used in Phase 1  │  │ embedded in      │
              │ eval in Phase 2  │  │ skill scoring    │  │ Phase 1 & 3      │
              └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                       │                     │                      │
                       │            ┌────────┘                      │
                       │            │         ┌─────────────────────┘
                       │            ▼         ▼
                       │   ┌────────────────────────────────────────────────────┐
                       │   │  PHASE 1 · Offline Preprocessing (precompute.py)  │
                       │   │  Run once · ~30–60 min · No time limit            │
                       │   ├────────────────────────────────────────────────────┤
                       │   │                                                    │
                       │   │  candidates.json ─► [ Load 100K NDJSON in batches ]│
                       │   │                              │                    │
                       │   │             ┌────────────────▼─────────────────┐  │
                       │   │             │  For each of 100,000 candidates  │  │
                       │   │             ├──────────────────────────────────┤  │
                       │   │             │  ① Honeypot Detection            │  │
                       │   │             │    5 rule-based checks           │  │
                       │   │             │    → honeypot_multiplier: 0 or 1 │  │
                       │   │             ├──────────────────────────────────┤  │
                       │   │             │  ② Feature Extraction            │  │
                       │   │             │    location_score                │  │
                       │   │             │    career_quality_score          │  │
                       │   │             │    availability_score            │  │
                       │   │             │    education_score               │  │
                       │   │             ├──────────────────────────────────┤  │
                       │   │             │  ③ Skill Score                   │  │
                       │   │             │    JD-weighted match  ◄──────────┼──┼── jd_skill_weights.json
                       │   │             │    platform assessment blend     │  │
                       │   │             │    → skills_fit_score            │  │
                       │   │             ├──────────────────────────────────┤  │
                       │   │             │  ④ Work Narrative                │  │
                       │   │             │    career descriptions ONLY      │  │
                       │   │             │    (no skills section)           │  │
                       │   │             └────────────────┬─────────────────┘  │
                       │   │                              │                    │
                       │   │             ┌────────────────▼─────────────────┐  │
                       │   │             │  After all 100K are processed    │  │
                       │   │             ├──────────────────────────────────┤  │
                       │   │             │  ⑤ Batch Embed Work Narratives   │  │
                       │   │             │    model: all-MiniLM-L6-v2       │  │
                       │   │             │    100K texts → 100K × 384 vecs  │  │
                       │   │             │    ~3–4 min on CPU               │  │
                       │   │             ├──────────────────────────────────┤  │
                       │   │             │  ⑥ Build FAISS Index             │  │
                       │   │             │    IndexFlatIP(384)              │  │
                       │   │             │    add all 100K vectors          │  │
                       │   │             └────────────────┬─────────────────┘  │
                       │   └────────────────────────────────────────────────────┘
                       │                                  │
                       │         ┌────────────────────────┤
                       │         │                        │
                       │  ┌──────▼──────────┐  ┌─────────▼──────────┐  ┌──────────────────┐
                       │  │features.parquet │  │candidates.faiss    │  │candidate_ids.json│
                       │  │                 │  │                    │  │                  │
                       │  │100K rows × 15   │  │100K × 384-dim      │  │list mapping      │
                       │  │features         │  │embedding vectors   │  │FAISS position    │
                       │  │indexed by       │  │                    │  │→ candidate_id    │
                       │  │candidate_id     │  │                    │  │                  │
                       │  └──────┬──────────┘  └─────────┬──────────┘  └────────┬─────────┘
                       │         │                        │                      │
                       └─────────┼──────────┬─────────────┘                      │
                                 │          │                   ┌────────────────┘
                                 ▼          ▼                   ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │  PHASE 2 · LLM Enrichment  (enrich.py)                 │
                       │  Run once · ~20–30 min · ~$6 API cost · No time limit  │
                       ├─────────────────────────────────────────────────────────┤
                       │                                                         │
                       │  Step 1 · FAISS Search                                 │
                       │  embed jd_focused_query ──► FAISS ──► top 2,000        │
                       │  (~50 milliseconds)                     candidates      │
                       │                                         │               │
                       │              ┌──────────────────────────▼────────────┐ │
                       │              │  For each of top 2,000 candidates     │ │
                       │              ├───────────────────────────────────────┤ │
                       │              │  Step 2 · LLM Work Evidence           │ │
                       │              │  career_history ──► GPT-4o-mini       │ │
                       │              │  signals from jd_signals.json  ◄──────┼─┼── jd_signals.json
                       │              │  → boolean JSON per signal            │ │
                       │              │  cost: ~$0.003 per candidate          │ │
                       │              └──────────────────────┬────────────────┘ │
                       │                                     │                  │
                       │  Step 3 · Re-score All 2,000                           │
                       │  full formula (work+semantic+career+skills+avail+loc)  │
                       │  → sort descending → take top 200                      │
                       │                                     │                  │
                       │              ┌──────────────────────▼────────────────┐ │
                       │              │  For each of top 200 candidates       │ │
                       │              ├───────────────────────────────────────┤ │
                       │              │  Step 4 · LLM Reasoning               │ │
                       │              │  score + profile facts ──► LLM        │ │
                       │              │  → 1-2 honest sentence per candidate  │ │
                       │              └──────────────────────┬────────────────┘ │
                       └──────────────────────────────────────┼─────────────────┘
                                                              │
                                      ┌───────────────────────┤
                                      │                       │
                          ┌───────────▼──────────┐  ┌────────▼───────────────┐
                          │ work_evidence.json    │  │ reasoning.json         │
                          │                       │  │                        │
                          │ 2,000 entries         │  │ 200 entries            │
                          │ boolean signals per   │  │ 1-2 sentence text      │
                          │ candidate from LLM    │  │ per candidate from LLM │
                          └───────────┬──────────┘  └────────┬───────────────┘
                                      │                       │
                                      └───────────┬───────────┘
                                                  │
                                                  ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │  PHASE 3 · Online Ranking  (rank.py)                    │
                       │  Run on demand · ≤ 5 min · CPU only · No internet       │
                       ├─────────────────────────────────────────────────────────┤
                       │                                                         │
                       │  Load 6 artifacts from disk  (~2 seconds)              │
                       │                                                         │
                       │  Embed jd_focused_query ──► FAISS ──► top 2,000        │
                       │                                         (~50ms)         │
                       │                                         │               │
                       │  For each of 2,000 · pure arithmetic · no LLM:         │
                       │  ┌───────────────────────────────────────────────────┐ │
                       │  │                                                   │ │
                       │  │  final_score =                                    │ │
                       │  │    0.25 × work_evidence_score ◄─ work_evidence.json│ │
                       │  │  + 0.20 × semantic_similarity ◄─ FAISS distances  │ │
                       │  │  + 0.20 × career_quality_score◄─ features.parquet │ │
                       │  │  + 0.15 × skills_fit_score    ◄─ features.parquet │ │
                       │  │  + 0.12 × availability_score  ◄─ features.parquet │ │
                       │  │  + 0.08 × location_score      ◄─ features.parquet │ │
                       │  │  ×       honeypot_multiplier  ◄─ features.parquet │ │
                       │  │                                                   │ │
                       │  └───────────────────────────────────────────────────┘ │
                       │                                         │               │
                       │  Sort all 2,000 by score                               │
                       │  → take top 100                                         │
                       │  → look up reasoning per candidate ◄── reasoning.json  │
                       │  → write CSV                                            │
                       └──────────────────────────────────────┬──────────────────┘
                                                              │
                                                              ▼
                       ┌─────────────────────────────────────────────────────────┐
                       │  submission.csv                                          │
                       │                                                         │
                       │  candidate_id  │ rank │  score  │ reasoning            │
                       │  ──────────────┼──────┼─────────┼───────────────────── │
                       │  CAND_0000031  │  1   │  0.9260 │ "Ela Singh brings 6  │
                       │                │      │         │  years building       │
                       │                │      │         │  ranking systems..."  │
                       │  CAND_XXXXXXX  │  2   │  0.9100 │ "..."                │
                       │  ...           │  ... │  ...    │  ...                 │
                       │  CAND_XXXXXXX  │  100 │  0.6200 │ "..."                │
                       └─────────────────────────────────────────────────────────┘
```

---

### 3.2 Artifact Flow Diagram

This shows exactly which artifacts are produced by each phase and consumed by which.

```
                    PRODUCED BY          CONSUMED BY
                    ┌──────────┐         ┌───────────────────────────┐
jd_signals.json     │  Phase 0 ├────────►│  Phase 2 (LLM eval)       │
                    │          │         │  Phase 3 (score compute)   │
                    └──────────┘         └───────────────────────────┘

                    ┌──────────┐         ┌───────────────────────────┐
jd_skill_weights    │  Phase 0 ├────────►│  Phase 1 (skill scoring)  │
.json               └──────────┘         └───────────────────────────┘

                    ┌──────────┐         ┌───────────────────────────┐
jd_focused_query    │  Phase 0 ├────────►│  Phase 1 (JD embedding)   │
.txt                │          │         │  Phase 2 (FAISS search)   │
                    └──────────┘         │  Phase 3 (FAISS search)   │
                                         └───────────────────────────┘

                    ┌──────────┐         ┌───────────────────────────┐
features.parquet    │  Phase 1 ├────────►│  Phase 2 (re-scoring)     │
                    │          │         │  Phase 3 (scoring formula) │
                    └──────────┘         └───────────────────────────┘

                    ┌──────────┐         ┌───────────────────────────┐
candidates.faiss    │  Phase 1 ├────────►│  Phase 2 (FAISS search)   │
                    │          │         │  Phase 3 (FAISS search)   │
                    └──────────┘         └───────────────────────────┘

                    ┌──────────┐         ┌───────────────────────────┐
candidate_ids       │  Phase 1 ├────────►│  Phase 2 (id lookup)      │
.json               │          │         │  Phase 3 (id lookup)      │
                    └──────────┘         └───────────────────────────┘

                    ┌──────────┐         ┌───────────────────────────┐
work_evidence       │  Phase 2 ├────────►│  Phase 3 (work_score)     │
.json               └──────────┘         └───────────────────────────┘

                    ┌──────────┐         ┌───────────────────────────┐
reasoning.json      │  Phase 2 ├────────►│  Phase 3 (CSV column)     │
                    └──────────┘         └───────────────────────────┘
```

---

### 3.3 Why Four Phases?

The judges impose a **5-minute wall-clock limit** on `rank.py`. All the expensive work
(LLM calls, embedding 100K texts, building a FAISS index) must happen beforehand
and be saved as artifacts. Phase 3 just loads those artifacts and does fast arithmetic.

This pattern — heavy offline precomputation, fast online serving — is exactly how
real production ML systems work at Swiggy, LinkedIn, and Spotify.

---

## 4. Phase 0 — JD Analysis

**Script:** `analyze_jd.py`
**Runs:** Once, before anything else.
**Input:** `job_description.txt`
**Outputs:** 3 artifact files

This phase reads the raw JD and uses the LLM to understand what the role
actually needs. Nothing here is hardcoded — run it with a different JD
and you get different signals, different skill weights, different query.

### What the LLM is asked to do

**Prompt (sent once):**
```
You are a senior technical recruiter. Read this job description carefully.

JD TEXT:
[full contents of job_description.txt]

Produce three outputs:

--- OUTPUT 1: work_evidence_signals ---
6-8 signals that reveal whether a candidate's WORK HISTORY (not skills list)
demonstrates genuine fit for this role.
Each signal must be answerable as true/false from reading career descriptions.
Weights must sum to 1.0.

Return as JSON:
{
  "signals": [
    {
      "key":      "snake_case_identifier",
      "question": "Did they do X in actual work history?",
      "weight":   0.0 to 1.0
    }
  ],
  "red_flag_patterns": ["list of disqualifying career patterns"],
  "preferred_company_types": ["product", "startup", "AI-native"]
}

--- OUTPUT 2: skill_relevance_weights ---
For each skill that might appear in a candidate's skills section, assign:
  1.0   = required, core to the role
  0.5-0.9 = useful, nice to have
  0.0   = irrelevant
  negative = unwanted (wrong domain for this role)

Return as JSON: { "skill_name_lowercase": weight, ... }

--- OUTPUT 3: focused_query ---
Write a 3-5 sentence focused summary of what this role ACTUALLY needs.
This will be embedded and compared against candidate work narratives.
Emphasise the work done, not the keyword list.
Return as plain text.
```

**For our Senior AI Engineer JD, the LLM returns:**

`jd_signals.json`:
```json
{
  "signals": [
    {"key": "built_production_ai",        "question": "Did they ship ML/AI systems to real users in production?",        "weight": 0.25},
    {"key": "vector_search_built",        "question": "Did they build semantic or vector search in a live product?",      "weight": 0.20},
    {"key": "ranking_system_built",       "question": "Did they build a ranking or recommendation pipeline?",             "weight": 0.20},
    {"key": "evaluation_framework_built", "question": "Did they build offline/online eval (NDCG, A/B tests, MRR)?",      "weight": 0.15},
    {"key": "embeddings_in_production",   "question": "Did they use text embeddings in a live production system?",        "weight": 0.10},
    {"key": "rag_or_retrieval_built",     "question": "Did they build RAG, information retrieval, or hybrid search?",    "weight": 0.10}
  ],
  "red_flag_patterns": [
    "entire career at IT consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini)",
    "pure academic/research background with no production deployment",
    "primary expertise in computer vision or speech without NLP/IR exposure"
  ],
  "preferred_company_types": ["product", "startup", "AI-native", "internet"]
}
```

`jd_skill_weights.json`:
```json
{
  "embeddings": 1.0,
  "vector search": 1.0,
  "faiss": 1.0,
  "pinecone": 1.0,
  "weaviate": 1.0,
  "qdrant": 1.0,
  "milvus": 1.0,
  "sentence transformers": 1.0,
  "information retrieval": 0.9,
  "ranking": 0.9,
  "recommendation systems": 0.9,
  "learning to rank": 0.9,
  "nlp": 0.8,
  "python": 0.8,
  "ndcg": 0.9,
  "mrr": 0.8,
  "fine-tuning": 0.6,
  "lora": 0.5,
  "qlora": 0.5,
  "computer vision": -0.2,
  "speech recognition": -0.2,
  "object detection": -0.2,
  "tts": -0.1
}
```

`jd_focused_query.txt`:
```
Senior AI engineer with production experience building embedding-based retrieval and
ranking systems at product companies. Has shipped learning-to-rank models, vector search
pipelines, and A/B evaluation frameworks to real users at scale. Strong Python engineering
skills and experience designing offline evaluation metrics (NDCG, MRR, MAP). Based in
India or willing to relocate to Pune or Noida. Actively looking for a senior IC role.
```

This focused query — not the raw JD — is what gets embedded and used for semantic search.

---

## 5. Phase 1 — Offline Preprocessing

**Script:** `precompute.py`
**Runs:** Once. Takes ~30–60 minutes. No time limit.
**Inputs:** `candidates.json` (100K NDJSON), all 3 Phase 0 artifacts
**Outputs:** `features.parquet`, `candidates.faiss`, `candidate_ids.json`

```
For each of 100,000 candidates (in order):
    Step 1 → Honeypot detection        → flag is_honeypot (True/False)
    Step 2 → Structured feature extract → ~15 numeric/boolean features
    Step 3 → Skill score computation   → skills_fit_score ∈ [0, 1]
    Step 4 → Build work narrative      → text string (career descriptions only)

After all 100K:
    Step 5 → Batch embed work narratives → FAISS index
    Step 6 → Save all artifacts to disk
```

### Step 1: Honeypot Detection

The dataset contains ~80 "honeypot" candidates — profiles with subtle impossibilities
planted to test if your system is actually reading profiles or just doing keyword
matching. Getting >10% honeypots in your top 100 means disqualification.

We detect honeypots using **5 rule-based checks**. If 2+ checks fire → honeypot.

```python
def detect_honeypot(candidate) -> bool:
    signals = []

    # Check 1: Expert/Advanced proficiency claimed with 0 months of usage
    # Real experts don't list 0 months on their primary skills
    zero_duration_experts = [
        s for s in candidate["skills"]
        if s["proficiency"] in ["advanced", "expert"] and s["duration_months"] == 0
    ]
    signals.append(len(zero_duration_experts) >= 2)

    # Check 2: Total career history months vs stated years_of_experience
    # Gap > 4 years is suspicious (some gap is fine — freelance, gap years, etc.)
    total_career_months = sum(j["duration_months"] for j in candidate["career_history"])
    stated_years = candidate["profile"]["years_of_experience"]
    signals.append(abs(stated_years - total_career_months / 12) > 4)

    # Check 3: Overlapping full-time jobs
    # Two jobs with overlapping date ranges AND both listed as full-time → impossible
    signals.append(has_overlapping_full_time_jobs(candidate["career_history"]))

    # Check 4: Impossible education timeline
    # e.g. PhD end year < B.Tech start year, or PhD started before Bachelor's ended
    signals.append(has_impossible_education_sequence(candidate["education"]))

    # Check 5: Job title vs description gross mismatch
    # Title = "Marketing Manager", description = "designed CAD subsystems in SolidWorks"
    signals.append(has_title_description_mismatch(candidate["career_history"]))

    return sum(signals) >= 2
```

**Result stored in features:** `honeypot_multiplier = 0.0` if honeypot, else `1.0`.
A score multiplied by 0.0 is always 0 — honeypots can never enter the top 100.

### Step 2: Structured Feature Extraction

Convert raw candidate JSON into a flat set of numbers Phase 3 can use directly.

```python
features = {
    # ── Experience ──────────────────────────────────────────────────────
    "years_of_experience":   6.0,   # from profile.years_of_experience
    "experience_fit_score":  0.90,  # bell-curve: peak at 5-8 yrs for this JD
                                    # <3 yrs = 0.2, 4 yrs = 0.6, 5-8 yrs = 1.0,
                                    # 9 yrs = 0.9, 12+ yrs = 0.7 (over-qualified risk)

    # ── Location ────────────────────────────────────────────────────────
    "is_india_based":        True,  # country == "IN"
    "willing_to_relocate":   True,  # from redrob_signals.willing_to_relocate
    "location_score":        1.00,  # 1.0 = India + willing
                                    # 0.7 = India only (not willing)
                                    # 0.3 = outside India but willing
                                    # 0.0 = outside India + not willing

    # ── Career Quality ──────────────────────────────────────────────────
    "consulting_ratio":      0.00,  # months at consulting firms / total months
    "has_engineering_title": True,  # at least one job title has engineer/scientist/dev
    "career_quality_score":  0.95,  # see formula below

    # ── Education ───────────────────────────────────────────────────────
    "highest_edu_tier":      2,     # 1 = IIT/IISc, 2 = NIT/SRM, 3 = others, 4 = unknown
    "edu_relevant_field":    True,  # CS/CE/EE/Maths/Statistics/AI/ML
    "education_score":       0.70,  # tier_1=1.0, tier_2=0.7, tier_3=0.5, tier_4=0.3

    # ── Skills ──────────────────────────────────────────────────────────
    "skills_fit_score":      0.82,  # computed in Step 3

    # ── Behavioral Availability ─────────────────────────────────────────
    "days_since_last_active":   22,
    "open_to_work":          True,
    "recruiter_response_rate": 0.91,
    "notice_period_days":     60,
    "interview_completion":   0.60,
    "availability_score":     0.856, # computed in Step 2 (sub-formula below)

    # ── Honeypot ────────────────────────────────────────────────────────
    "is_honeypot":           False,
    "honeypot_multiplier":   1.0,
}
```

**career_quality_score sub-formula:**

```python
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "hexaware", "ltimindtree", "mindtree"
}

def compute_career_quality(candidate):
    jobs = candidate["career_history"]
    total_months = sum(j["duration_months"] for j in jobs)
    if total_months == 0:
        return 0.0

    consulting_months = sum(
        j["duration_months"] for j in jobs
        if any(firm in j["company"].lower() for firm in CONSULTING_FIRMS)
    )
    consulting_ratio = consulting_months / total_months

    # Base: 1.0 = all product companies, 0.0 = all consulting
    base = 1.0 - consulting_ratio

    # Title bonus: engineering roles get a slight boost
    has_eng_titles = any(
        any(word in j["title"].lower()
            for word in ["engineer", "scientist", "developer", "architect", "researcher"])
        for j in jobs
    )
    title_factor = 1.05 if has_eng_titles else 0.90

    # Experience fit: penalise too junior or too senior for this JD
    yoe = candidate["profile"]["years_of_experience"]
    if yoe < 3:
        exp_factor = 0.4
    elif yoe < 5:
        exp_factor = 0.75
    elif yoe <= 9:
        exp_factor = 1.0     # sweet spot for this JD
    elif yoe <= 12:
        exp_factor = 0.85
    else:
        exp_factor = 0.70    # potentially over-qualified / out of touch

    return min(base * title_factor * exp_factor, 1.0)
```

Note: `experience_fit_score` from earlier becomes the `exp_factor` inside
`career_quality_score`. It is not a separate scoring component — it adjusts
career quality. This keeps the final formula clean.

**availability_score sub-formula:**

```python
def compute_availability(signals, today):
    days_inactive = (today - parse_date(signals["last_active_date"])).days
    recency   = max(0.0, 1.0 - days_inactive / 180)   # 0 if inactive > 6 months
    open_work = 1.0 if signals["open_to_work_flag"] else 0.6
    response  = float(signals["recruiter_response_rate"])
    notice    = max(0.0, 1.0 - signals["notice_period_days"] / 180)
    interview = float(signals["interview_completion_rate"])

    return (
        0.30 * recency
      + 0.25 * open_work
      + 0.20 * response
      + 0.15 * notice
      + 0.10 * interview
    )
```

### Step 3: Skill Score Computation

Skills are low-trust but not zero-trust. We weight them by:
- Relevance to JD (from `jd_skill_weights.json` — generated in Phase 0)
- Claimed proficiency level
- Platform assessment score (if available — overrides self-claim)
- Duration of usage (capped at 5 years)

```python
PROFICIENCY_WEIGHT = {
    "beginner": 0.3, "intermediate": 0.6, "advanced": 0.9, "expert": 1.0
}

def compute_skill_score(candidate, jd_skill_weights):
    total_weighted = 0.0
    max_possible   = 0.0

    for skill in candidate["skills"]:
        name      = skill["name"].lower()
        relevance = jd_skill_weights.get(name, 0.0)

        if relevance == 0.0:
            continue   # irrelevant skill — skip entirely

        # Start with self-claimed proficiency
        proficiency = PROFICIENCY_WEIGHT[skill["proficiency"]]

        # If platform tested this skill, blend test score with self-claim
        # Platform score is more reliable → we give it 60% weight
        assessments = candidate["redrob_signals"]["skill_assessment_scores"]
        if skill["name"] in assessments:
            platform_score = assessments[skill["name"]] / 100.0
            proficiency = 0.40 * proficiency + 0.60 * platform_score

        # Duration factor: longer use = more credibility (cap at 60 months = 5 years)
        duration_factor = 0.5 + 0.5 * min(skill["duration_months"], 60) / 60
        # duration_factor ranges from 0.5 (0 months) to 1.0 (60+ months)

        contribution = relevance * proficiency * duration_factor
        # Negative relevance (e.g. speech recognition for AI eng role) reduces total
        total_weighted += contribution
        max_possible   += abs(relevance)

    if max_possible == 0.0:
        return 0.0

    return max(0.0, min(total_weighted / max_possible, 1.0))
```

### Step 4: Build Work Narrative

This is the text that gets embedded into the FAISS index.
We use **only** career history descriptions — NOT the skills section.

**Why only career descriptions?**

If we embed the full profile including skills, a candidate who lists
"RAG, FAISS, LangChain, Pinecone, Embeddings" in their skills will have a
very similar embedding to the JD — even if they never built any of these.
By embedding only work descriptions, their narrative reflects what they
*actually did*, not what they *claim to know*.

```python
def build_work_narrative(candidate):
    parts = []
    for job in candidate["career_history"]:
        company_type = classify_company(job["company"])  # "product" or "consulting"
        parts.append(
            f"Role: {job['title']} at {job['company']} "
            f"({company_type}, {job['industry']}, {job['duration_months']} months)\n"
            f"Work done: {job['description']}"
        )
    return "\n\n---\n\n".join(parts)
```

### Step 5: Embed and Build FAISS Index

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
# all-MiniLM-L6-v2: 384-dim vectors, fast on CPU, good English quality, free

# Load the focused JD query from Phase 0 (NOT the raw JD text)
jd_query = open("artifacts/jd_focused_query.txt").read()

# Embed all 100K work narratives in batches
# ~500 texts/second on a modern CPU → 100K takes ~200 seconds
narratives = [build_work_narrative(c) for c in candidates]
embeddings = model.encode(narratives, batch_size=64, show_progress_bar=True)
# Result shape: (100000, 384) — one 384-dim vector per candidate

# Normalize vectors so inner product = cosine similarity
faiss.normalize_L2(embeddings)

# Build FAISS index
# IndexFlatIP = exact search using inner product (= cosine similarity on normalized vectors)
# For 100K candidates this is fast enough. For 10M+ we'd use IndexIVFFlat (approximate).
index = faiss.IndexFlatIP(384)
index.add(embeddings.astype("float32"))

# Save everything
faiss.write_index(index, "artifacts/candidates.faiss")

candidate_ids = [c["candidate_id"] for c in candidates]
json.dump(candidate_ids, open("artifacts/candidate_ids.json", "w"))

features_df = pd.DataFrame(all_features)
features_df.set_index("candidate_id", inplace=True)   # index = candidate_id for fast .loc[]
features_df.to_parquet("artifacts/features.parquet")
```

**Total Phase 1 time breakdown:**
- Loading 100K NDJSON: ~2 min
- Feature extraction: ~5 min
- Embedding 100K texts: ~3–4 min
- Building FAISS index: ~10 seconds
- **Total: ~30–45 minutes. One-time cost.**

---

## 6. Phase 2 — LLM Enrichment

**Script:** `enrich.py`
**Runs:** Once. ~20–30 minutes. ~$6 API cost. No time limit.
**Inputs:** All Phase 0 + Phase 1 artifacts, `candidates.json`
**Outputs:** `work_evidence.json`, `reasoning.json`

### Step 1: FAISS Search → Top 2,000

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
jd_query = open("artifacts/jd_focused_query.txt").read()  # focused query from Phase 0
jd_embedding = model.encode([jd_query])
faiss.normalize_L2(jd_embedding)

index = faiss.read_index("artifacts/candidates.faiss")
distances, indices = index.search(jd_embedding.astype("float32"), k=2000)

candidate_ids = json.load(open("artifacts/candidate_ids.json"))
top_2000_ids  = [candidate_ids[i] for i in indices[0]]
top_2000_sims = {candidate_ids[i]: float(distances[0][j])
                 for j, i in enumerate(indices[0])}
```

Why 2,000? It's a wide enough net that the true top 100 is almost certainly inside,
yet small enough that LLM costs stay manageable (~$6).

### Step 2: Work Evidence Extraction (LLM on top 2,000)

For each of the top 2,000 candidates, we ask the LLM to evaluate their career
history against the JD-derived signals from `jd_signals.json`.

**Prompt template (per candidate):**
```
You are an expert technical recruiter.

Job context: [one-line summary from jd_signals.json]

Candidate career history (read this carefully — no skills section):
[career_history descriptions concatenated]

For each signal below, answer TRUE or FALSE based ONLY on what is
explicitly described in the work history above. Do not infer from skills.
Do not guess. If unsure, answer FALSE.

Signals:
- built_production_ai:        Did they ship ML/AI systems to real users in production?
- vector_search_built:        Did they build semantic or vector search in a live product?
- ranking_system_built:       Did they build a ranking or recommendation pipeline?
- evaluation_framework_built: Did they build offline/online eval (NDCG, A/B tests)?
- embeddings_in_production:   Did they use text embeddings in a live production system?
- rag_or_retrieval_built:     Did they build RAG, information retrieval, or hybrid search?

Also answer:
- company_type: "product" | "consulting" | "mixed"
  (consulting = TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, or similar)
- red_flags: list any red_flag_patterns from the JD that apply to this candidate
- confidence: 0.0 to 1.0 — how confident are you given the available text?

Return valid JSON only. No other text.
```

**Result saved to `artifacts/work_evidence.json`:**
```json
{
  "CAND_0000031": {
    "built_production_ai":        true,
    "vector_search_built":        true,
    "ranking_system_built":       true,
    "evaluation_framework_built": true,
    "embeddings_in_production":   true,
    "rag_or_retrieval_built":     true,
    "company_type":               "product",
    "red_flags":                  [],
    "confidence":                 0.95
  },
  "CAND_0000001": {
    "built_production_ai":        false,
    "vector_search_built":        false,
    "ranking_system_built":       false,
    "evaluation_framework_built": false,
    "embeddings_in_production":   false,
    "rag_or_retrieval_built":     false,
    "company_type":               "mixed",
    "red_flags":                  ["career shows only data engineering, no AI/ML systems"],
    "confidence":                 0.88
  }
}
```

**Cost:** 2,000 API calls × $0.003 (GPT-4o-mini) ≈ **$6**

### Step 3: Re-score All 2,000 → Take Top 200

With work evidence now available, compute the full multi-signal score for
all 2,000 candidates. This is the same formula Phase 3 will use.

```python
features_df   = pd.read_parquet("artifacts/features.parquet")
jd_signals    = json.load(open("artifacts/jd_signals.json"))
work_evidence = json.load(open("artifacts/work_evidence.json"))

scores = {}
for cand_id in top_2000_ids:
    feat      = features_df.loc[cand_id]
    evidence  = work_evidence.get(cand_id, {})
    sem_sim   = top_2000_sims[cand_id]

    work_score = compute_work_evidence_score(evidence, jd_signals)

    scores[cand_id] = (
        0.25 * work_score
      + 0.20 * sem_sim
      + 0.20 * feat["career_quality_score"]
      + 0.15 * feat["skills_fit_score"]
      + 0.12 * feat["availability_score"]
      + 0.08 * feat["location_score"]
    ) * feat["honeypot_multiplier"]

# Sort descending → take top 200 (buffer of 100 over the required 100)
top_200 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:200]
```

**Why top 200 instead of top 100?**
Phase 3 will re-run the same FAISS + scoring. Due to floating-point precision
differences in different environments, a handful of borderline candidates might
swap positions near rank 95–105. Generating reasoning for 200 ensures every
candidate Phase 3 selects for the top 100 already has reasoning ready.

### Step 4: Reasoning Generation (LLM on top 200 only)

We generate reasoning only after re-ranking because:
- We only want reasoning for candidates who actually made it into the top 200
- The rank number in the reasoning prompt reflects their actual relative position

```python
candidate_lookup = build_candidate_lookup(candidates)  # candidate_id → full JSON

reasoning = {}
for rank_in_200, (cand_id, score) in enumerate(top_200, start=1):
    cand     = candidate_lookup[cand_id]
    evidence = work_evidence[cand_id]
    feat     = features_df.loc[cand_id]

    prompt = f"""
Write a 1-2 sentence recruiter reasoning for this candidate's position in a
Senior AI Engineer shortlist. Be specific. Reference facts from their profile.
Acknowledge concerns honestly. Do not hallucinate facts not present below.

Approximate rank: #{rank_in_200} of 200 shortlisted
Score: {score:.3f}

Candidate profile:
- Current role:  {cand['profile']['current_title']} at {cand['profile']['current_company']}
- Career path:   {' → '.join(j['company'] for j in cand['career_history'])}
- Company types: {evidence.get('company_type', 'unknown')}
- Work signals:  {[k for k, v in evidence.items() if v is True]}
- Top skills:    {get_top_relevant_skills(cand, jd_skill_weights, n=3)}
- Location:      {cand['profile']['location']}. Relocate: {feat['willing_to_relocate']}
- Last active:   {feat['days_since_last_active']} days ago
- Response rate: {cand['redrob_signals']['recruiter_response_rate']}
- Red flags:     {evidence.get('red_flags', [])}
"""
    reasoning[cand_id] = call_llm(prompt)

json.dump(reasoning, open("artifacts/reasoning.json", "w"), indent=2)
```

---

## 7. Phase 3 — Online Ranking

**Script:** `rank.py`
**Runs:** On demand, every time the judges (or you) want to produce the CSV.
**Must finish:** ≤ 5 minutes, CPU only, no internet
**Inputs:** All 6 artifacts from Phases 0–2
**Output:** `submission.csv`

### Why Phase 3 re-runs FAISS when Phase 2 already did it

The hackathon judges run ONLY `rank.py`. They do not run Phase 0, 1, or 2.
So `rank.py` must be fully self-contained — it must produce the final CSV
from artifacts alone.

Phase 3 re-runs FAISS search + scoring. Both steps are fast (milliseconds —
no LLM calls). Because the FAISS index, JD query, features, and work_evidence
are all deterministic, Phase 3 always produces the same top 100 that Phase 2
identified. Phase 2's top 200 ⊇ Phase 3's top 100, so every candidate in the
final CSV has reasoning pre-generated.

```
Phase 2: FAISS(k=2000) → LLM enrichment → re-score → top 200 → reasoning
Phase 3: FAISS(k=2000) → re-score (no LLM) → top 100 → attach reasoning → CSV

Phase 3 top 100  ⊂  Phase 2 top 200       (always — same deterministic formula)
```

### Complete `rank.py` code

```python
import json, faiss, pandas as pd
from sentence_transformers import SentenceTransformer

# ── Step 1: Load all artifacts ────────────────────────────────────────
candidate_ids = json.load(open("artifacts/candidate_ids.json"))
features_df   = pd.read_parquet("artifacts/features.parquet")
# features_df is indexed by candidate_id so .loc[cand_id] works directly

index         = faiss.read_index("artifacts/candidates.faiss")
jd_signals    = json.load(open("artifacts/jd_signals.json"))
work_evidence = json.load(open("artifacts/work_evidence.json"))
reasoning     = json.load(open("artifacts/reasoning.json"))

# ── Step 2: Embed the focused JD query (~0.1 seconds) ────────────────
model    = SentenceTransformer("all-MiniLM-L6-v2")
jd_query = open("artifacts/jd_focused_query.txt").read()   # Phase 0 artifact
jd_vec   = model.encode([jd_query], convert_to_numpy=True)
faiss.normalize_L2(jd_vec)

# ── Step 3: FAISS search → top 2,000 (~50 ms) ────────────────────────
distances, indices = index.search(jd_vec.astype("float32"), k=2000)
top_2000_ids  = [candidate_ids[i] for i in indices[0]]
sem_sim_map   = {candidate_ids[i]: float(distances[0][j])
                 for j, i in enumerate(indices[0])}

# ── Step 4: Score all 2,000 (pure arithmetic, no LLM) ────────────────
scores = {}
for cand_id in top_2000_ids:
    feat      = features_df.loc[cand_id]
    evidence  = work_evidence.get(cand_id, {})   # {} → work_score = 0.0 (safe fallback)
    sem_sim   = sem_sim_map[cand_id]

    work_score = compute_work_evidence_score(evidence, jd_signals)

    scores[cand_id] = (
        0.25 * work_score
      + 0.20 * sem_sim
      + 0.20 * float(feat["career_quality_score"])
      + 0.15 * float(feat["skills_fit_score"])
      + 0.12 * float(feat["availability_score"])
      + 0.08 * float(feat["location_score"])
    ) * float(feat["honeypot_multiplier"])

# ── Step 5: Sort → top 100 → build CSV ───────────────────────────────
top_100 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]

rows = []
for rank, (cand_id, score) in enumerate(top_100, start=1):
    rows.append({
        "candidate_id": cand_id,
        "rank":         rank,
        "score":        round(score, 4),
        "reasoning":    reasoning.get(cand_id, "")
    })

pd.DataFrame(rows).to_csv("submission.csv", index=False)
print("Done. submission.csv written.")
# ── Total runtime: < 2 minutes ────────────────────────────────────────
```

### The `compute_work_evidence_score` helper

```python
def compute_work_evidence_score(evidence: dict, jd_signals: dict) -> float:
    """
    Converts raw LLM-extracted boolean signals into a 0-1 score.
    Uses weights from jd_signals.json (generated dynamically from the JD in Phase 0).
    Works for any JD — no hardcoded signal names.
    """
    if not evidence:
        return 0.0   # candidate had no LLM analysis (should not happen for top 2K)

    score = 0.0
    for signal in jd_signals["signals"]:
        key    = signal["key"]
        weight = signal["weight"]
        value  = evidence.get(key, False)

        if isinstance(value, bool):
            score += weight * (1.0 if value else 0.0)
        # Non-boolean signals (like "scale": "large") can be added here if needed

    # Penalty for consulting-only background (multiplicative, not additive)
    # Multiplicative is cleaner than subtracting a fixed amount
    if evidence.get("company_type") == "consulting":
        score *= 0.6   # 40% reduction — still ranked, just penalised

    return min(score, 1.0)
```

---

## 8. The Scoring Formula — Fully Explained

### Final formula

```
final_score = (
    0.25 × work_evidence_score       ← what they actually built (LLM-extracted)
  + 0.20 × semantic_similarity       ← career narrative vs JD meaning (FAISS)
  + 0.20 × career_quality_score      ← product co. history; role type; experience band
  + 0.15 × skills_fit_score          ← skill overlap, trust-adjusted by assessments
  + 0.12 × availability_score        ← behavioral signals: can we actually hire them?
  + 0.08 × location_score            ← India / willing to relocate to Pune/Noida
) × honeypot_multiplier              ← 0.0 eliminates impossible profiles entirely
```

### Why these weights?

| Component | Weight | Justification from JD |
|---|---|---|
| work_evidence | 0.25 | JD says "keyword filters can't see what actually matters" — work done > skills listed |
| semantic_similarity | 0.20 | Career narrative captures domain alignment holistically |
| career_quality | 0.20 | JD explicitly disqualifies consulting-only + wrong-experience-band candidates |
| skills_fit | 0.15 | Skills matter but are low-trust — weighted below work evidence |
| availability | 0.12 | JD says "a perfect candidate who hasn't logged in for 6 months is not actually available" |
| location | 0.08 | JD says Pune/Noida with India preference, but there is some flexibility |

### Where each score comes from

| Score | Source | When Computed |
|---|---|---|
| `work_evidence_score` | LLM reads career descriptions → JSON booleans | Phase 2 Step 2 |
| `semantic_similarity` | FAISS cosine similarity (work narrative vs JD query) | Phase 3 Step 3 |
| `career_quality_score` | Rule-based (consulting ratio + title + exp band) | Phase 1 Step 2 |
| `skills_fit_score` | Weighted skill match (JD weights + platform assessment) | Phase 1 Step 3 |
| `availability_score` | Behavioral signals composite | Phase 1 Step 2 |
| `location_score` | country + willing_to_relocate lookup | Phase 1 Step 2 |
| `honeypot_multiplier` | Rule-based 5-check detection | Phase 1 Step 1 |

---

## 9. Tech Terms Glossary

### Embedding
Converting text into a fixed-length list of numbers (a vector) so computers can
measure meaning similarity. "Built ranking systems" and "developed recommendation
models" produce very similar vectors. "Managed Excel spreadsheets" produces a
very different vector.

### Vector
A list of numbers representing a point in multi-dimensional space.
`[0.38, -0.21, 0.64, 0.47, ...]` — in our case, 384 numbers per candidate.

### Cosine Similarity
A way to measure how "close" two vectors are. Range: -1.0 to 1.0.
1.0 = identical meaning. 0.0 = unrelated. -1.0 = opposite.
Two career narratives about search and ranking systems will score ~0.8+.
A marketing career vs an ML engineering JD will score ~0.2.

### Sentence Transformers
A Python library that wraps pre-trained models to convert sentences/paragraphs
into embeddings. We use `all-MiniLM-L6-v2`: 384-dim output, runs on CPU,
no API key needed, good English quality.

### FAISS
Facebook AI Similarity Search. A library that stores millions of embedding vectors
on disk/RAM and finds the top-K most similar to a query vector in milliseconds.
Think of it as a "similarity search engine" — like Google, but for meaning instead
of keywords. At rank time, FAISS returns the 2,000 most similar candidates to the JD
in about 50ms.

### NDCG (Normalized Discounted Cumulative Gain)
The metric used to score this hackathon. It measures how well you ranked candidates
by rewarding putting the best candidates at the top. A perfect ranking = NDCG of 1.0.
The competition weights NDCG@10 at 50% — your top 10 picks matter most.

### Hybrid Scoring
Combining multiple signals (semantic similarity + work evidence + skills + behavioral)
with weights into a final score. More robust than any single signal alone.

### Re-ranking
A two-stage retrieval pattern. Stage 1: fast approximate search (FAISS, top 2,000).
Stage 2: slower, more careful scoring re-ranks those 2,000 to find the true top 100.
Standard in all production search systems.

### Honeypot
A fake/planted candidate with a subtly impossible profile — to test whether your
system reads profiles or just keyword-matches. >10% honeypots in your top 100 = disqualified.

### Parquet
A columnar file format for storing tabular data. Like CSV but compressed and
much faster to read/write for large datasets. We use it for the 100K × 15 feature matrix.

### NDJSON
Newline-Delimited JSON. Each line is a complete, valid JSON object.
The candidates.json file is 100,000 lines long, each line is one candidate.

---

## 10. Three Candidate Walkthroughs

Three real candidates from the dataset, chosen to represent the three most
important cases our system must handle correctly:

| Candidate | Type | Expected outcome |
|---|---|---|
| Ela Singh (CAND_0000031) | Strong, genuine fit | Top 5 rank |
| Anil Bose (CAND_0000004) | Honeypot trap | Score = 0.000, not ranked |
| Ira Vora (CAND_0000001) | Keyword stuffer | Outside top 100 |

---

### Candidate A: Ela Singh — CAND_0000031

**Raw profile snapshot:**
```
Title:    Recommendation Systems Engineer
Company:  Swiggy (Food Delivery, 5001–10000 employees)
Location: Hyderabad, Telangana, India
YoE:      6.0 years
```

**Career history:**
```
1. Swiggy [14 months, current]
   "Trained and shipped multiple ranking models for discovery feed using XGBoost
   and LightGBM. Designed features: content metadata, user behavior, engagement
   history. Owned offline-online correlation analysis for A/B outcomes."

2. Mad Street Den — Search Engineer [16 months] (AI/ML product company)
   "Trained and shipped ranking models for discovery feed. Designed features.
   Owned offline-online correlation analysis."

3. Uber — NLP Engineer [27 months]
   "Trained and shipped ranking models. Designed features across three families.
   Offline-online correlation analysis."

4. Zomato — Applied ML Engineer [13 months]
   "Owned ranking layer for e-commerce search. Evolved hand-tuned scoring to
   learning-to-rank model. Designed relevance labeling pipeline (click-through +
   human judgments). Improved revenue-per-search by 12%."
```

**Skills (selected):**
```
FAISS:               advanced, 35 months, 19 endorsements, assessment 68.4/100
Pinecone:            expert,   88 months, 34 endorsements, assessment 53.6/100
Embeddings:          expert,   60 months, 48 endorsements  (no assessment)
Sentence Transformers: expert, 69 months, 16 endorsements  (no assessment)
Information Retrieval: expert, 84 months,  2 endorsements  (no assessment)
MLflow:              advanced, 21 months, 59 endorsements, assessment 75.1/100
Speech Recognition:  intermediate, 24 months               (no assessment)
```

**Behavioral signals:**
```
last_active_date:           2026-05-24  (22 days ago)
open_to_work_flag:          true
recruiter_response_rate:    0.91
avg_response_time_hours:    76.1
notice_period_days:         60
interview_completion_rate:  0.60
offer_acceptance_rate:      0.38
github_activity_score:      32.6
willing_to_relocate:        true
```

---

#### PHASE 0 effect on Ela's processing

Phase 0 already ran before Ela's profile is touched. It produced:
- `jd_signals.json` with 6 signal keys and weights
- `jd_skill_weights.json` listing FAISS=1.0, Pinecone=1.0, Embeddings=1.0 etc.
- `jd_focused_query.txt` describing the AI engineer profile in plain prose

These artifacts tell Phase 1 exactly how to score Ela's skills and which signals
to extract later.

---

#### PHASE 1 WALKTHROUGH — Ela Singh

**Step 1: Honeypot Detection**
```
Check 1 — Advanced/expert skills with 0 months?
  FAISS: advanced, 35 months ✓
  Pinecone: expert, 88 months ✓
  All skills have reasonable duration.
  → 0 flags

Check 2 — Career months vs stated YoE?
  Career total: 14 + 16 + 27 + 13 = 70 months = 5.83 years
  Stated YoE: 6.0 years   |   difference: 0.17 years
  → 0 flags (well within 4-year threshold)

Check 3 — Overlapping full-time jobs?
  Zomato ended 2021-07-22, Uber started 2021-07-22  (same day = no overlap)
  Uber ended 2023-10-10, Mad Street Den started 2023-10-10  (same day = no overlap)
  → 0 flags

Check 4 — Impossible education sequence?
  M.Tech at SRM: 2002–2006
  First job (Zomato) started 2020. 14-year gap unusual but possible.
  No chronological contradiction.
  → 0 flags

Check 5 — Title vs description mismatch?
  "Recommendation Systems Engineer" + "Trained and shipped ranking models" ✓
  → 0 flags

RESULT: 0 honeypot signals → is_honeypot = False → honeypot_multiplier = 1.0
```

**Step 2: Structured Feature Extraction**
```
years_of_experience:        6.0

Location:
  country = "India" → is_india_based = True
  willing_to_relocate = True
  location_score = 1.0   (India + willing = maximum)

Career quality:
  Swiggy:         Food Delivery  → product ✓
  Mad Street Den: AI/ML          → product ✓
  Uber:           Transportation → product ✓
  Zomato:         Food Delivery  → product ✓
  consulting_ratio = 0/70 = 0.00
  has_engineering_title = True  ("Engineer", "ML Engineer")
  yoe = 6.0 → exp_factor = 1.0  (sweet spot 5–9)
  career_quality_score = 1.0 × 1.05 × 1.0 = 1.0  (capped at 1.0)

Education:
  M.Tech, SRM University, tier_2, Computer Engineering
  highest_edu_tier = 2 → education_score = 0.70
  edu_relevant_field = True

Availability:
  days_inactive = 22  → recency = 1.0 − 22/180 = 0.878
  open_to_work = True → 1.0
  response_rate = 0.91
  notice_score = 1.0 − 60/180 = 0.667
  interview_completion = 0.60

  availability_score = 0.30×0.878 + 0.25×1.0 + 0.20×0.91 + 0.15×0.667 + 0.10×0.60
                     = 0.263 + 0.250 + 0.182 + 0.100 + 0.060
                     = 0.855
```

**Step 3: Skill Score**
```
FAISS (relevance = 1.0):
  proficiency = advanced → 0.9
  platform    = 68.4/100 → 0.684
  blended     = 0.40×0.9 + 0.60×0.684 = 0.360 + 0.410 = 0.770
  duration    = 0.5 + 0.5×(35/60) = 0.792
  contribution = 1.0 × 0.770 × 0.792 = 0.610

Pinecone (relevance = 1.0):
  proficiency = expert → 1.0
  platform    = 53.6/100 → 0.536
  blended     = 0.40×1.0 + 0.60×0.536 = 0.400 + 0.322 = 0.722
  duration    = 0.5 + 0.5×(60/60) = 1.0  (capped at 60)
  contribution = 1.0 × 0.722 × 1.0 = 0.722

Embeddings (relevance = 1.0):
  proficiency = expert → 1.0  (no platform assessment available)
  duration    = 0.5 + 0.5×(60/60) = 1.0
  contribution = 1.0 × 1.0 × 1.0 = 1.000

Sentence Transformers (relevance = 1.0):
  proficiency = expert → 1.0
  duration    = 0.5 + 0.5×(60/60) = 1.0  (capped)
  contribution = 1.000

Information Retrieval (relevance = 0.9):
  proficiency = expert → 1.0
  duration    = 1.0
  contribution = 0.9 × 1.0 × 1.0 = 0.900

Speech Recognition (relevance = -0.2):  ← negative for this JD
  proficiency = intermediate → 0.6
  duration    = 0.5 + 0.5×(24/60) = 0.700
  contribution = -0.2 × 0.6 × 0.700 = -0.084

max_possible = 1.0+1.0+1.0+1.0+0.9+0.2+... ≈ 6.2 (sum of |relevance| for all matched)
total_weighted = 0.610+0.722+1.0+1.0+0.9+(−0.084)+... ≈ 4.9

skills_fit_score = 4.9 / 6.2 ≈ 0.79  (strong, but penalised slightly by speech skill)
```

**Step 4: Work Narrative Built and Embedded**
```
"Role: Recommendation Systems Engineer at Swiggy (product, Food Delivery, 14 months)
Work done: Trained and shipped multiple ranking models for our product's discovery
feed using XGBoost and LightGBM. Designed features across three families: content
metadata, user behavior signals, and item engagement history. Owned the offline-online
correlation analysis that determined which offline metrics actually predicted A/B test
outcomes. Worked closely with PMs to define the optimization target.

---

Role: Search Engineer at Mad Street Den (product, AI/ML, 16 months)
Work done: [same pattern — ranking models, feature design, A/B analysis]

---

Role: NLP Engineer at Uber (product, Transportation, 27 months)
Work done: [same pattern at larger scale]

---

Role: Applied ML Engineer at Zomato (product, Food Delivery, 13 months)
Work done: Owned ranking layer for e-commerce search product. Evolved from hand-tuned
scoring function to learning-to-rank model. Designed relevance labeling pipeline.
Final model improved revenue-per-search by 12%."
```

This text gets embedded → vector `[0.38, -0.21, 0.64, 0.47, ...]` stored in FAISS.

---

#### PHASE 2 WALKTHROUGH — Ela Singh

Ela almost certainly enters the FAISS top 2,000 (likely top 20). LLM receives
her career history and returns:

**Work Evidence (GPT-4o-mini output):**
```json
{
  "built_production_ai":        true,
  "vector_search_built":        true,
  "ranking_system_built":       true,
  "evaluation_framework_built": true,
  "embeddings_in_production":   true,
  "rag_or_retrieval_built":     true,
  "company_type":               "product",
  "red_flags":                  [],
  "confidence":                 0.95
}
```

**work_evidence_score computation:**
```
built_production_ai        true  × weight 0.25 = 0.250
vector_search_built        true  × weight 0.20 = 0.200
ranking_system_built       true  × weight 0.20 = 0.200
evaluation_framework_built true  × weight 0.15 = 0.150
embeddings_in_production   true  × weight 0.10 = 0.100
rag_or_retrieval_built     true  × weight 0.10 = 0.100

Raw sum = 1.000
company_type = "product" → no penalty
work_evidence_score = 1.000  (perfect)
```

**Re-score in Phase 2 (same formula as Phase 3):**
```
work_evidence_score:   1.000  × 0.25 = 0.250
semantic_similarity:   0.870  × 0.20 = 0.174
career_quality_score:  1.000  × 0.20 = 0.200
skills_fit_score:      0.790  × 0.15 = 0.119
availability_score:    0.855  × 0.12 = 0.103
location_score:        1.000  × 0.08 = 0.080
                                       ─────
subtotal:                              0.926
× honeypot_multiplier:                 × 1.0
                                       ─────
Phase 2 score:                         0.926
```

Ela ranks ~#1 in the Phase 2 top 200. Reasoning is generated for her.

**Reasoning (GPT-4o-mini output):**
```
"Ela Singh has 6 years building ranking and retrieval systems exclusively at
product companies (Swiggy, Uber, Zomato, Mad Street Den), with confirmed
production deployment of learning-to-rank models, embedding-based retrieval,
and A/B evaluation frameworks — precisely the profile this JD describes.
Concern: interview_completion_rate (0.60) and offer_acceptance_rate (0.38)
indicate she evaluates offers selectively; recommend early engagement."
```

---

#### PHASE 3 WALKTHROUGH — Ela Singh

Phase 3 re-runs FAISS → Ela appears in top 2,000. Same scoring formula
produces the same score.

**Final score:**
```
work_evidence_score:   1.000  × 0.25 = 0.250
semantic_similarity:   0.870  × 0.20 = 0.174
career_quality_score:  1.000  × 0.20 = 0.200
skills_fit_score:      0.790  × 0.15 = 0.119
availability_score:    0.855  × 0.12 = 0.103
location_score:        1.000  × 0.08 = 0.080
                                       ─────
subtotal:                              0.926
× honeypot_multiplier:                 × 1.0
                                       ─────
FINAL SCORE: 0.926

Reasoning loaded from reasoning.json (dict lookup, ~0 ms).
```

**CSV row written:**
```csv
CAND_0000031,1,0.9260,"Ela Singh has 6 years building ranking and retrieval systems..."
```

---

### Candidate B: Anil Bose — CAND_0000004

**Raw profile snapshot:**
```
Title:    Marketing Manager
Company:  Dunder Mifflin (Paper Products, 201-500 employees)
Location: Sydney, Australia
YoE:      3.8 years
```

**Career history:**
```
1. Dunder Mifflin — "Marketing Manager" [14 months]
   Description: "Mechanical engineering design role... CAD (SolidWorks, Creo),
   FEA (ANSYS)..."
   ← MISMATCH: Title = Marketing Manager, work = mechanical engineering design

2. Infosys — "Operations Manager" [20 months]
   Description: "Content writing and SEO strategy for a tech publication..."
   ← MISMATCH: Title = Operations Manager, work = content writing

3. Globex Inc — "Business Analyst" [10 months]
   Description: "Operations management role at a logistics company..."
   ← Partial mismatch
```

**Education:**
```
Ph.D  at Lovely Professional University, Electronics, 2013–2016
B.Tech at Local Engineering College, Machine Learning, 2015–2019

← IMPOSSIBLE: PhD completed (2016) before B.Tech ended (2019)
```

**Behavioral signals:**
```
profile_completeness_score: 28.5   (very low — profile barely filled)
last_active_date:           2026-03-25 (82 days ago)
open_to_work_flag:          false
github_activity_score:      -1     (no GitHub linked)
expected_salary:            4.6–6.7 LPA   (Senior AI Engineers earn 40–80 LPA)
```

---

#### PHASE 1 WALKTHROUGH — Anil Bose

**Step 1: Honeypot Detection**
```
Check 1 — Advanced/expert skills with 0 months?
  No advanced/expert skills with 0 duration in skill list.
  → 0 flags

Check 2 — Career months vs stated YoE?
  Career total: 14 + 20 + 10 = 44 months = 3.67 years
  Stated YoE: 3.8 years   |   difference: 0.13 years
  → 0 flags

Check 3 — Overlapping full-time jobs?
  Globex:       2022-08-02 → 2023-05-29
  Infosys:      2023-07-28 → 2025-03-19  (gap of 2 months between, no overlap)
  Dunder Mifflin: 2025-04-02 → now
  → 0 flags

Check 4 — Impossible education sequence?
  PhD at LPU:   start 2013, END 2016
  B.Tech at LEC: start 2015, END 2019

  The PhD was awarded in 2016. The B.Tech was NOT completed until 2019.
  He received a PhD BEFORE completing his Bachelor's degree.
  In any standard educational system this is impossible.
  → 1 flag ✓

Check 5 — Title vs description mismatch?
  Job 1: title "Marketing Manager", description = "CAD, SolidWorks, FEA" → MISMATCH ✓
  Job 2: title "Operations Manager", description = "content writing, SEO" → MISMATCH ✓
  → 1 flag ✓

RESULT: 2 honeypot signals triggered
is_honeypot = True → honeypot_multiplier = 0.0

ELIMINATED. Score will be 0.000 regardless of any other signals.
Processing continues for illustration only.
```

**Step 2 (illustration only — has no effect on outcome):**
```
is_india_based = False (Sydney, Australia)
willing_to_relocate = True
location_score = 0.3

consulting_ratio = 20/(14+20+10) = 0.45  (Infosys is consulting)
career_quality_score ≈ 0.31

yoe = 3.8 → below JD's 5-year preference → exp_factor = 0.75
```

**Phase 3 outcome:**
```
final_score = [anything] × honeypot_multiplier
            = [anything] × 0.0
            = 0.000

FINAL RANK: Not in top 100. Not in submission CSV.
```

---

### Candidate C: Ira Vora — CAND_0000001

**Raw profile snapshot:**
```
Title:    Backend Engineer
Company:  Mindtree (IT Services, 10001+ employees)
Location: Toronto, Canada
YoE:      6.9 years
```

**Career history:**
```
1. Mindtree (IT Services) — Backend Engineer [27 months]
   "Implemented streaming data pipelines on Kafka and Spark Streaming for
   a real-time user-activity processing platform. Designed schema-registry
   integration, watermark/state management, deduplication logic for late-
   arriving events. Most of my career has been data engineering, with some
   adjacent ML exposure."

2. Dunder Mifflin (Paper Products, 201-500) — Analytics Engineer [55 months]
   "Built data pipelines on Airflow processing ~500GB daily data. Spark
   (PySpark) for batch processing, dbt for Snowflake warehouse. Owned on-call
   for data quality. Wrote checks for schema drift and volume changes."
```

**Skills listed — note the suspicious gap between claims and assessments:**
```
NLP:              advanced, 26 months, 37 endorsements → assessment 38.8/100 ← LOW
Fine-tuning LLMs: advanced, 36 months, 21 endorsements → assessment 41.6/100 ← LOW
Milvus:           advanced, 35 months, 40 endorsements → no assessment
Speech Recog.:    advanced, 33 months, 52 endorsements → assessment 53.7/100
Image Classif.:   advanced, 40 months,  7 endorsements → assessment 64.8/100
TTS:              advanced, 60 months, 56 endorsements → no assessment
LoRA:             intermediate, 28 months, 0 endorsements
```

**Pattern:** Claims advanced NLP and LLM fine-tuning, but platform tests reveal
38.8 and 41.6 out of 100. These are below-average scores for claimed proficiency.
Career history mentions none of these technologies — all work is data pipelines.

**Behavioral signals:**
```
last_active_date:          2026-05-20 (26 days ago)
open_to_work_flag:         true
recruiter_response_rate:   0.34   (low — responds to only 1 in 3)
avg_response_time_hours:   177.8  (7+ days to reply)
notice_period_days:        60
willing_to_relocate:       false  ← in Canada, won't relocate
github_activity_score:     9.2    (low)
```

---

#### PHASE 1 WALKTHROUGH — Ira Vora

**Step 1: Honeypot Detection**
```
Check 1 — Advanced/expert skills with 0 months?
  LoRA: intermediate (not advanced) with 0 endorsements → doesn't trigger
  All "advanced" skills have duration > 0.
  → 0 flags

Check 2 — Career months vs stated YoE?
  Career total: 27 + 55 = 82 months = 6.83 years
  Stated YoE: 6.9 years   |   difference: 0.07 years
  → 0 flags

Check 3 — Overlapping jobs?
  Dunder Mifflin ended 2024-01-08, Mindtree started 2024-03-08 (2-month gap)
  → 0 flags

Check 4 — Impossible education?
  B.E. at LPU: 2017–2020
  First job started 2019-07-03 → working before graduation
  This could be an internship or part-time — borderline. Threshold not met alone.
  → borderline, not flagged

Check 5 — Title vs description mismatch?
  "Backend Engineer" + Kafka/Spark/Airflow work → consistent ✓
  → 0 flags

RESULT: 0 honeypot signals
is_honeypot = False → honeypot_multiplier = 1.0
```

Ira is not a honeypot — she is a real person whose skills section
does not reflect her actual work.

**Step 2: Structured Feature Extraction**
```
years_of_experience: 6.9

Location:
  country = "Canada" → is_india_based = False
  willing_to_relocate = False
  location_score = 0.0   ← ZERO. Outside India + not willing to relocate.
                            This JD requires India/Pune/Noida.

Career quality:
  Mindtree (IT Services 10001+) → consulting firm ✓
  Dunder Mifflin (Paper Products 201-500) → not in consulting list → treated as other
  consulting_months = 27,  total_months = 82
  consulting_ratio = 27/82 = 0.329
  has_engineering_title = True  ("Backend Engineer", "Analytics Engineer")
  yoe = 6.9 → exp_factor = 1.0  (within 5-9 sweet spot)
  base = 1.0 − 0.329 = 0.671
  career_quality_score = 0.671 × 1.05 × 1.0 = 0.705

Availability:
  days_inactive = 26 → recency = 1.0 − 26/180 = 0.856
  open_to_work = True → 1.0
  response_rate = 0.34  ← low
  notice_score = 1.0 − 60/180 = 0.667
  interview_completion = 0.71

  availability_score = 0.30×0.856 + 0.25×1.0 + 0.20×0.34 + 0.15×0.667 + 0.10×0.71
                     = 0.257 + 0.250 + 0.068 + 0.100 + 0.071
                     = 0.746
```

**Step 3: Skill Score — The Disconnect Is Revealed**
```
NLP (relevance = 0.8):
  self-claimed: advanced → 0.9
  platform:     38.8/100 → 0.388
  blended:      0.40×0.9 + 0.60×0.388 = 0.360 + 0.233 = 0.593  ← dragged way down
  duration:     0.5 + 0.5×(26/60) = 0.717
  contribution: 0.8 × 0.593 × 0.717 = 0.340

Fine-tuning LLMs (relevance = 0.6):
  self-claimed: advanced → 0.9
  platform:     41.6/100 → 0.416
  blended:      0.40×0.9 + 0.60×0.416 = 0.360 + 0.250 = 0.610
  duration:     0.5 + 0.5×(36/60) = 0.800
  contribution: 0.6 × 0.610 × 0.800 = 0.293

Milvus (relevance = 1.0):  ← the only genuine AI-relevant skill
  self-claimed: advanced → 0.9  (no platform assessment)
  duration:     0.5 + 0.5×(35/60) = 0.792
  contribution: 1.0 × 0.9 × 0.792 = 0.713
  (But: career history never mentions Milvus — suspicious)

Speech Recognition (relevance = -0.2):  ← negative for this JD
  contribution ≈ -0.084

TTS (relevance = -0.1):  ← irrelevant/slightly negative
  contribution ≈ -0.050

skills_fit_score (normalized) ≈ 0.38   ← mediocre, platform tests reveal the gap
```

**Step 4: Work Narrative**
```
"Role: Backend Engineer at Mindtree (consulting, IT Services, 27 months)
Work done: Implemented streaming data pipelines on Kafka and Spark Streaming.
Designed schema-registry integration, watermark/state management, deduplication
for late-arriving events.

---

Role: Analytics Engineer at Dunder Mifflin (other, Paper Products, 55 months)
Work done: Built Airflow pipelines processing ~500GB daily. Spark/PySpark for
batch processing, dbt for Snowflake warehouse. Data quality checks for schema
drift and volume anomalies."
```

This embeds as a vector very different from the JD's focused query
(which is about ranking systems, embeddings, retrieval).

**Semantic similarity against JD:** ~0.28 (data engineering ≠ ML ranking)

---

#### PHASE 2 WALKTHROUGH — Ira Vora

Ira may or may not make it into the FAISS top 2,000. Her work narrative
is about Kafka/Spark/Airflow — semantically distant from the JD (which
is about ranking, embeddings, retrieval). If she does make it in, she
receives LLM work evidence extraction:

**Work Evidence (GPT-4o-mini output):**
```json
{
  "built_production_ai":        false,
  "vector_search_built":        false,
  "ranking_system_built":       false,
  "evaluation_framework_built": false,
  "embeddings_in_production":   false,
  "rag_or_retrieval_built":     false,
  "company_type":               "mixed",
  "red_flags":                  [
    "career history shows data engineering only — no AI/ML systems built or deployed",
    "platform assessment scores (NLP: 38.8, LLMs: 41.6) contradict advanced proficiency claims",
    "33% career at IT consulting (Mindtree)"
  ],
  "confidence":                 0.91
}
```

**work_evidence_score:** all signals false → 0.000

---

#### PHASE 3 WALKTHROUGH — Ira Vora

**Final score:**
```
work_evidence_score:   0.000  × 0.25 = 0.000   ← no AI work built
semantic_similarity:   0.280  × 0.20 = 0.056   ← data eng ≠ ML ranking
career_quality_score:  0.705  × 0.20 = 0.141
skills_fit_score:      0.380  × 0.15 = 0.057   ← platform tests reveal gap
availability_score:    0.746  × 0.12 = 0.090   ← dragged down by 0.34 response rate
location_score:        0.000  × 0.08 = 0.000   ← Canada + not relocating
                                        ─────
subtotal:                               0.344
× honeypot_multiplier:                  × 1.0
                                        ─────
FINAL SCORE: 0.344

FINAL RANK: Outside top 100 (likely ~800–2000 range)
No CSV row written.
```

---

## 11. Final Score Comparison

| | Ela Singh | Anil Bose | Ira Vora |
|---|---|---|---|
| **Candidate ID** | CAND_0000031 | CAND_0000004 | CAND_0000001 |
| **Archetype** | Genuine fit | Honeypot | Keyword stuffer |
| **Honeypot detected?** | No → 1.0 | Yes → 0.0 | No → 1.0 |
| **work_evidence_score** | 1.000 | eliminated | 0.000 |
| **semantic_similarity** | 0.870 | eliminated | 0.280 |
| **career_quality_score** | 1.000 | eliminated | 0.705 |
| **skills_fit_score** | 0.790 | eliminated | 0.380 |
| **availability_score** | 0.855 | eliminated | 0.746 |
| **location_score** | 1.000 | eliminated | 0.000 |
| **FINAL SCORE** | **0.926** | **0.000** | **0.344** |
| **Expected Rank** | **~#1** | **Not ranked** | **~#800+** |

The system behaves exactly as a great recruiter would:
- Ela ranks at the top because her *work* matches the role, she's available, and she's in India
- Anil is eliminated because his profile is internally impossible (education timeline fraud)
- Ira scores mid-range despite AI-sounding skills because her *work history* shows
  data pipelines only, platform tests expose inflated proficiency, and she's not in India

---

## 12. Artifact Reference

Every file that flows between phases:

| File | Created in | Used in | Contents |
|---|---|---|---|
| `jd_signals.json` | Phase 0 | Phase 2, Phase 3 | Work evidence signal schema: keys, questions, weights |
| `jd_skill_weights.json` | Phase 0 | Phase 1 | Skill name → relevance weight mapping |
| `jd_focused_query.txt` | Phase 0 | Phase 1, Phase 3 | Clean JD prose for embedding |
| `features.parquet` | Phase 1 | Phase 2, Phase 3 | 100K rows × structured features, indexed by candidate_id |
| `candidates.faiss` | Phase 1 | Phase 2, Phase 3 | FAISS index of 100K work narrative vectors |
| `candidate_ids.json` | Phase 1 | Phase 2, Phase 3 | List mapping FAISS position → candidate_id |
| `work_evidence.json` | Phase 2 | Phase 3 | LLM-extracted signals for top 2,000 candidates |
| `reasoning.json` | Phase 2 | Phase 3 | Pre-generated reasoning text for top 200 candidates |

---

## 13. Project File Structure

```
Resume Shotlister/
│
├── ARCHITECTURE.md              ← this document
├── README.md                    ← hackathon submission README
├── requirements.txt
├── submission_metadata.yaml     ← required by hackathon rules
│
├── analyze_jd.py                ← Phase 0: JD analysis, generate 3 artifacts
├── precompute.py                ← Phase 1: feature extraction + FAISS index
├── enrich.py                    ← Phase 2: LLM work evidence + reasoning
├── rank.py                      ← Phase 3: produce submission.csv (≤ 5 min)
│
├── src/
│   ├── data/
│   │   └── loader.py            ← load NDJSON candidates efficiently
│   ├── features/
│   │   ├── honeypot.py          ← 5 honeypot detection checks
│   │   ├── extractor.py         ← structured feature extraction
│   │   ├── skills.py            ← skill score computation
│   │   └── embedder.py          ← sentence-transformers wrapper
│   ├── scoring/
│   │   ├── availability.py      ← behavioral signals → availability_score
│   │   ├── career.py            ← career_quality_score
│   │   └── ranker.py            ← final scoring formula
│   └── llm/
│       ├── jd_analyzer.py       ← Phase 0 LLM calls
│       ├── evidence.py          ← Phase 2 work evidence extraction
│       └── reasoning.py         ← Phase 2 reasoning generation
│
├── artifacts/                   ← pre-computed files (committed to git)
│   ├── jd_signals.json
│   ├── jd_skill_weights.json
│   ├── jd_focused_query.txt
│   ├── features.parquet
│   ├── candidates.faiss
│   ├── candidate_ids.json
│   ├── work_evidence.json
│   └── reasoning.json
│
├── evaluation/
│   └── evaluate.py              ← compute NDCG@10/50, MAP locally
│
└── India_runs_data_and_ai_challenge/   ← original data (not committed to git)
    ├── candidates.json                 ← 100K NDJSON (487 MB)
    ├── job_description.docx
    └── ...
```
