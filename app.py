"""
Resume Shotlister — Streamlit demo for HuggingFace Spaces.

Loads pre-computed artifacts and displays the top 100 ranked candidates
for the Senior AI Engineer role. No API calls, no GPU, runs on CPU.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Resume Shotlister | Trendy Brains",
    page_icon="🎯",
    layout="wide",
)

ARTIFACTS = Path("artifacts")


@st.cache_data
def load_data():
    sub      = pd.read_csv("submission.csv")
    reasoning = json.loads((ARTIFACTS / "reasoning.json").read_text())
    features  = pd.read_parquet(ARTIFACTS / "features.parquet")
    work_ev   = json.loads((ARTIFACTS / "work_evidence.json").read_text())
    signals   = json.loads((ARTIFACTS / "jd_signals.json").read_text())
    jd_query  = (ARTIFACTS / "jd_focused_query.txt").read_text().strip()
    return sub, reasoning, features, work_ev, signals, jd_query


sub, reasoning, features, work_ev, signals, jd_query = load_data()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎯 Resume Shotlister")
    st.caption("by **Trendy Brains** · Redrob Hackathon 2026")
    st.divider()

    st.markdown("**Pipeline at a glance:**")
    st.markdown("""
- Phase 0 — JD analysis (1 LLM call)
- Phase 1 — 100K feature extraction + FAISS index
- Phase 2 — LLM work evidence for top 2,000
- Phase 3 — Final ranking (**4 seconds, CPU only**)
""")
    st.divider()

    st.markdown("**Hard constraints met:**")
    st.markdown("""
✅ Zero API calls at rank time
✅ CPU only — no GPU anywhere
✅ Exactly 100 rows, scores non-increasing
✅ 0 honeypots in top 100
✅ Phase 3 runtime: **4.1s / 300s budget**
""")
    st.divider()
    st.markdown("[GitHub ↗](https://github.com/PrateekApurva/resume-shortlister)")


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("Resume Shotlister")
st.markdown(
    "Ranks **100,000 candidates** for a Senior AI Engineer role — "
    "by reading what they actually built, not what they listed in a skills section."
)

tab1, tab2, tab3 = st.tabs(["🏆 Top 100 Candidates", "🔍 Candidate Detail", "⚙️ How It Works"])


# ── Tab 1: Top 100 ───────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Candidates", "100,000")
    c2.metric("NDCG@10", "1.0000")
    c3.metric("Phase 3 Runtime", "4.1s")
    c4.metric("Honeypots in Top 100", "0")

    st.divider()

    search = st.text_input(
        "Search by keyword",
        placeholder="e.g. NLP, product company, embeddings, Bangalore..."
    )

    display = sub.copy()
    display["reasoning_preview"] = display["reasoning"].str[:130] + "..."

    if search:
        mask    = display["reasoning"].str.contains(search, case=False, na=False)
        display = display[mask]
        st.caption(f"{len(display)} candidate(s) match '{search}'")

    st.dataframe(
        display[["rank", "candidate_id", "score", "reasoning_preview"]].rename(columns={
            "rank":             "Rank",
            "candidate_id":     "Candidate ID",
            "score":            "Score",
            "reasoning_preview": "Recruiter Note (preview)",
        }),
        use_container_width=True,
        hide_index=True,
        height=520,
    )


# ── Tab 2: Candidate Detail ───────────────────────────────────────────────────
with tab2:
    options  = [
        f"#{int(r['rank'])}  {r['candidate_id']}  (score: {r['score']:.4f})"
        for _, r in sub.iterrows()
    ]
    selected = st.selectbox("Pick a candidate to inspect", options)

    if selected:
        cid      = selected.split()[1]
        row      = sub[sub["candidate_id"] == cid].iloc[0]
        feat     = features.loc[cid] if cid in features.index else None
        evidence = work_ev.get(cid, {})
        note     = reasoning.get(cid, "No reasoning available.")

        st.subheader(f"Candidate {cid}")
        st.markdown(
            f"**Rank:** #{int(row['rank'])}  ·  "
            f"**Final Score:** `{row['score']:.6f}`"
        )

        st.info(f"**Recruiter note:** {note}")

        if feat is not None:
            st.markdown("#### Score components")
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Career Quality", f"{float(feat['career_quality_score']):.3f}",
                help="Product company history + experience band (20% weight)"
            )
            col2.metric(
                "Skills Fit", f"{float(feat['skills_fit_score']):.3f}",
                help="JD-weighted skill match + platform assessments (15% weight)"
            )
            col3.metric(
                "Availability", f"{float(feat['availability_score']):.3f}",
                help="Open to work, response rate, notice period (12% weight)"
            )

            col4, col5, col6 = st.columns(3)
            col4.metric(
                "Location", f"{float(feat['location_score']):.3f}",
                help="India / willing to relocate to Pune/Noida (8% weight)"
            )
            col5.metric(
                "Honeypot", "Clean ✅" if float(feat['honeypot_multiplier']) == 1.0 else "FLAGGED ❌",
                help="0 = impossible profile detected, 1 = passes all checks"
            )
            col6.metric(
                "Education", f"{float(feat.get('education_score', 0)):.3f}",
                help="Tier-1/2/3 institution scoring"
            )

        if evidence:
            st.markdown("#### LLM work evidence")
            st.caption(
                "GPT-4o-mini read this candidate's career history (not skills section) "
                "and answered these yes/no questions:"
            )
            sig_keys = [s["key"] for s in signals["signals"]]
            cols = st.columns(2)
            for i, key in enumerate(sig_keys):
                val  = evidence.get(key, False)
                icon = "✅" if val else "❌"
                cols[i % 2].markdown(f"{icon} `{key}`")

            st.markdown(
                f"**Company type:** `{evidence.get('company_type', 'unknown')}`  ·  "
                f"**LLM confidence:** `{evidence.get('confidence', 0):.2f}`"
            )
            red_flags = [
                f for f in evidence.get("red_flags", [])
                if not f.startswith("extraction_error")
            ]
            if red_flags:
                st.warning("Red flags: " + " · ".join(red_flags))


# ── Tab 3: How It Works ───────────────────────────────────────────────────────
with tab3:
    st.markdown("## Why not keyword matching?")
    st.markdown("""
Most resume filters match keywords. A candidate who *"built a semantic search system
serving millions at Swiggy"* gets missed if they never wrote "RAG" in their skills section.

This system reads what candidates **actually built** — their career history descriptions —
and ranks by genuine fit, the way a great recruiter would.
""")

    st.divider()
    st.markdown("## 4-Phase Pipeline")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Phase 0 — JD Analysis** *(1 LLM call, ~1 min)*
- GPT-4o-mini reads the job description
- Produces 6 boolean signals ("did they build X?")
- 21 skill weights (1.0 = required, 0.5 = nice-to-have)
- A focused prose query for embedding

**Phase 1 — Preprocessing** *(no API, ~45 min)*
- Honeypot detection: 5 rule-based checks per candidate
- Feature extraction: location, career quality, skills fit, availability
- Embed **work narratives** (career history only, not skills section)
  using `all-MiniLM-L6-v2` → 100K × 384-dim vectors → FAISS index
""")
    with col2:
        st.markdown("""
**Phase 2 — LLM Enrichment** *(~30 min, ~$6)*
- FAISS retrieves top 2,000 by semantic similarity
- GPT-4o-mini evaluates each candidate's career history
  against the 6 signals (20 parallel workers)
- Top 200 get recruiter-style reasoning text

**Phase 3 — Final Ranking** *(CPU only, 4.1 seconds)*
- Embed JD → FAISS search → 2,000 candidates
- Score all 2,000 with the weighted formula
- Sort → top 100 → `submission.csv`
- **Zero API calls. Zero GPU. Pure arithmetic.**
""")

    st.divider()
    st.markdown("## Scoring Formula")
    st.code("""
final_score = (
    0.25 × work_evidence_score    # LLM: did they actually build X?
  + 0.20 × semantic_similarity    # FAISS: career narrative vs JD
  + 0.20 × career_quality_score   # product company + experience band
  + 0.15 × skills_fit_score       # JD-weighted skill match
  + 0.12 × availability_score     # behavioral signals
  + 0.08 × location_score         # India / willing to relocate
) × honeypot_multiplier           # 0.0 kills impossible profiles
""", language="python")

    st.divider()
    st.markdown("## JD Focused Query")
    st.caption("This is the prose description embedded into FAISS to find the best matches:")
    st.info(jd_query)
