"""
World-as-Oracle — Streamlit Dashboard
Guruprasad Venkatakrishnan (2026)
Verslan / predictmarkets.finance / verslan.xyz
"""

import time
import streamlit as st
from datetime import datetime, date
from oracle import WorldOracle, EventCategory
from oracle.impact_scorer import create_article_from_dict
from oracle.models import OracleState
from db import (init_db, save_event, get_event, get_all_events,
                save_article, get_articles, save_history_snapshot,
                get_history, save_resolution, event_exists)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World-as-Oracle",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px; border-radius: 10px; margin-bottom: 20px;
        border: 1px solid #f0b42944;
    }
    .prob-card {
        background: linear-gradient(135deg, #0d1117, #161b22);
        border: 2px solid #f0b429; border-radius: 12px;
        padding: 30px; text-align: center;
    }
    .prob-number { font-size: 80px; font-weight: 800;
                   background: linear-gradient(135deg, #f0b429, #ff6b35);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .state-pill { display: inline-block; padding: 4px 14px; border-radius: 20px;
                  font-size: 13px; font-weight: 600; letter-spacing: 0.5px; }
    .metric-box { background: #0d1117; border: 1px solid #30363d;
                  border-radius: 8px; padding: 16px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── Init DB ────────────────────────────────────────────────────────────────
init_db()

# ── API Key notice (optional) ──────────────────────────────────────────────
import os
if not os.getenv("ANTHROPIC_API_KEY"):
    st.info(
        "**Auto-scoring is off** — no ANTHROPIC_API_KEY set. "
        "You can still use the oracle: just set **Raw Impact** and **Direction** manually when ingesting articles. "
        "The temporal filter, probability engine, and all math runs without any API key.",
        icon="ℹ️"
    )

# ── Oracle cache ───────────────────────────────────────────────────────────
if "oracles" not in st.session_state:
    st.session_state.oracles = {}

def get_oracle(event_id: str) -> WorldOracle:
    if event_id not in st.session_state.oracles:
        ev = get_event(event_id)
        if not ev:
            st.error(f"Event {event_id} not found")
            st.stop()
        from oracle.models import SourceTier, Article
        from oracle.config import CATEGORY_CONFIGS
        cat_map = {c.value: c for c in EventCategory}
        oracle = WorldOracle(
            event_id=event_id,
            event_description=ev["description"],
            resolution_criteria=ev["resolution"],
            category=cat_map.get(ev["category"], EventCategory.CORPORATE_LEGAL),
        )
        oracle._prior = ev["prior"]
        tier_map = {i: SourceTier(i) for i in range(1, 7)}
        for row in get_articles(event_id):
            a = Article(
                article_id=row["article_id"], source_name=row["source_name"],
                tier=tier_map.get(row["tier"], SourceTier.TIER4_WIRE),
                publication_time=row["publication_time"], headline=row["headline"],
                content_summary=row.get("content_summary", ""),
                raw_impact=row.get("raw_impact", 0.0), direction=row.get("direction", 0),
                independence_score=row.get("independence_score", 0.0),
                reasoning_chain=row.get("reasoning_chain", ""),
            )
            oracle._articles.append(a)
        st.session_state.oracles[event_id] = oracle
    return st.session_state.oracles[event_id]

# ── State colours ──────────────────────────────────────────────────────────
STATE_COLORS = {
    "BASELINE":            ("#6e7681", "#21262d"),
    "SHOCK_ACTIVE":        ("#f0b429", "#3d1f00"),
    "BUILDING":            ("#3fb950", "#1f3d00"),
    "SUSTAINED":           ("#39d353", "#003d1f"),
    "CONTESTED":           ("#f85149", "#3d0000"),
    "INSUFFICIENT_SIGNAL": ("#6e7681", "#161b22"),
}

def state_badge(state: str) -> str:
    fg, bg = STATE_COLORS.get(state, ("#8b949e", "#21262d"))
    return f'<span style="background:{bg};color:{fg};padding:4px 12px;border-radius:20px;font-size:13px;font-weight:600;">{state}</span>'

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1 style="color:#f0b429;margin:0;">⬡ World-as-Oracle</h1>
  <p style="color:#8b949e;margin:4px 0 0 0;">
    Adversarial-Resistant AI Probability Estimation for Event-Driven Financial Instruments<br>
    <small>Guruprasad Venkatakrishnan (2026) · Verslan · predictmarkets.finance · verslan.xyz</small>
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Event Management ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⬡ Events")

    events = get_all_events()
    event_ids = [e["event_id"] for e in events]

    if not event_ids:
        st.info("No events yet. Create one below.")
        selected_event = None
    else:
        selected_event = st.selectbox("Select Event", event_ids)

    st.divider()

    with st.expander("➕ Create New Event", expanded=not bool(event_ids)):
        with st.form("create_event_form"):
            new_id = st.text_input("Event ID", placeholder="fed_rate_jun2023")
            new_desc = st.text_area("Description", placeholder="Federal Reserve raises rates at June 14 2023 FOMC", height=80)
            new_res = st.text_area("Resolution Criteria", placeholder="Fed funds rate increases by 25bps or more", height=80)
            new_cat = st.selectbox("Category", [
                "central_bank", "corporate_legal", "geopolitical",
                "electoral", "macro_data", "sovereign_credit", "crypto_protocol"
            ])
            col1, col2, col3 = st.columns(3)
            with col1: hist_rate = st.number_input("Hist. Base Rate", 0.0, 1.0, 0.15, 0.01)
            with col2: struct_prior = st.number_input("Structural Prior", 0.0, 1.0, 0.20, 0.01)
            with col3: mkt_prior = st.number_input("Market Prior", 0.0, 1.0, 0.0, 0.01)

            if st.form_submit_button("Create Event", type="primary"):
                if not new_id or not new_desc:
                    st.error("Event ID and description required")
                elif event_exists(new_id):
                    st.error(f"Event '{new_id}' already exists")
                else:
                    cat_map = {c.value: c for c in EventCategory}
                    category = cat_map.get(new_cat, EventCategory.CORPORATE_LEGAL)
                    oracle = WorldOracle(
                        event_id=new_id, event_description=new_desc,
                        resolution_criteria=new_res, category=category,
                    )
                    mkt = mkt_prior if mkt_prior > 0 else None
                    oracle.set_prior(hist_rate, struct_prior, mkt)
                    save_event(new_id, new_desc, new_res, new_cat, oracle._prior,
                               hist_rate, struct_prior, mkt)
                    st.session_state.oracles[new_id] = oracle
                    st.success(f"✓ Created '{new_id}' — Prior: {oracle._prior:.1%}")
                    st.rerun()

# ── Main area ──────────────────────────────────────────────────────────────
if not selected_event:
    st.markdown("### ← Create your first event in the sidebar")
    st.stop()

oracle = get_oracle(selected_event)
ev_info = get_event(selected_event)

# Event info bar
st.markdown(f"**{ev_info['description']}**")
st.caption(f"Category: `{ev_info['category']}` · Resolution: _{ev_info['resolution']}_")
st.divider()

tab_oracle, tab_ingest, tab_history, tab_resolve = st.tabs([
    "📊 Oracle Output", "📥 Ingest Article", "📈 History", "✅ Resolve"
])

# ── TAB 1: Oracle Output ───────────────────────────────────────────────────
with tab_oracle:
    out = oracle.compute()
    p = out.probability

    col_prob, col_meta = st.columns([1, 2])

    with col_prob:
        st.markdown(f"""
        <div class="prob-card">
          <div class="prob-number">{p:.0%}</div>
          <div style="color:#8b949e;font-size:14px;margin:8px 0;">P(Event Resolves YES)</div>
          {state_badge(out.state.value)}
          <div style="color:#8b949e;font-size:12px;margin-top:12px;">
            CI: [{out.ci_lower:.1%}, {out.ci_upper:.1%}]
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_meta:
        st.markdown("#### Signal Decomposition")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Prior P₀", f"{out.prior:.1%}", help="α·H(E) + β·S(E) + γ·M(E)")
        with c2:
            delta_shock = out.fast_shock * 100
            st.metric("Fast Shock F(t)", f"{out.fast_shock:+.3f}",
                      delta=f"{delta_shock:+.1f}pp",
                      delta_color="normal" if delta_shock >= 0 else "inverse",
                      help="Exponentially decaying. Capped at ±F_max=20%")
        with c3:
            delta_p = out.persistent_signal * 100
            st.metric("Persistent P(t)", f"{out.persistent_signal:+.3f}",
                      delta=f"{delta_p:+.1f}pp",
                      delta_color="normal" if delta_p >= 0 else "inverse",
                      help=f"Threshold-gated. Requires ρ_min={oracle.config.rho_min} independent sources")

        st.markdown("#### Source Intelligence")
        c4, c5, c6 = st.columns(3)
        with c4:
            st.metric("Independent Sources", out.independent_source_count,
                      help=f"Qualifying sources with ι(a_i) ≥ {oracle.config.theta_ind}")
        with c5:
            st.metric("Total Articles", out.total_articles_processed)
        with c6:
            st.metric("ρ_min Required", oracle.config.rho_min,
                      help="Min independent sources to activate persistent signal")

        # Confidence interval bar
        st.markdown("**Confidence Interval**")
        ci_pct = int((out.ci_upper - out.ci_lower) * 100)
        lo_pct = int(out.ci_lower * 100)
        st.progress(min(out.ci_upper, 1.0), text=f"[{out.ci_lower:.1%} → {out.ci_upper:.1%}]  width={ci_pct}pp")

    # Audit trail
    if out.audit_trail:
        st.markdown("#### Qualifying Sources (Audit Trail)")
        for a in out.audit_trail:
            direction_icon = "🔴" if a.get("signed_impact", 0) < 0 else "🟢"
            with st.expander(f"{direction_icon} Tier {a['tier']} · {a['source']} · impact={a.get('signed_impact', '?')}"):
                st.write(f"**Independence score:** {a.get('independence_score', 0):.3f}")
                if a.get("reasoning"):
                    st.write(f"**Reasoning:** {a['reasoning']}")
    else:
        st.info("No qualifying sources yet. Ingest articles with independence ≥ θ_ind=0.40 and |impact| ≥ θ_imp=3.0")

    if st.button("↻ Refresh", key="refresh_btn"):
        st.rerun()

# ── TAB 2: Ingest Article ──────────────────────────────────────────────────
with tab_ingest:
    st.markdown("### Ingest Article or Signal")

    with st.form("ingest_form"):
        col1, col2 = st.columns(2)
        with col1:
            art_id = st.text_input("Article ID", value=f"art_{int(time.time())}")
            source = st.text_input("Source Name", placeholder="Bloomberg / DOJ Filing / Reuters")
            tier = st.selectbox("Source Tier", options=[1, 2, 3, 4, 5, 6],
                                format_func=lambda x: {
                                    1: "Tier 1 — Court filing, regulatory order, central bank statement",
                                    2: "Tier 2 — Local financial press (ET, Business Standard)",
                                    3: "Tier 3 — Regional international (FT, Nikkei Asia)",
                                    4: "Tier 4 — International wire (Bloomberg, Reuters)",
                                    5: "Tier 5 — Capital flow signal (FII, CDS, bond yield)",
                                    6: "Tier 6 — Unverified / social media",
                                }[x], index=3)
            pub_date = st.date_input("Publication Date", value=date.today())

        with col2:
            raw_impact = st.slider("Raw Impact (1–10)", 1.0, 10.0, 5.0, 0.5,
                                   help="How strongly does this article move the probability?")
            direction = st.radio("Direction", options=[1, 0, -1],
                                 format_func=lambda x: {
                                     1: "🔴 +1 (Increases probability)",
                                     0: "⚪ 0 (Neutral / informational)",
                                     -1: "🟢 -1 (Decreases probability)"
                                 }[x], index=0)
            url = st.text_input("URL (optional)", placeholder="https://bloomberg.com/...")

        headline = st.text_input("Headline *", placeholder="DOJ Files Criminal Charges Against...")
        content = st.text_area("Content Summary", placeholder="Brief summary of the key claims in this article...", height=100)
        reasoning = st.text_input("Reasoning (optional)", placeholder="Why does this article impact the probability?")

        submitted = st.form_submit_button("⬡ Ingest Article", type="primary")

    if submitted:
        if not source or not headline:
            st.error("Source name and headline are required")
        else:
            pub_time = datetime.combine(pub_date, datetime.min.time()).timestamp()
            data = {
                "article_id": art_id, "source_name": source, "tier": tier,
                "publication_time": pub_time, "headline": headline,
                "content_summary": content, "url": url,
                "raw_impact": raw_impact, "direction": direction,
                "reasoning_chain": reasoning,
            }
            article = create_article_from_dict(data)
            oracle.ingest_articles([article], auto_score_independence=True, auto_score_impact=False)
            save_article(selected_event, oracle._articles[-1])

            out = oracle.compute()
            snapshot = {
                "timestamp": time.time(), "probability": round(out.probability, 4),
                "state": out.state.value, "fast_shock": round(out.fast_shock, 4),
                "persistent_signal": round(out.persistent_signal, 4),
                "independent_sources": out.independent_source_count,
            }
            save_history_snapshot(selected_event, snapshot)

            inde_score = oracle._articles[-1].independence_score
            st.success(f"""
            ✓ Ingested successfully
            - **Independence score:** {inde_score:.3f} {'✓ Qualifies' if inde_score >= oracle.config.theta_ind else '✗ Below θ_ind — derivative source'}
            - **Signed impact:** {oracle._articles[-1].signed_impact:+.2f}
            - **New probability:** {out.probability:.1%} [{out.state.value}]
            """)

# ── TAB 3: History ─────────────────────────────────────────────────────────
with tab_history:
    history = get_history(selected_event)

    if not history:
        st.info("No history yet. Ingest articles to build probability history.")
    else:
        import pandas as pd

        df = pd.DataFrame(history)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df["probability_pct"] = df["probability"] * 100

        st.markdown("#### Probability Over Time")
        st.line_chart(df.set_index("datetime")["probability_pct"],
                      use_container_width=True, height=280,
                      color="#f0b429")

        st.markdown("#### Signal Decomposition Over Time")
        decomp_cols = [c for c in ["fast_shock", "persistent_signal"] if c in df.columns]
        if decomp_cols:
            st.line_chart(df.set_index("datetime")[decomp_cols],
                          use_container_width=True, height=200)

        st.markdown("#### History Table")
        display_df = df[["datetime", "probability_pct", "state",
                          "fast_shock", "persistent_signal", "independent_sources"]].copy()
        display_df.columns = ["Datetime", "P(%) ", "State", "Fast Shock", "Persistent", "Sources"]
        display_df = display_df.sort_values("Datetime", ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ── TAB 4: Resolve ─────────────────────────────────────────────────────────
with tab_resolve:
    st.markdown("### Resolve Event")
    st.warning("⚠️ Resolving records the actual outcome and computes Brier Score. This cannot be undone.")

    if ev_info.get("outcome") is not None:
        outcome_label = "YES ✓" if ev_info["outcome"] == 1 else "NO ✗"
        st.success(f"This event has already been resolved: **{outcome_label}**")
    else:
        out = oracle.compute()
        st.markdown(f"**Current oracle probability:** {out.probability:.1%}")
        st.markdown(f"**State:** {out.state.value}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Resolve YES (Event Happened)", type="primary", use_container_width=True):
                oracle.record_resolution(1)
                report = oracle.calibration_report()
                save_resolution(selected_event, 1, out.probability, report.get("overall_brier_score"))
                bs = report.get("overall_brier_score")
                st.success(f"Resolved YES. Brier Score: {bs:.4f}" if bs else "Resolved YES.")
                st.balloons()
                st.rerun()
        with col2:
            if st.button("❌ Resolve NO (Event Did Not Happen)", use_container_width=True):
                oracle.record_resolution(0)
                report = oracle.calibration_report()
                save_resolution(selected_event, 0, out.probability, report.get("overall_brier_score"))
                bs = report.get("overall_brier_score")
                st.success(f"Resolved NO. Brier Score: {bs:.4f}" if bs else "Resolved NO.")
                st.rerun()

        with st.expander("What is Brier Score?"):
            st.markdown("""
            **BS = (P_oracle - Outcome)²**

            - **0.00** = Perfect prediction
            - **0.10** = Very good (better than most prediction markets)
            - **0.25** = No skill (same as always saying 50%)

            The oracle targets BS < 0.12 (better than CME FedWatch baseline of ~0.11).
            """)

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#6e7681;font-size:12px;padding:8px;">
    World-as-Oracle v1.0 · Guruprasad Venkatakrishnan (2026) ·
    Verslan · predictmarkets.finance · verslan.xyz
</div>
""", unsafe_allow_html=True)
