"""
World-as-Oracle — Professional Trading Dashboard
Bayesian Engine + Vol Surface + Greeks + Correlation + Live Feed

Guruprasad Venkatakrishnan (2026)
Verslan / predictmarkets.finance / verslan.xyz
"""

import time, os, math
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, date, timedelta

from oracle import WorldOracle, EventCategory
from oracle.models import OracleState, SourceTier
from oracle.impact_scorer import create_article_from_dict
from oracle.bayesian_engine import BayesianOracleEngine
from oracle.vol_surface import compute_vol_surface, compute_greeks
from oracle.correlation import CorrelationMatrix
from oracle.live_feed import live_ingest, build_query_from_event
from oracle.independence import score_all_articles
from db import (init_db, save_event, get_event, get_all_events,
                save_article, get_articles, save_history_snapshot,
                get_history, save_resolution, event_exists)

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="World-as-Oracle | predictmarkets.finance",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono&display=swap');

* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0a0e1a; }

.oracle-header {
    background: linear-gradient(135deg, #0d1421 0%, #111827 100%);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 20px 28px; margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
}
.prob-mega {
    font-size: 88px; font-weight: 800; line-height: 1;
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #f59e0b, #ef4444, #f59e0b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: pulse 3s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.85} }

.metric-card {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 10px; padding: 16px; text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #f59e0b; }
.metric-val { font-size: 28px; font-weight: 700; color: #f59e0b; font-family: 'JetBrains Mono'; }
.metric-lbl { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }

.state-BASELINE          { background:#1f2937;color:#9ca3af; }
.state-SHOCK_ACTIVE      { background:#451a03;color:#fbbf24; }
.state-BUILDING          { background:#052e16;color:#34d399; }
.state-SUSTAINED         { background:#022c22;color:#10b981; }
.state-CONTESTED         { background:#450a0a;color:#f87171; }
.state-INSUFFICIENT_SIGNAL{background:#111827;color:#6b7280; }

.state-pill {
    display:inline-block; padding:6px 16px; border-radius:20px;
    font-size:13px; font-weight:600; letter-spacing:0.5px;
}
.vol-quiet  { color: #34d399; }
.vol-active { color: #fbbf24; }
.vol-crisis { color: #f87171; }

.greek-box {
    background:#0d1421; border:1px solid #1e3a5f; border-radius:8px;
    padding:14px; margin-bottom:8px;
}
.greek-val { font-size:22px; font-weight:700; color:#60a5fa; font-family:'JetBrains Mono'; }
.greek-lbl { font-size:11px; color:#6b7280; }
.greek-int { font-size:12px; color:#9ca3af; margin-top:4px; }

.lr-bar-pos { background: linear-gradient(90deg, #059669, #10b981); height:8px; border-radius:4px; }
.lr-bar-neg { background: linear-gradient(90deg, #dc2626, #ef4444); height:8px; border-radius:4px; }

.live-dot { display:inline-block; width:8px; height:8px; background:#10b981;
            border-radius:50%; animation: blink 1s infinite; margin-right:6px; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

[data-testid="stSidebar"] { background: #0d1421 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label { color: #9ca3af !important; }
div[data-testid="metric-container"] { background:#111827; border:1px solid #1f2937; border-radius:8px; padding:12px; }
</style>
""", unsafe_allow_html=True)

# ── Init ──────────────────────────────────────────────────────────────────
init_db()

if "oracles" not in st.session_state:
    st.session_state.oracles = {}
if "bayesian" not in st.session_state:
    st.session_state.bayesian = {}
if "correlation" not in st.session_state:
    st.session_state.correlation = CorrelationMatrix()
if "live_articles" not in st.session_state:
    st.session_state.live_articles = {}

def get_oracle(event_id):
    if event_id not in st.session_state.oracles:
        ev = get_event(event_id)
        if not ev: st.error(f"Event {event_id} not found"); st.stop()
        cat_map = {c.value: c for c in EventCategory}
        from oracle.models import Article
        oracle = WorldOracle(
            event_id=event_id, event_description=ev["description"],
            resolution_criteria=ev["resolution"],
            category=cat_map.get(ev["category"], EventCategory.CORPORATE_LEGAL),
        )
        oracle._prior = ev["prior"]
        tier_map = {i: SourceTier(i) for i in range(1,7)}
        for row in get_articles(event_id):
            a = Article(
                article_id=row["article_id"], source_name=row["source_name"],
                tier=tier_map.get(row["tier"], SourceTier.TIER4_WIRE),
                publication_time=row["publication_time"], headline=row["headline"],
                content_summary=row.get("content_summary",""),
                raw_impact=row.get("raw_impact",0.0), direction=row.get("direction",0),
                independence_score=row.get("independence_score",0.0),
                reasoning_chain=row.get("reasoning_chain",""),
            )
            oracle._articles.append(a)
        st.session_state.oracles[event_id] = oracle
    return st.session_state.oracles[event_id]

def get_bayesian(event_id, oracle):
    if event_id not in st.session_state.bayesian:
        from oracle.config import CATEGORY_CONFIGS
        config = oracle.config
        st.session_state.bayesian[event_id] = BayesianOracleEngine(oracle._prior, config)
    return st.session_state.bayesian[event_id]

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⬡ predictmarkets.finance")
    st.caption("World-as-Oracle · OTC Derivatives Desk")
    st.divider()

    events = get_all_events()
    event_ids = [e["event_id"] for e in events]
    selected_event = st.selectbox("Active Event", event_ids if event_ids else ["—"])

    if st.button("⟳ Live Ingest", use_container_width=True, type="primary"):
        if selected_event and selected_event != "—":
            oracle = get_oracle(selected_event)
            ev = get_event(selected_event)
            query = build_query_from_event(ev["description"])
            with st.spinner(f"Fetching live articles for: {query}..."):
                articles = live_ingest(query, hours_back=48, max_per_source=15)
                if articles:
                    oracle.ingest_articles(articles, auto_score_independence=True)
                    st.session_state.live_articles[selected_event] = articles
                    st.success(f"✓ Ingested {len(articles)} live articles")
                else:
                    st.warning("No articles found. Try GDELT manually.")

    st.divider()

    with st.expander("➕ New Event"):
        with st.form("new_event"):
            eid     = st.text_input("Event ID")
            edesc   = st.text_area("Description", height=70)
            eres    = st.text_area("Resolution Criteria", height=70)
            ecat    = st.selectbox("Category", ["central_bank","corporate_legal","geopolitical","electoral","macro_data","sovereign_credit","crypto_protocol"])
            c1,c2   = st.columns(2)
            with c1: ehist = st.number_input("Base Rate", 0.0, 1.0, 0.15, 0.01)
            with c2: estruct = st.number_input("Structural", 0.0, 1.0, 0.20, 0.01)
            emkt = st.number_input("Market Prior", 0.0, 1.0, 0.0, 0.01)
            notional = st.number_input("Notional ($)", 100000, 100000000, 1000000, 100000)
            res_date = st.date_input("Resolution Date", value=date.today() + timedelta(days=90))

            if st.form_submit_button("Create", type="primary"):
                if eid and edesc:
                    cat_map = {c.value: c for c in EventCategory}
                    cat = cat_map.get(ecat, EventCategory.CORPORATE_LEGAL)
                    oracle = WorldOracle(event_id=eid, event_description=edesc,
                                        resolution_criteria=eres, category=cat)
                    mkt = emkt if emkt > 0 else None
                    oracle.set_prior(ehist, estruct, mkt)
                    save_event(eid, edesc, eres, ecat, oracle._prior, ehist, estruct, mkt)
                    st.session_state.oracles[eid] = oracle
                    st.session_state.correlation.add_event(eid)
                    st.success(f"✓ Created {eid}")
                    st.rerun()

    # Notional & resolution date for Greeks
    st.divider()
    st.markdown("**Derivatives Desk**")
    notional_desk = st.number_input("Notional ($)", 100000, 100000000, 1000000, 100000, key="desk_notional")
    fixed_prob = st.slider("Fixed Probability (K)", 0.05, 0.95, 0.50, 0.01, key="desk_k")
    res_days = st.number_input("Days to Resolution", 1, 365, 90, key="desk_days")

# ── Main ───────────────────────────────────────────────────────────────────
if not selected_event or selected_event == "—":
    st.markdown("## ⬡ World-as-Oracle")
    st.markdown("**Create an event in the sidebar to begin.**")
    st.stop()

oracle = get_oracle(selected_event)
ev_info = get_event(selected_event)
bayes = get_bayesian(selected_event, oracle)
current_time = time.time()

# Run oracle
out = oracle.compute(current_time)
p = out.probability

# Bayesian posterior
bayes_result = bayes.compute(oracle._articles, current_time, use_bootstrap=len(oracle._articles) >= 3)
p_bayes = bayes_result["posterior"]
p_std = bayes_result["std"]

# Use Bayesian P if we have enough signals, else oracle P
p_display = p_bayes if bayes_result["n_signals"] >= 2 else p

# Volatility surface
posterior_hist = [h["probability"] for h in get_history(selected_event)[-30:]]
vol_data = compute_vol_surface(
    p=p_display,
    category=ev_info["category"],
    state=out.state,
    days_to_resolution=res_days,
    n_signals_24h=bayes_result["n_signals"],
    posterior_history=posterior_hist,
)

# Greeks
greeks = compute_greeks(
    p=p_display,
    notional=notional_desk,
    daily_vol=vol_data["daily_vol"],
    days_to_resolution=res_days,
    fixed_probability=fixed_prob,
)

# ── HEADER ─────────────────────────────────────────────────────────────────
state_colors = {
    "BASELINE":"#4b5563","SHOCK_ACTIVE":"#fbbf24","BUILDING":"#34d399",
    "SUSTAINED":"#10b981","CONTESTED":"#f87171","INSUFFICIENT_SIGNAL":"#6b7280"
}
sc = state_colors.get(out.state.value, "#9ca3af")
vol_color = {"QUIET":"#34d399","ACTIVE":"#fbbf24","CRISIS":"#f87171"}.get(vol_data["regime"],"#9ca3af")

col_prob, col_info = st.columns([1, 2])

with col_prob:
    st.markdown(f"""
    <div style="background:#0d1421;border:2px solid #f59e0b44;border-radius:16px;padding:32px;text-align:center;">
      <div class="prob-mega">{p_display:.0%}</div>
      <div style="color:#9ca3af;font-size:13px;margin:8px 0;">P(Event Resolves YES)</div>
      <div><span class="state-pill state-{out.state.value}"
           style="background:{sc}22;color:{sc};border:1px solid {sc}44;">
        {out.state.value}
      </span></div>
      <div style="color:#6b7280;font-size:12px;margin-top:12px;">
        Bayesian CI [{bayes_result['ci_5']:.1%}, {bayes_result['ci_95']:.1%}]
      </div>
      <div style="color:{vol_color};font-size:13px;margin-top:6px;font-weight:600;">
        σ = {vol_data['daily_vol']:.1%}/day · {vol_data['regime']}
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_info:
    st.markdown(f"**{ev_info['description']}**")
    st.caption(ev_info['resolution'])

    # Key metrics row
    m1,m2,m3,m4,m5 = st.columns(5)
    with m1: st.metric("Bayesian P", f"{p_bayes:.1%}", delta=f"{(p_bayes-oracle._prior)*100:+.1f}pp vs prior")
    with m2: st.metric("Oracle P", f"{p:.1%}", delta=f"{(p-p_bayes)*100:+.1f}pp vs Bayes")
    with m3: st.metric("Epistemic σ", f"{p_std:.1%}", help="Bootstrap uncertainty around Bayesian posterior")
    with m4: st.metric("Ann. Vol", f"{vol_data['annual_vol']:.0%}", delta=vol_data['regime'])
    with m5: st.metric("Signals", f"{bayes_result['n_signals']}", help="Qualifying independent signals")

st.divider()

# ── TABS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Oracle", "⚡ Greeks & Pricing", "📈 Vol Surface",
    "🔗 Correlation", "📥 Ingest", "📰 Live Feed"
])

# ── TAB 1: Oracle ──────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Bayesian Signal Decomposition")
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{oracle._prior:.1%}</div>
          <div class="metric-lbl">Prior P₀ = α·H + β·S + γ·M</div>
        </div>
        """, unsafe_allow_html=True)

        if oracle._articles:
            st.markdown("**Likelihood Ratios by Source**")
            from oracle.bayesian_engine import likelihood_ratio
            from oracle.temporal_filter import decay_factor, count_confirming_signals

            for art in sorted(oracle._articles, key=lambda a: abs(a.signed_impact), reverse=True)[:6]:
                dt = max(0, (current_time - art.publication_time) / 86400)
                nc = count_confirming_signals(art, oracle._articles, oracle.config.theta_ind, oracle.config.theta_imp)
                d = decay_factor(dt, oracle.config.decay_rate_lambda, nc)
                lr = likelihood_ratio(art, d)
                lr_display = f"{lr:.2f}×"
                bar_w = min(100, abs(lr-1) * 50)
                bar_class = "lr-bar-pos" if lr >= 1 else "lr-bar-neg"
                direction_icon = "🔴" if art.direction > 0 else ("🟢" if art.direction < 0 else "⚪")
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                  <div style="font-size:12px;color:#9ca3af;">{direction_icon} {art.source_name[:30]} · T{art.tier.value}</div>
                  <div style="font-size:11px;color:#6b7280;">{art.headline[:60]}...</div>
                  <div style="display:flex;align-items:center;gap:8px;margin-top:4px;">
                    <div class="{bar_class}" style="width:{bar_w}%;"></div>
                    <span style="color:#f59e0b;font-size:12px;font-weight:600;font-family:monospace;">{lr_display}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    with c2:
        st.markdown("#### Temporal Filter")
        c2a, c2b = st.columns(2)
        with c2a:
            st.metric("Fast Shock F(t)", f"{out.fast_shock:+.3f}", help="Exponential decay, capped ±20%")
        with c2b:
            st.metric("Persistent P(t)", f"{out.persistent_signal:+.3f}", help="Threshold-gated, ±80%")

        # Prior → Posterior waterfall
        st.markdown("**Bayesian Update Path**")
        waterfall_data = {
            "Component": ["Prior", "Fast Shock", "Persistent", "Bayes Posterior"],
            "Value": [oracle._prior, out.fast_shock, out.persistent_signal, p_bayes],
            "Color": ["#60a5fa", "#fbbf24" if out.fast_shock > 0 else "#f87171",
                     "#34d399" if out.persistent_signal > 0 else "#f87171", "#f59e0b"],
        }
        wf_df = pd.DataFrame(waterfall_data)
        st.dataframe(wf_df[["Component","Value"]].assign(
            Value=wf_df["Value"].apply(lambda x: f"{x:+.3f}")
        ), hide_index=True, use_container_width=True)

    with c3:
        st.markdown("#### Oracle State Machine")
        for state in OracleState:
            is_active = state == out.state
            color = state_colors.get(state.value, "#4b5563")
            bg = f"{color}33" if is_active else "#111827"
            border = color if is_active else "#1f2937"
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:8px;
                 padding:8px 12px;margin-bottom:6px;">
              <span style="color:{color};font-weight:{'700' if is_active else '400'};font-size:13px;">
                {'▶ ' if is_active else '  '}{state.value}
              </span>
            </div>
            """, unsafe_allow_html=True)

    # History chart
    history = get_history(selected_event)
    if len(history) >= 2:
        st.markdown("#### Probability History")
        hist_df = pd.DataFrame(history)
        hist_df["datetime"] = pd.to_datetime(hist_df["timestamp"], unit="s")
        hist_df["P (%)"] = hist_df["probability"] * 100
        hist_df["Prior"] = oracle._prior * 100

        chart_df = hist_df.set_index("datetime")[["P (%)", "Prior"]]
        st.line_chart(chart_df, use_container_width=True, height=220, color=["#f59e0b", "#374151"])

# ── TAB 2: Greeks & Pricing ───────────────────────────────────────────────
with tab2:
    st.markdown(f"### Event Probability Swap · Notional: ${notional_desk:,.0f} · K = {fixed_prob:.0%}")

    g1, g2, g3, g4 = st.columns(4)

    with g1:
        st.markdown(f"""
        <div class="greek-box">
          <div class="greek-lbl">SWAP DELTA (∂V/∂P)</div>
          <div class="greek-val">${greeks['swap_delta']:,.0f}</div>
          <div class="greek-int">{greeks['interpretation']['swap_delta']}</div>
        </div>
        """, unsafe_allow_html=True)

    with g2:
        st.markdown(f"""
        <div class="greek-box">
          <div class="greek-lbl">BINARY DELTA</div>
          <div class="greek-val">{greeks['binary_delta']:,.1f}</div>
          <div class="greek-int">Binary option sensitivity</div>
        </div>
        """, unsafe_allow_html=True)

    with g3:
        st.markdown(f"""
        <div class="greek-box">
          <div class="greek-lbl">GAMMA (∂²V/∂P²)</div>
          <div class="greek-val">{greeks['gamma']:.4f}</div>
          <div class="greek-int">{greeks['interpretation']['gamma']}</div>
        </div>
        """, unsafe_allow_html=True)

    with g4:
        st.markdown(f"""
        <div class="greek-box">
          <div class="greek-lbl">THETA (daily decay)</div>
          <div class="greek-val">${abs(greeks['theta_daily']):,.0f}</div>
          <div class="greek-int">{greeks['interpretation']['theta']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    v1, v2, v3 = st.columns(3)

    with v1:
        st.markdown(f"""
        <div class="greek-box">
          <div class="greek-lbl">VEGA (per 1% vol)</div>
          <div class="greek-val">${abs(greeks['vega_per_1pct_vol']):,.0f}</div>
          <div class="greek-int">{greeks['interpretation']['vega']}</div>
        </div>
        """, unsafe_allow_html=True)

    with v2:
        st.markdown(f"""
        <div class="greek-box">
          <div class="greek-lbl">BINARY OPTION PRICE</div>
          <div class="greek-val">${greeks['binary_option_price']:,.0f}</div>
          <div class="greek-int">Normal approx, K={fixed_prob:.0%}</div>
        </div>
        """, unsafe_allow_html=True)

    with v3:
        st.markdown(f"""
        <div class="greek-box">
          <div class="greek-lbl">d1 (Black-Scholes)</div>
          <div class="greek-val">{greeks['d1']:.3f}</div>
          <div class="greek-int">Standard deviations from strike</div>
        </div>
        """, unsafe_allow_html=True)

    # P&L scenario table
    st.markdown("#### P&L Scenarios")
    scenarios = []
    for p_scenario in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        pnl_long = (p_scenario - p_display) * notional_desk
        pnl_short = -pnl_long
        scenarios.append({
            "Oracle P": f"{p_scenario:.0%}",
            "ΔP vs Now": f"{(p_scenario-p_display)*100:+.0f}pp",
            "Long Float P&L": f"${pnl_long:+,.0f}",
            "Short Float P&L": f"${pnl_short:+,.0f}",
            "Margin Used": f"${abs(pnl_long):,.0f}",
        })

    pnl_df = pd.DataFrame(scenarios)
    st.dataframe(pnl_df, hide_index=True, use_container_width=True)

# ── TAB 3: Vol Surface ─────────────────────────────────────────────────────
with tab3:
    st.markdown("### Probability Volatility Surface")
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        vc = {"QUIET":"#34d399","ACTIVE":"#fbbf24","CRISIS":"#f87171"}.get(vol_data["regime"])
        st.markdown(f"""<div class="metric-card">
        <div class="metric-val" style="color:{vc};">{vol_data['daily_vol']:.1%}</div>
        <div class="metric-lbl">Daily Vol σ/day</div></div>""", unsafe_allow_html=True)
    with v2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-val">{vol_data['annual_vol']:.0%}</div>
        <div class="metric-lbl">Annual Vol σ/√252</div></div>""", unsafe_allow_html=True)
    with v3:
        r1d = vol_data["vol_1d_range"]
        st.markdown(f"""<div class="metric-card">
        <div class="metric-val" style="font-size:18px;">[{r1d[0]:.1%}, {r1d[1]:.1%}]</div>
        <div class="metric-lbl">1-Day 90% Range</div></div>""", unsafe_allow_html=True)
    with v4:
        r7d = vol_data["vol_7d_range"]
        st.markdown(f"""<div class="metric-card">
        <div class="metric-val" style="font-size:18px;">[{r7d[0]:.1%}, {r7d[1]:.1%}]</div>
        <div class="metric-lbl">7-Day 90% Range</div></div>""", unsafe_allow_html=True)

    st.divider()

    # Vol smile across probability levels
    st.markdown("#### Vol Smile — σ across P levels")
    smile_data = []
    for p_lvl in np.arange(0.05, 0.96, 0.05):
        vd = compute_vol_surface(p_lvl, ev_info["category"], out.state,
                                 res_days, bayes_result["n_signals"])
        smile_data.append({"P": p_lvl, "Daily Vol": vd["daily_vol"], "Annual Vol": vd["annual_vol"]})
    smile_df = pd.DataFrame(smile_data).set_index("P")
    st.line_chart(smile_df["Annual Vol"], use_container_width=True, height=220, color="#f59e0b")

    # Vol term structure
    st.markdown("#### Vol Term Structure — σ by days to resolution")
    term_data = []
    for days in [1,3,7,14,30,60,90,180,365]:
        vd = compute_vol_surface(p_display, ev_info["category"], out.state,
                                 days, bayes_result["n_signals"])
        term_data.append({"Days": days, "Daily Vol": vd["daily_vol"]})
    term_df = pd.DataFrame(term_data).set_index("Days")
    st.line_chart(term_df["Daily Vol"], use_container_width=True, height=200, color="#60a5fa")

    st.divider()
    st.markdown("#### Vol Surface Components")
    comps = vol_data["components"]
    comp_df = pd.DataFrame([
        {"Component": "Base Vol (category)", "Multiplier": f"{comps['base_vol']:.3f}"},
        {"Component": "Smile (level adj.)", "Multiplier": f"{comps['smile_multiplier']:.3f}×"},
        {"Component": "State multiplier", "Multiplier": f"{comps['state_multiplier']}×"},
        {"Component": "Term structure", "Multiplier": f"{comps['term_multiplier']}×"},
        {"Component": "Signal density", "Multiplier": f"{comps['density_multiplier']}×"},
        {"Component": "→ Final daily vol", "Multiplier": f"{vol_data['daily_vol']:.3f}"},
    ])
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

# ── TAB 4: Correlation ─────────────────────────────────────────────────────
with tab4:
    st.markdown("### Cross-Event Correlation")
    corr = st.session_state.correlation

    if len(event_ids) < 2:
        st.info("Create at least 2 events to see correlation analysis.")
    else:
        # Ensure all events registered
        for eid in event_ids:
            corr.add_event(eid)

        # Custom correlation setter
        st.markdown("#### Set Correlation")
        c1, c2, c3 = st.columns(3)
        with c1: ev_a = st.selectbox("Event A", event_ids, key="corr_a")
        with c2: ev_b = st.selectbox("Event B", event_ids, key="corr_b")
        with c3: rho = st.slider("ρ", -0.99, 0.99, 0.0, 0.01)
        if st.button("Set Correlation"):
            corr.set_correlation(ev_a, ev_b, rho)
            st.success(f"ρ({ev_a}, {ev_b}) = {rho:.2f}")

        # Correlation matrix display
        if len(corr._events) >= 2:
            st.markdown("#### Correlation Matrix")
            corr_df = pd.DataFrame(
                corr._matrix,
                index=corr._events,
                columns=corr._events
            )
            st.dataframe(corr_df.round(2), use_container_width=True)

            # Shock propagation
            st.markdown("#### Shock Propagation")
            shock_source = st.selectbox("Source event", corr._events)
            shock_size = st.slider("Shock magnitude (pp)", -30, 30, 10)

            current_probs = {}
            for eid in corr._events:
                try:
                    o = get_oracle(eid)
                    out_tmp = o.compute()
                    current_probs[eid] = out_tmp.probability
                except: current_probs[eid] = 0.5

            propagated = corr.propagate_shock(shock_source, shock_size/100, current_probs)
            prop_df = pd.DataFrame([
                {"Event": k, "Current P": f"{current_probs.get(k,0.5):.1%}",
                 "Implied ΔP": f"{v*100:+.1f}pp",
                 "New P": f"{min(0.98,max(0.02,current_probs.get(k,0.5)+v)):.1%}"}
                for k, v in propagated.items()
            ])
            st.dataframe(prop_df, hide_index=True, use_container_width=True)

            # Monte Carlo joint simulation
            if st.button("Run Monte Carlo (10,000 paths)", type="primary"):
                daily_vols = {}
                for eid in corr._events:
                    try:
                        o = get_oracle(eid)
                        o_out = o.compute()
                        vd = compute_vol_surface(current_probs[eid], get_event(eid)["category"], o_out.state)
                        daily_vols[eid] = vd["daily_vol"]
                    except: daily_vols[eid] = 0.08

                with st.spinner("Simulating 10,000 correlated probability paths..."):
                    mc_results = corr.joint_simulation(current_probs, daily_vols, n_paths=10000, n_days=30)

                mc_df = pd.DataFrame([
                    {"Event": eid, "Mean": f"{r['mean']:.1%}", "Std": f"{r['std']:.1%}",
                     "P5": f"{r['p5']:.1%}", "P25": f"{r['p25']:.1%}",
                     "Median": f"{r['p50']:.1%}", "P75": f"{r['p75']:.1%}", "P95": f"{r['p95']:.1%}"}
                    for eid, r in mc_results.items()
                ])
                st.dataframe(mc_df, hide_index=True, use_container_width=True)

# ── TAB 5: Ingest ──────────────────────────────────────────────────────────
with tab5:
    st.markdown("### Manual Article Ingestion")
    with st.form("ingest_form"):
        c1, c2 = st.columns(2)
        with c1:
            art_id   = st.text_input("Article ID", value=f"art_{int(time.time())}")
            source   = st.text_input("Source Name")
            tier     = st.selectbox("Tier", [1,2,3,4,5,6], index=3,
                         format_func=lambda x:{1:"T1 — Court/Regulatory",2:"T2 — Local Press",
                         3:"T3 — Regional Intl",4:"T4 — Wire (Bloomberg/Reuters)",
                         5:"T5 — Capital Flow Signal",6:"T6 — Unverified"}[x])
            pub_date = st.date_input("Date", value=date.today())
            url      = st.text_input("URL")
        with c2:
            impact   = st.slider("Raw Impact", 1.0, 10.0, 5.0, 0.5)
            direction= st.radio("Direction",
                        [1,0,-1], format_func=lambda x:{1:"🔴 +1 Increases P",0:"⚪ 0 Neutral",-1:"🟢 -1 Decreases P"}[x])
            headline = st.text_input("Headline *")
            content  = st.text_area("Content Summary", height=80)
            reasoning= st.text_input("Reasoning")

        if st.form_submit_button("⬡ Ingest", type="primary"):
            if source and headline:
                pub_time = datetime.combine(pub_date, datetime.min.time()).timestamp()
                article = create_article_from_dict({
                    "article_id": art_id, "source_name": source, "tier": tier,
                    "publication_time": pub_time, "headline": headline,
                    "content_summary": content, "url": url,
                    "raw_impact": impact, "direction": direction, "reasoning_chain": reasoning,
                })
                oracle.ingest_articles([article], auto_score_independence=True)
                save_article(selected_event, oracle._articles[-1])

                out_new = oracle.compute()
                save_history_snapshot(selected_event, {
                    "timestamp": time.time(), "probability": round(out_new.probability,4),
                    "state": out_new.state.value, "fast_shock": round(out_new.fast_shock,4),
                    "persistent_signal": round(out_new.persistent_signal,4),
                    "independent_sources": out_new.independent_source_count,
                })
                inde = oracle._articles[-1].independence_score
                st.success(f"✓ ι={inde:.3f} · New P = {out_new.probability:.1%} [{out_new.state.value}]")
                st.rerun()
            else:
                st.error("Source and headline required")

# ── TAB 6: Live Feed ───────────────────────────────────────────────────────
with tab6:
    st.markdown("### Live Article Feed — GDELT + NewsAPI + GNews")

    ev_info2 = get_event(selected_event)
    default_query = build_query_from_event(ev_info2["description"])
    query = st.text_input("Search query", value=default_query)
    c1, c2, c3 = st.columns(3)
    with c1: hours_back = st.selectbox("Time window", [6,12,24,48,72], index=2)
    with c2: max_per = st.selectbox("Max per source", [5,10,15,20], index=1)
    with c3: st.markdown("<br>", unsafe_allow_html=True)
             
    if st.button("🔍 Fetch Live Articles", type="primary"):
        with st.spinner(f"Querying: {query}..."):
            articles = live_ingest(query, hours_back=hours_back, max_per_source=max_per)
            st.session_state.live_articles[selected_event] = articles

    live_arts = st.session_state.live_articles.get(selected_event, [])
    if live_arts:
        st.markdown(f"**{len(live_arts)} articles found** — Review and ingest:")
        for i, art in enumerate(live_arts):
            tier_colors = {1:"#f59e0b",2:"#34d399",3:"#60a5fa",4:"#9ca3af",5:"#a78bfa",6:"#6b7280"}
            tc = tier_colors.get(art.tier.value,"#9ca3af")
            with st.expander(f"T{art.tier.value} · {art.source_name} · {art.headline[:80]}"):
                st.markdown(f"**URL:** {art.url}")
                st.markdown(f"**Published:** {datetime.fromtimestamp(art.publication_time).strftime('%Y-%m-%d %H:%M')}")
                c1,c2,c3 = st.columns(3)
                with c1: impact_sel = st.slider("Impact", 1.0, 10.0, 5.0, 0.5, key=f"imp_{i}")
                with c2: dir_sel = st.radio("Direction", [1,0,-1],
                           format_func=lambda x:{1:"+1",0:"0",-1:"-1"}[x], key=f"dir_{i}")
                with c3: st.markdown(f"<br><span style='color:{tc}'>Tier {art.tier.value}</span>", unsafe_allow_html=True)

                if st.button(f"⬡ Ingest this article", key=f"ingest_{i}"):
                    art.raw_impact = impact_sel
                    art.direction = dir_sel
                    oracle.ingest_articles([art], auto_score_independence=True)
                    save_article(selected_event, oracle._articles[-1])
                    out_new = oracle.compute()
                    save_history_snapshot(selected_event, {
                        "timestamp": time.time(), "probability": round(out_new.probability,4),
                        "state": out_new.state.value, "fast_shock": round(out_new.fast_shock,4),
                        "persistent_signal": round(out_new.persistent_signal,4),
                        "independent_sources": out_new.independent_source_count,
                    })
                    st.success(f"✓ Ingested · New P = {out_new.probability:.1%}")
                    st.rerun()
    else:
        st.info("Click 'Fetch Live Articles' to pull from GDELT (free, no key needed) and NewsAPI/GNews (if keys set in Advanced settings).")

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#374151;font-size:12px;">
World-as-Oracle v2.0 · Bayesian Engine · Vol Surface · Greeks · Correlation · Live Feed<br>
Guruprasad Venkatakrishnan (2026) · Verslan · predictmarkets.finance · verslan.xyz
</div>
""", unsafe_allow_html=True)
