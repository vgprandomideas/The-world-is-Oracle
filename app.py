"""
World-as-Oracle research and risk workbench.
"""

import copy
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st

from db import (
    get_all_events,
    get_articles,
    get_event,
    get_history,
    init_db,
    save_article,
    save_event,
    save_history_snapshot,
)
from oracle import EventCategory, WorldOracle
from oracle.bayesian_engine import BayesianOracleEngine, likelihood_ratio
from oracle.car_presets import get_car_preset, list_car_presets
from oracle.correlation import CorrelationMatrix
from oracle.deterministic_car import run_deterministic_car
from oracle.impact_scorer import create_article_from_dict
from oracle.live_feed import build_query_from_event, live_ingest
from oracle.models import Article, SourceTier
from oracle.portfolio import PortfolioPosition, analyze_portfolio_risk, summarize_positions
from oracle.temporal_filter import count_confirming_signals, decay_factor
from oracle.vol_surface import compute_greeks, compute_vol_surface


st.set_page_config(
    page_title="World-as-Oracle | Advanced Workbench",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], .stMarkdown, .stText, label, button, input, textarea, select {
        font-family: 'Inter', sans-serif !important;
    }
    span.material-symbols-rounded, span.material-symbols-sharp, span.material-symbols-outlined {
        font-family: "Material Symbols Rounded" !important;
    }
    .stApp {
        background: #09111f;
        color: #e5edf8;
    }
    [data-testid="stSidebar"] {
        background: #0d1728 !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.75rem;
    }
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
    }
    [data-testid="stMetric"] {
        background: #0d1728;
        border: 1px solid #1b2c47;
        border-radius: 10px;
        padding: 10px 12px;
    }
    .sidebar-card {
        background: #111c2e;
        border: 1px solid #223654;
        border-radius: 10px;
        padding: 14px;
    }
    .sidebar-title {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #91a4c2;
        margin-bottom: 8px;
    }
    .hero {
        background: linear-gradient(135deg, #0c1728 0%, #111f36 100%);
        border: 1px solid #203553;
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }
    .hero-prob {
        font-size: 74px;
        line-height: 1;
        font-weight: 800;
        color: #f59e0b;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .panel {
        background: #0d1728;
        border: 1px solid #1b2c47;
        border-radius: 10px;
        padding: 16px;
    }
    .mono { font-family: 'JetBrains Mono', monospace !important; }
    .small { color: #91a4c2; font-size: 12px; }
    .state-pill {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }
    @media (max-width: 1100px) {
        .hero-prob { font-size: 58px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


init_db()

if "oracles" not in st.session_state:
    st.session_state.oracles = {}
if "bayesian" not in st.session_state:
    st.session_state.bayesian = {}
if "correlation" not in st.session_state:
    st.session_state.correlation = CorrelationMatrix()
if "live_articles" not in st.session_state:
    st.session_state.live_articles = {}
if "portfolio_positions" not in st.session_state:
    st.session_state.portfolio_positions = []
if "portfolio_result" not in st.session_state:
    st.session_state.portfolio_result = None


STATE_COLORS = {
    "BASELINE": "#6b7280",
    "SHOCK_ACTIVE": "#f59e0b",
    "BUILDING": "#22c55e",
    "SUSTAINED": "#14b8a6",
    "CONTESTED": "#ef4444",
    "INSUFFICIENT_SIGNAL": "#64748b",
    "ORACLE_DEGRADED": "#a855f7",
}


def get_oracle(event_id):
    if event_id not in st.session_state.oracles:
        event = get_event(event_id)
        if not event:
            st.error(f"Event {event_id} not found")
            st.stop()

        category_map = {category.value: category for category in EventCategory}
        oracle = WorldOracle(
            event_id=event_id,
            event_description=event["description"],
            resolution_criteria=event["resolution"],
            category=category_map.get(event["category"], EventCategory.CORPORATE_LEGAL),
            platt_a=event.get("platt_a", 1.0),
            platt_b=event.get("platt_b", 0.0),
        )
        oracle._prior = event["prior"]

        tier_map = {index: SourceTier(index) for index in range(1, 7)}
        for row in get_articles(event_id):
            article = Article(
                article_id=row["article_id"],
                source_name=row["source_name"],
                tier=tier_map.get(row["tier"], SourceTier.TIER4_WIRE),
                publication_time=row["publication_time"],
                headline=row["headline"],
                content_summary=row.get("content_summary", ""),
                url=row.get("url", ""),
                raw_impact=row.get("raw_impact", 0.0),
                direction=row.get("direction", 0),
                independence_score=row.get("independence_score", 0.0),
                reasoning_chain=row.get("reasoning_chain", ""),
            )
            oracle._articles.append(article)

        st.session_state.oracles[event_id] = oracle
    return st.session_state.oracles[event_id]


def get_bayesian(event_id, oracle):
    if event_id not in st.session_state.bayesian:
        st.session_state.bayesian[event_id] = BayesianOracleEngine(oracle._prior, oracle.config)
    return st.session_state.bayesian[event_id]


def save_snapshot(event_id, oracle_output):
    save_history_snapshot(
        event_id,
        {
            "timestamp": time.time(),
            "probability": round(oracle_output.probability, 4),
            "state": oracle_output.state.value,
            "fast_shock": round(oracle_output.fast_shock, 4),
            "persistent_signal": round(oracle_output.persistent_signal, 4),
            "independent_sources": oracle_output.independent_source_count,
        },
    )


def create_event_from_inputs(event_id, description, resolution, category, hist_base, structural, market_prior):
    category_map = {item.value: item for item in EventCategory}
    oracle = WorldOracle(
        event_id=event_id,
        event_description=description,
        resolution_criteria=resolution,
        category=category_map.get(category, EventCategory.CORPORATE_LEGAL),
    )
    oracle.set_prior(hist_base, structural, market_prior if market_prior and market_prior > 0 else None)
    save_event(
        event_id,
        description,
        resolution,
        category,
        oracle._prior,
        hist_base,
        structural,
        market_prior if market_prior and market_prior > 0 else None,
    )
    st.session_state.oracles[event_id] = oracle
    st.session_state.correlation.add_event(event_id)
    st.session_state.selected_event_override = event_id


def create_event_from_preset(preset_key):
    preset = get_car_preset(preset_key)
    seed = preset["seed_event"]
    create_event_from_inputs(
        seed["event_id"],
        seed["description"],
        seed["resolution"],
        seed["category"],
        seed["historical_base_rate"],
        seed["structural_prior"],
        seed["market_implied_prior"],
    )


def render_new_event_form(key_suffix):
    with st.form(f"new_event_{key_suffix}"):
        event_id = st.text_input("Event ID", key=f"event_id_{key_suffix}")
        description = st.text_area("Description", height=80, key=f"description_{key_suffix}")
        resolution = st.text_area("Resolution Criteria", height=80, key=f"resolution_{key_suffix}")
        category = st.selectbox(
            "Category",
            [
                "central_bank",
                "corporate_legal",
                "geopolitical",
                "electoral",
                "macro_data",
                "sovereign_credit",
                "crypto_protocol",
            ],
            key=f"category_{key_suffix}",
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            hist_base = st.number_input("Base Rate", 0.0, 1.0, 0.15, 0.01, key=f"hist_base_{key_suffix}")
        with col_b:
            structural = st.number_input("Structural", 0.0, 1.0, 0.20, 0.01, key=f"structural_{key_suffix}")
        with col_c:
            market_prior = st.number_input("Market Prior", 0.0, 1.0, 0.0, 0.01, key=f"market_prior_{key_suffix}")

        submitted = st.form_submit_button("Create Event", type="primary", use_container_width=True)
        if submitted:
            if event_id and description:
                create_event_from_inputs(event_id, description, resolution, category, hist_base, structural, market_prior)
                st.success(f"Created {event_id}")
                st.rerun()
            else:
                st.error("Event ID and description are required.")


def render_preset_loader(key_suffix):
    presets = list_car_presets()
    preset_keys = [item["key"] for item in presets]
    preset_key = st.selectbox(
        "Preset Case",
        preset_keys,
        format_func=lambda key: next(item["title"] for item in presets if item["key"] == key),
        key=f"preset_key_{key_suffix}",
    )
    preset = get_car_preset(preset_key)
    stats = st.columns(3)
    stats[0].metric("Naive P", f"{preset['naive_probability']:.0%}")
    stats[1].metric("Actors", f"{len(preset['actors'])}")
    stats[2].metric("Category", preset["seed_event"]["category"].replace("_", " ").title())
    st.caption(preset["question"])
    st.write(preset["context"])
    if st.button("Load Preset Event", use_container_width=True, key=f"load_preset_{key_suffix}"):
        create_event_from_preset(preset_key)
        st.success(f"Loaded {preset['title']}")
        st.rerun()


def current_probability_map(event_ids):
    values = {}
    for event_id in event_ids:
        try:
            values[event_id] = get_oracle(event_id).compute().probability
        except Exception:
            values[event_id] = 0.5
    return values


def current_daily_vols(event_ids, probability_map):
    vol_map = {}
    for event_id in event_ids:
        event = get_event(event_id)
        oracle = get_oracle(event_id)
        output = oracle.compute()
        vol_map[event_id] = compute_vol_surface(
            p=probability_map.get(event_id, 0.5),
            category=event["category"],
            state=output.state,
            n_signals_24h=len(oracle._articles),
        )["daily_vol"]
    return vol_map


with st.sidebar:
    st.markdown("### predictmarkets.finance")
    st.caption("World-as-Oracle research and derivatives desk")
    st.divider()

    events = get_all_events()
    event_ids = [event["event_id"] for event in events]
    pending_event = st.session_state.pop("selected_event_override", None)
    default_index = event_ids.index(pending_event) if pending_event in event_ids else 0
    selected_event = st.selectbox("Active Event", event_ids if event_ids else ["--"], index=default_index if event_ids else 0)

    if st.button("Live Ingest", use_container_width=True, type="primary"):
        if selected_event and selected_event != "--":
            oracle = get_oracle(selected_event)
            event = get_event(selected_event)
            query = build_query_from_event(event["description"])
            with st.spinner(f"Fetching live articles for: {query}"):
                articles = live_ingest(query, hours_back=48, max_per_source=15)
            if articles:
                oracle.ingest_articles(articles, auto_score_independence=True)
                st.session_state.live_articles[selected_event] = articles
                st.success(f"Ingested {len(articles)} live articles")
            else:
                st.warning("No articles found.")

    if selected_event and selected_event != "--":
        current_event = get_event(selected_event)
        st.markdown(
            f"""
            <div class="sidebar-card">
              <div class="sidebar-title">Current Event</div>
              <div style="font-size:14px;font-weight:600;margin-bottom:4px;">{selected_event}</div>
              <div class="small" style="line-height:1.5;">{current_event['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Create or load an event from the Setup workspace.")

    st.markdown('<div class="sidebar-title">Desk Settings</div>', unsafe_allow_html=True)
    desk_notional = st.number_input("Notional ($)", 100000, 100000000, 1000000, 100000)
    desk_fixed_prob = st.slider("Fixed Probability (K)", 0.05, 0.95, 0.50, 0.01)
    desk_days = st.number_input("Days to Resolution", 1, 365, 90)


if not selected_event or selected_event == "--":
    st.markdown(
        """
        <div class="hero">
          <div style="font-size:14px;color:#91a4c2;margin-bottom:8px;">Research and Risk Workbench</div>
          <div style="font-size:34px;font-weight:700;margin-bottom:8px;">World-as-Oracle</div>
          <div style="max-width:760px;color:#b7c6dc;">
            Create an event or load a paper-backed preset to explore the core oracle,
            deterministic CAR analysis, and portfolio risk tooling in one place.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    landing_left, landing_right = st.columns([1.15, 1.0])
    with landing_left:
        st.markdown("### Create a new event")
        render_new_event_form("landing")
    with landing_right:
        st.markdown("### Or start from a paper-backed preset")
        render_preset_loader("landing")
    st.stop()


oracle = get_oracle(selected_event)
event_info = get_event(selected_event)
bayes = get_bayesian(selected_event, oracle)
now = time.time()

output = oracle.compute(now)
bayes_result = bayes.compute(oracle._articles, now, use_bootstrap=len(oracle._articles) >= 3)
display_probability = bayes_result["posterior"] if bayes_result["n_signals"] >= 2 else output.probability

history = get_history(selected_event)
posterior_hist = [row["probability"] for row in history[-30:]]
vol_data = compute_vol_surface(
    p=display_probability,
    category=event_info["category"],
    state=output.state,
    days_to_resolution=desk_days,
    n_signals_24h=bayes_result["n_signals"],
    posterior_history=posterior_hist,
)
greeks = compute_greeks(
    p=display_probability,
    notional=desk_notional,
    daily_vol=vol_data["daily_vol"],
    days_to_resolution=desk_days,
    fixed_probability=desk_fixed_prob,
)

state_color = STATE_COLORS.get(output.state.value, "#94a3b8")

header_left, header_right = st.columns([1.0, 1.7])
with header_left:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-prob">{display_probability:.0%}</div>
          <div class="small" style="margin-top:8px;">Probability of YES resolution</div>
          <div style="margin-top:12px;">
            <span class="state-pill" style="background:{state_color}22;color:{state_color};border:1px solid {state_color}55;">
              {output.state.value}
            </span>
          </div>
          <div class="small" style="margin-top:12px;">
            Bayesian CI [{bayes_result['ci_5']:.1%}, {bayes_result['ci_95']:.1%}]<br/>
            Daily vol {vol_data['daily_vol']:.1%} | {vol_data['regime']}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(f"### {event_info['description']}")
    st.caption(event_info["resolution"])
    row_a = st.columns(3)
    row_a[0].metric("Bayesian P", f"{bayes_result['posterior']:.1%}", delta=f"{(bayes_result['posterior'] - oracle._prior) * 100:+.1f}pp")
    row_a[1].metric("Oracle P", f"{output.probability:.1%}", delta=f"{(output.probability - bayes_result['posterior']) * 100:+.1f}pp")
    row_a[2].metric("Signals", f"{bayes_result['n_signals']}")
    row_b = st.columns(2)
    row_b[0].metric("Epistemic Sigma", f"{bayes_result['std']:.1%}")
    row_b[1].metric("Annual Vol", f"{vol_data['annual_vol']:.0%}", delta=vol_data["regime"])


top_tabs = st.tabs(["Setup", "Overview", "Analysis", "Trading", "Data"])


with top_tabs[0]:
    setup_left, setup_right = st.columns([1.1, 1.0])
    with setup_left:
        st.markdown("### New event")
        st.caption("Create a custom event without fighting the sidebar.")
        render_new_event_form("setup")
    with setup_right:
        st.markdown("### Research presets")
        st.caption("Load a validated case and start from a usable baseline.")
        render_preset_loader("setup")


with top_tabs[1]:
    col_a, col_b, col_c = st.columns([1.1, 1.0, 1.0])

    with col_a:
        st.markdown("#### Signal decomposition")
        st.metric("Prior", f"{oracle._prior:.1%}")
        st.metric("Fast Shock", f"{output.fast_shock:+.3f}")
        st.metric("Persistent Signal", f"{output.persistent_signal:+.3f}")
        st.metric("Independent Sources", f"{output.independent_source_count}")

    with col_b:
        st.markdown("#### Strongest articles")
        ranked = sorted(oracle._articles, key=lambda item: abs(item.signed_impact), reverse=True)[:6]
        if ranked:
            for article in ranked:
                age_days = max(0.0, (now - article.publication_time) / 86400)
                confirmations = count_confirming_signals(article, oracle._articles, oracle.config.theta_ind, oracle.config.theta_imp)
                decay = decay_factor(age_days, oracle.config.decay_rate_lambda, confirmations)
                lr = likelihood_ratio(article, decay)
                st.markdown(
                    f"""
                    <div class="panel" style="margin-bottom:8px;">
                      <div style="font-size:13px;font-weight:600;">{article.source_name} | Tier {article.tier.value}</div>
                      <div class="small">{article.headline[:90]}</div>
                      <div class="small mono" style="margin-top:6px;">impact={article.signed_impact:+.2f} | lr={lr:.2f}x | ind={article.independence_score:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No articles ingested yet.")

    with col_c:
        st.markdown("#### Audit trail")
        if output.audit_trail:
            audit_df = pd.DataFrame(output.audit_trail)
            st.dataframe(audit_df, hide_index=True, use_container_width=True)
        else:
            st.info("Audit trail will populate after qualifying articles are ingested.")

    if len(history) >= 2:
        st.markdown("#### Probability history")
        hist_df = pd.DataFrame(history)
        hist_df["datetime"] = pd.to_datetime(hist_df["timestamp"], unit="s")
        st.line_chart(hist_df.set_index("datetime")[["probability"]], use_container_width=True, height=240)


with top_tabs[2]:
    analysis_tabs = st.tabs(["CAR Lab", "Correlation"])

    with analysis_tabs[0]:
        presets = list_car_presets()
        preset_key = next((item["key"] for item in presets if item["event_id"] == selected_event), presets[0]["key"])
        chosen_key = st.selectbox(
            "Case Study",
            [item["key"] for item in presets],
            index=[item["key"] for item in presets].index(preset_key),
            format_func=lambda key: next(item["title"] for item in presets if item["key"] == key),
        )
        car_preset = get_car_preset(chosen_key)

        car_left, car_right = st.columns([1.3, 1.0])
        with car_left:
            question = st.text_input("Question", value=car_preset["question"])
            use_oracle_probability = st.checkbox(
                "Start from the current oracle probability",
                value=(selected_event == car_preset["seed_event"]["event_id"]),
            )
            default_naive = display_probability if use_oracle_probability else car_preset["naive_probability"]
            naive_probability = st.slider("Naive probability", 0.02, 0.98, float(default_naive), 0.01)
            st.caption(car_preset["context"])

        actor_profiles = copy.deepcopy(car_preset["actors"])
        with car_right:
            tuned_actor_name = st.selectbox("Tune actor", [actor["actor"] for actor in actor_profiles])
            tuned_actor = next(actor for actor in actor_profiles if actor["actor"] == tuned_actor_name)
            tuned_actor["utility_yes"] = st.slider("Utility if YES", 0.0, 10.0, float(tuned_actor["utility_yes"]), 0.1)
            tuned_actor["utility_no"] = st.slider("Utility if NO", 0.0, 10.0, float(tuned_actor["utility_no"]), 0.1)
            tuned_actor["blocking_power"] = st.slider("Blocking power", 0.0, 1.0, float(tuned_actor["blocking_power"]), 0.01)
            tuned_actor["structural_silence_score"] = st.slider(
                "Structural silence",
                0.0,
                5.0,
                float(tuned_actor.get("structural_silence_score", 0.0)),
                0.01,
            )

        car_result = run_deterministic_car(question, naive_probability, actor_profiles)
        car_metrics = st.columns(4)
        car_metrics[0].metric("Naive P", f"{car_result['final']['p_naive']:.1%}")
        car_metrics[1].metric("CAR P", f"{car_result['final']['p_car']:.1%}", delta=f"{(car_result['final']['p_car'] - car_result['final']['p_naive']) * 100:+.1f}pp")
        car_metrics[2].metric("Haircut", f"{car_result['final']['adversarial_haircut']:.1%}")
        car_metrics[3].metric("Equilibrium", car_result["final"]["equilibrium_type"])

        coalition_left, coalition_right = st.columns([1.1, 1.2])
        with coalition_left:
            actor_df = pd.DataFrame(car_result["actor_profiles"])[
                ["actor", "actor_class", "utility_yes", "utility_no", "blocking_power", "structural_silence_score"]
            ]
            st.markdown("#### Actor map")
            st.dataframe(actor_df, hide_index=True, use_container_width=True)

        with coalition_right:
            coalition_df = pd.DataFrame(
                [
                    {
                        "Coalition": "Blocking",
                        "Actors": ", ".join(car_result["game_model"]["blocking_coalition"]["members"]) or "--",
                        "Power": car_result["game_model"]["blocking_coalition"]["combined_blocking_power"],
                    },
                    {
                        "Coalition": "Resolving",
                        "Actors": ", ".join(car_result["game_model"]["resolving_coalition"]["members"]) or "--",
                        "Power": car_result["game_model"]["resolving_coalition"]["combined_resolving_power"],
                    },
                ]
            )
            st.markdown("#### Coalition balance")
            st.dataframe(coalition_df, hide_index=True, use_container_width=True)
            st.markdown(
                f"""
                <div class="panel">
                  <div style="font-size:13px;font-weight:600;">Dominant scenario</div>
                  <div style="margin-top:6px;">{car_result['final']['dominant_scenario']}</div>
                  <div class="small" style="margin-top:8px;">{car_result['game_model']['key_insight']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Marginal drag")
        st.dataframe(pd.DataFrame(car_result["marginal_impacts"]), hide_index=True, use_container_width=True)

        if st.button("Load this case as active event"):
            create_event_from_preset(chosen_key)
            st.success(f"Loaded {car_preset['title']} into the event book")
            st.rerun()

    with analysis_tabs[1]:
        corr = st.session_state.correlation
        for event_id in event_ids:
            corr.add_event(event_id)

        if len(event_ids) < 2:
            st.info("Create at least two events to unlock cross-event correlation and shock propagation.")
        else:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                corr_a = st.selectbox("Event A", event_ids, key="corr_a")
            with col_b:
                corr_b = st.selectbox("Event B", event_ids, key="corr_b")
            with col_c:
                rho = st.slider("Correlation", -0.99, 0.99, 0.0, 0.01)
            if st.button("Set correlation"):
                corr.set_correlation(corr_a, corr_b, rho)
                st.success(f"Set rho({corr_a}, {corr_b}) = {rho:.2f}")

            corr_df = pd.DataFrame(corr._matrix, index=corr._events, columns=corr._events)
            st.dataframe(corr_df.round(2), use_container_width=True)

            prob_map = current_probability_map(corr._events)
            source_event = st.selectbox("Shock source", corr._events)
            shock_pp = st.slider("Shock magnitude (pp)", -30, 30, 10)
            shock_result = corr.propagate_shock(source_event, shock_pp / 100.0, prob_map)
            shock_rows = []
            for event_id, delta_p in shock_result.items():
                base_p = prob_map.get(event_id, 0.5)
                shock_rows.append(
                    {
                        "Event": event_id,
                        "Current P": f"{base_p:.1%}",
                        "Implied Delta": f"{delta_p * 100:+.1f}pp",
                        "New P": f"{min(0.98, max(0.02, base_p + delta_p)):.1%}",
                    }
                )
            st.dataframe(pd.DataFrame(shock_rows), hide_index=True, use_container_width=True)


with top_tabs[3]:
    trading_tabs = st.tabs(["Pricing", "Vol Surface", "Portfolio"])

    with trading_tabs[0]:
        st.markdown(f"### Event probability swap | Notional ${desk_notional:,.0f} | K = {desk_fixed_prob:.0%}")
        greek_cols = st.columns(4)
        greek_cols[0].metric("Swap Delta", f"${greeks['swap_delta']:,.0f}")
        greek_cols[1].metric("Binary Delta", f"{greeks['binary_delta']:.1f}")
        greek_cols[2].metric("Gamma", f"{greeks['gamma']:.4f}")
        greek_cols[3].metric("Theta", f"${abs(greeks['theta_daily']):,.0f}")

        vega_cols = st.columns(3)
        vega_cols[0].metric("Vega / 1% vol", f"${abs(greeks['vega_per_1pct_vol']):,.0f}")
        vega_cols[1].metric("Binary option price", f"${greeks['binary_option_price']:,.0f}")
        vega_cols[2].metric("d1", f"{greeks['d1']:.3f}")

        scenario_rows = []
        for scenario_p in np.arange(0.10, 0.91, 0.10):
            pnl = (scenario_p - display_probability) * desk_notional
            scenario_rows.append(
                {
                    "Scenario P": f"{scenario_p:.0%}",
                    "Delta vs now": f"{(scenario_p - display_probability) * 100:+.0f}pp",
                    "Long floating PnL": f"${pnl:+,.0f}",
                    "Short floating PnL": f"${-pnl:+,.0f}",
                }
            )
        st.dataframe(pd.DataFrame(scenario_rows), hide_index=True, use_container_width=True)

    with trading_tabs[1]:
        top = st.columns(4)
        top[0].metric("Daily Vol", f"{vol_data['daily_vol']:.1%}")
        top[1].metric("Annual Vol", f"{vol_data['annual_vol']:.0%}")
        top[2].metric("1d range", f"[{vol_data['vol_1d_range'][0]:.1%}, {vol_data['vol_1d_range'][1]:.1%}]")
        top[3].metric("7d range", f"[{vol_data['vol_7d_range'][0]:.1%}, {vol_data['vol_7d_range'][1]:.1%}]")

        smile_rows = []
        for level in np.arange(0.05, 0.96, 0.05):
            result = compute_vol_surface(level, event_info["category"], output.state, desk_days, bayes_result["n_signals"])
            smile_rows.append({"P": level, "Annual Vol": result["annual_vol"]})
        st.markdown("#### Smile")
        st.line_chart(pd.DataFrame(smile_rows).set_index("P"), use_container_width=True, height=220)

        term_rows = []
        for days in [1, 3, 7, 14, 30, 60, 90, 180, 365]:
            result = compute_vol_surface(display_probability, event_info["category"], output.state, days, bayes_result["n_signals"])
            term_rows.append({"Days": days, "Daily Vol": result["daily_vol"]})
        st.markdown("#### Term structure")
        st.line_chart(pd.DataFrame(term_rows).set_index("Days"), use_container_width=True, height=220)

        component_rows = [{"Component": key, "Value": value} for key, value in vol_data["components"].items()]
        st.dataframe(pd.DataFrame(component_rows), hide_index=True, use_container_width=True)

    with trading_tabs[2]:
        st.markdown("### Portfolio risk")
        if event_ids:
            with st.form("portfolio_add"):
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    portfolio_event = st.selectbox("Event", event_ids, key="portfolio_event")
                with col_b:
                    direction = st.selectbox("Direction", ["Long floating", "Short floating"])
                with col_c:
                    notional = st.number_input("Notional", 100000, 100000000, 1000000, 100000, key="portfolio_notional")
                with col_d:
                    fixed_probability = st.slider("Fixed P", 0.02, 0.98, float(display_probability), 0.01, key="portfolio_fixed")
                submitted = st.form_submit_button("Add position", type="primary")
                if submitted:
                    position = PortfolioPosition(
                        position_id=f"pos_{int(time.time())}",
                        event_id=portfolio_event,
                        notional=notional,
                        fixed_probability=fixed_probability,
                        long_floating=(direction == "Long floating"),
                        label=f"{portfolio_event}:{direction}",
                    )
                    st.session_state.portfolio_positions.append(position)
                    st.success("Position added")
                    st.rerun()

        action_cols = st.columns(3)
        if action_cols[0].button("Run portfolio risk", use_container_width=True):
            portfolio_events = list({position.event_id for position in st.session_state.portfolio_positions})
            prob_map = current_probability_map(portfolio_events)
            vol_map = current_daily_vols(portfolio_events, prob_map) if portfolio_events else {}
            st.session_state.portfolio_result = analyze_portfolio_risk(
                positions=st.session_state.portfolio_positions,
                current_probabilities=prob_map,
                daily_vols=vol_map,
                correlation_matrix=st.session_state.correlation,
                n_paths=5000,
                n_days=30,
            )
        if action_cols[1].button("Remove last", use_container_width=True) and st.session_state.portfolio_positions:
            st.session_state.portfolio_positions.pop()
            st.session_state.portfolio_result = None
            st.rerun()
        if action_cols[2].button("Clear book", use_container_width=True):
            st.session_state.portfolio_positions = []
            st.session_state.portfolio_result = None
            st.rerun()

        if st.session_state.portfolio_positions:
            summary = summarize_positions(
                st.session_state.portfolio_positions,
                current_probability_map(list({position.event_id for position in st.session_state.portfolio_positions})),
            )
            st.dataframe(pd.DataFrame(summary["positions"]), hide_index=True, use_container_width=True)
            totals = summary["totals"]
            total_cols = st.columns(3)
            total_cols[0].metric("Current MTM", f"${totals['current_mtm']:,.0f}")
            total_cols[1].metric("Gross notional", f"${totals['gross_notional']:,.0f}")
            total_cols[2].metric("Posted margin", f"${totals['margin']:,.0f}")

        if st.session_state.portfolio_result and st.session_state.portfolio_result["risk"]:
            risk = st.session_state.portfolio_result["risk"]
            risk_cols = st.columns(4)
            risk_cols[0].metric("VaR 95%", f"${risk['var_95']:,.0f}")
            risk_cols[1].metric("ES 95%", f"${risk['expected_shortfall_95']:,.0f}")
            risk_cols[2].metric("Worst case", f"${risk['worst_case']:,.0f}")
            risk_cols[3].metric("Best case", f"${risk['best_case']:,.0f}")

            hist_df = pd.DataFrame(st.session_state.portfolio_result["histogram"])
            if not hist_df.empty:
                hist_df["bucket"] = hist_df.apply(lambda row: f"{row['bin_start']:.0f} to {row['bin_end']:.0f}", axis=1)
                st.bar_chart(hist_df.set_index("bucket")["count"], use_container_width=True, height=220)

            st.markdown("#### Tail contributors")
            st.dataframe(pd.DataFrame(st.session_state.portfolio_result["tail_contributors"]), hide_index=True, use_container_width=True)


with top_tabs[4]:
    data_tabs = st.tabs(["Manual Ingest", "Live Feed"])

    with data_tabs[0]:
        st.markdown("### Manual article ingestion")
        with st.form("manual_ingest"):
            left, right = st.columns(2)
            with left:
                article_id = st.text_input("Article ID", value=f"art_{int(time.time())}")
                source_name = st.text_input("Source Name")
                tier = st.selectbox("Tier", [1, 2, 3, 4, 5, 6], index=3)
                publication_date = st.date_input("Date", value=date.today())
                url = st.text_input("URL")
            with right:
                impact = st.slider("Raw Impact", 1.0, 10.0, 5.0, 0.5)
                direction = st.radio("Direction", [1, 0, -1], format_func=lambda value: {1: "+1 increases", 0: "0 neutral", -1: "-1 decreases"}[value])
                headline = st.text_input("Headline")
                content = st.text_area("Content Summary", height=90)
                reasoning = st.text_input("Reasoning")

            if st.form_submit_button("Ingest", type="primary"):
                if source_name and headline:
                    article = create_article_from_dict(
                        {
                            "article_id": article_id,
                            "source_name": source_name,
                            "tier": tier,
                            "publication_time": datetime.combine(publication_date, datetime.min.time()).timestamp(),
                            "headline": headline,
                            "content_summary": content,
                            "url": url,
                            "raw_impact": impact,
                            "direction": direction,
                            "reasoning_chain": reasoning,
                        }
                    )
                    oracle.ingest_articles([article], auto_score_independence=True)
                    save_article(selected_event, oracle._articles[-1])
                    new_output = oracle.compute()
                    save_snapshot(selected_event, new_output)
                    st.success(f"Ingested article. New P = {new_output.probability:.1%}")
                    st.rerun()
                else:
                    st.error("Source name and headline are required.")

    with data_tabs[1]:
        st.markdown("### Live article feed")
        default_query = build_query_from_event(event_info["description"])
        query = st.text_input("Search query", value=default_query)
        col_a, col_b = st.columns(2)
        with col_a:
            hours_back = st.selectbox("Time window", [6, 12, 24, 48, 72], index=2)
        with col_b:
            max_per_source = st.selectbox("Max per source", [5, 10, 15, 20], index=1)

        if st.button("Fetch live articles", type="primary"):
            with st.spinner(f"Querying {query}"):
                st.session_state.live_articles[selected_event] = live_ingest(query, hours_back=hours_back, max_per_source=max_per_source)

        live_articles = st.session_state.live_articles.get(selected_event, [])
        if live_articles:
            st.markdown(f"**{len(live_articles)} articles found**")
            for index, article in enumerate(live_articles):
                with st.expander(f"Tier {article.tier.value} | {article.source_name} | {article.headline[:90]}"):
                    st.write(article.url)
                    st.write(datetime.fromtimestamp(article.publication_time).strftime("%Y-%m-%d %H:%M"))
                    adj_left, adj_right = st.columns(2)
                    with adj_left:
                        impact = st.slider("Impact", 1.0, 10.0, 5.0, 0.5, key=f"live_impact_{index}")
                    with adj_right:
                        direction = st.radio("Direction", [1, 0, -1], format_func=lambda value: {1: "+1", 0: "0", -1: "-1"}[value], key=f"live_direction_{index}")

                    if st.button("Ingest article", key=f"live_ingest_{index}"):
                        article.raw_impact = impact
                        article.direction = direction
                        oracle.ingest_articles([article], auto_score_independence=True)
                        save_article(selected_event, oracle._articles[-1])
                        new_output = oracle.compute()
                        save_snapshot(selected_event, new_output)
                        st.success(f"Ingested. New P = {new_output.probability:.1%}")
                        st.rerun()
        else:
            st.info("Fetch articles to review them here before ingesting.")


st.divider()
st.caption("World-as-Oracle advanced workbench | Bayesian engine | deterministic CAR | portfolio risk")
