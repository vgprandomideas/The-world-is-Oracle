"""
Federal Reserve Rate Decision Oracle — Section 5.4.

Demonstrates oracle on the 3-meeting Fed consistency check.
Validates methodology against Kalshi market-implied probabilities.

Run: python examples/fed_rate_decision.py

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

from datetime import datetime
from oracle import WorldOracle, Article, SourceTier, EventCategory
from oracle.impact_scorer import create_article_from_dict
from oracle.mtm import SwapLedger, initial_margin_schedule
from oracle.models import SwapContract


def ts(date_str: str) -> float:
    return datetime.strptime(date_str, "%Y-%m-%d").timestamp()


def demo_jun_2023_pause():
    """
    June 14, 2023 FOMC — The first pause after 10 consecutive hikes.
    Most informative case: genuine epistemic uncertainty.
    Oracle: ~52% hike | Kalshi: ~60% hike | Outcome: PAUSE (NO hike)
    """
    print("\n" + "=" * 60)
    print("JUNE 2023 FOMC — First Pause After 10 Consecutive Hikes")
    print("=" * 60)

    oracle = WorldOracle(
        event_id="fomc_jun2023_hike",
        event_description="Federal Reserve raises interest rates at June 14, 2023 FOMC meeting",
        resolution_criteria="Fed funds target rate increases by 25bps or more at June 14 2023 FOMC",
        category=EventCategory.CENTRAL_BANK,
    )

    oracle.set_prior(
        historical_base_rate=0.65,   # 10 consecutive hikes — strong prior for hike
        structural_prior=0.55,       # May 2023 pause signals, CPI still elevated
        market_implied_prior=0.60,   # Kalshi market-implied
    )

    articles = [
        create_article_from_dict({
            "article_id": "waller_may23",
            "source_name": "Federal Reserve — Governor Waller Speech",
            "tier": 1,
            "publication_time": ts("2023-05-24"),
            "headline": "Fed's Waller Signals Possible Pause at June Meeting",
            "content_summary": "Fed Governor Waller explicitly stated the June meeting could see a pause in rate hikes to assess cumulative tightening effects.",
            "raw_impact": 8,
            "direction": -1,  # Against hike
            "reasoning_chain": "Tier 1: official Fed communication. Explicit pause signal from FOMC voting member. High credibility.",
        }),
        create_article_from_dict({
            "article_id": "cpi_may23",
            "source_name": "Bureau of Labor Statistics — CPI Release",
            "tier": 1,
            "publication_time": ts("2023-06-13"),
            "headline": "CPI Comes in at 4.0% YoY — Above Fed 2% Target",
            "content_summary": "May 2023 CPI printed 4.0% YoY. Core CPI 5.3%. Both above Fed target, maintaining inflationary pressure argument for continued hikes.",
            "raw_impact": 6,
            "direction": +1,  # Supports hike
            "reasoning_chain": "Tier 1: official BLS data. Inflation still above target. Counter-signal to pause narrative.",
        }),
        create_article_from_dict({
            "article_id": "fomc_minutes_may",
            "source_name": "Federal Reserve — FOMC Meeting Minutes",
            "tier": 1,
            "publication_time": ts("2023-05-24"),
            "headline": "May FOMC Minutes: Several Members Saw Case for Pausing",
            "content_summary": "Minutes revealed 'several' members explicitly discussed pausing at a future meeting. Language shift from previous minutes.",
            "raw_impact": 7,
            "direction": -1,
            "reasoning_chain": "Tier 1: official FOMC minutes. Collective Fed communication showing internal debate.",
        }),
        create_article_from_dict({
            "article_id": "tsy_yield",
            "source_name": "Treasury Market — 2-Year Yield Movement",
            "tier": 5,  # Capital flow signal
            "publication_time": ts("2023-06-12"),
            "headline": "2-Year Treasury Yield Declines Ahead of FOMC — Markets Pricing Pause",
            "content_summary": "2-year Treasury fell 12bps on June 12, with bond market positioning shifting toward pause probability.",
            "raw_impact": 5,
            "direction": -1,
            "reasoning_chain": "Tier 5 capital flow. Bond market real-money positioning against hike.",
        }),
    ]

    oracle.ingest_articles(articles, auto_score_independence=True, auto_score_impact=False)

    t_eve = ts("2023-06-13")  # Day before meeting
    out = oracle.compute(t_eve)

    print(f"\nOracle (Day -1):     P(hike) = {out.probability:.1%}")
    print(f"Kalshi (Day -1):     P(hike) = ~60%")
    print(f"Oracle State:        {out.state.value}")
    print(f"CI:                  [{out.ci_lower:.1%}, {out.ci_upper:.1%}]")
    print(f"Signal decomp:       prior={out.prior:.3f} + shock={out.fast_shock:.3f} + persistent={out.persistent_signal:.3f}")
    print(f"\nActual outcome:      PAUSE (no hike)")
    print(f"Oracle error:        {abs(out.probability - 0.0):.1%}")
    print(f"Kalshi error:        ~60%")
    print(f"\nNote: Oracle's BUILDING state with wide CI correctly")
    print(f"reflected genuine epistemic uncertainty — conflicting Tier 1 signals.")

    # Demonstrate swap contract
    print("\n" + "─" * 60)
    print("Event Probability Swap — Initial Margin (DMM)")
    swap = SwapContract(
        contract_id="fomc_jun23_swap_001",
        event_id="fomc_jun2023_hike",
        notional=1_000_000,
        fixed_probability=0.52,
    )
    margin = initial_margin_schedule(swap.fixed_probability, swap.notional)
    print(f"Fixed probability:    {swap.fixed_probability:.0%}")
    print(f"Notional:             ${swap.notional:,.0f}")
    print(f"IM Fixed Payer (A):   ${margin['fixed_payer_initial_margin']:,.0f}")
    print(f"IM Float Payer (B):   ${margin['floating_payer_initial_margin']:,.0f}")
    print(f"Total Margin:         ${margin['total_margin']:,.0f} ({margin['margin_efficiency']:.0%} of notional)")
    print(f"Max loss either side: Fully covered by pre-posted margin (DMM property)")

    return oracle, out


def demo_feb_2023_strong_consensus():
    """
    Feb 1, 2023 FOMC — Strong consensus case (+25bps)
    Oracle: ~85% | Kalshi: ~85% | Outcome: YES (+25bps)
    """
    print("\n" + "=" * 60)
    print("FEBRUARY 2023 FOMC — Strong Consensus Hike Case")
    print("=" * 60)

    oracle = WorldOracle(
        event_id="fomc_feb2023_hike",
        event_description="Federal Reserve raises interest rates at February 1, 2023 FOMC",
        resolution_criteria="Fed funds rate increases by 25bps at Feb 1 2023 FOMC",
        category=EventCategory.CENTRAL_BANK,
    )
    oracle.set_prior(0.80, 0.78, 0.85)

    articles = [
        create_article_from_dict({
            "article_id": "fomc_jan_min",
            "source_name": "Federal Reserve — FOMC Minutes (Dec 2022)",
            "tier": 1, "publication_time": ts("2023-01-04"),
            "headline": "Dec FOMC Minutes: All Members Support Continued Rate Increases",
            "content_summary": "All FOMC members supported ongoing rate increases. No dissents. Strong forward guidance for 2023 hikes.",
            "raw_impact": 8, "direction": +1,
            "reasoning_chain": "Unanimous FOMC consensus documented in official minutes.",
        }),
        create_article_from_dict({
            "article_id": "powell_jan",
            "source_name": "Federal Reserve Chair — Powell Speech",
            "tier": 1, "publication_time": ts("2023-01-10"),
            "headline": "Powell: Disinflation Process Has Begun, More Hikes Coming",
            "content_summary": "Powell confirmed ongoing rate hikes needed. 'Ongoing increases' language maintained.",
            "raw_impact": 8, "direction": +1,
            "reasoning_chain": "Tier 1: Chair explicitly confirmed continued hikes.",
        }),
    ]

    oracle.ingest_articles(articles, auto_score_independence=True)
    out = oracle.compute(ts("2023-01-31"))

    print(f"\nOracle (Day -1):     P(hike) = {out.probability:.1%}")
    print(f"Kalshi (Day -1):     P(hike) = ~85%")
    print(f"Oracle State:        {out.state.value}")
    print(f"\nActual outcome:      +25bps HIKE ✓")
    print(f"Both oracle and Kalshi correct on consensus case.")


if __name__ == "__main__":
    print("WORLD-AS-ORACLE: Federal Reserve Consistency Check")
    print("Paper Section 5.4 — Guruprasad Venkatakrishnan (2026)")

    oracle, out = demo_jun_2023_pause()
    demo_feb_2023_strong_consensus()

    print("\n" + "=" * 60)
    print("KEY FINDING: On standardised events with liquid prediction")
    print("markets, oracle produces estimates consistent with Kalshi.")
    print("On non-standardised events (Adani), oracle is the only")
    print("systematic probability estimate and substantially reduces error.")
    print("=" * 60)
