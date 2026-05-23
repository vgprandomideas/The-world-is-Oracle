# World-as-Oracle

**Adversarial-Resistant AI Probability Estimation for Event-Driven Financial Instruments**

> *"The world is the oracle. Events speak for themselves — if you know how to listen across the right time horizon with the right source verification."*

**Paper:** Guruprasad Venkatakrishnan (2026) — *The World as Oracle: Adversarial-Resistant AI Probability Estimation for Event-Driven Financial Instruments*

**Affiliation:** Verslan | [predictmarkets.finance](https://predictmarkets.finance) | [verslan.xyz](https://verslan.xyz)

---

## What This Is

A Python implementation of the 7-layer oracle architecture described in the paper. The oracle estimates the probability of real-world binary events (regulatory actions, court outcomes, central bank decisions, geopolitical developments) by reading the world's information flows directly — rather than aggregating market prices or expert opinions.

The core mechanism is **temporal persistence filtering**: distinguishing genuine sustained information flows from manufactured narrative bursts.

## Architecture

```
Layer 1 — Data Ingestion        Continuous multi-source ingestion (6 tiers)
Layer 2 — Event Detection       Independence scoring: ι(a_i) = 1 - max sim(a_i, a_j)
Layer 3 — Impact Scoring        LLM chain-of-thought: magnitude + direction
Layer 4 — Temporal Filter       Fast shock (20%, exponential decay) + Persistent signal (80%, threshold-gated)
Layer 5 — Probability Engine    P_raw = P₀ + F(t) + P(t) → Platt scaling → [0.02, 0.98]
Layer 6 — Calibration           Platt scaling + isotonic regression + Brier score tracking
Layer 7 — Oracle Output         Probability + CI + state flag + full audit trail
```

## The Temporal Persistence Filter (Section 3)

```
Fast Shock:  F(t) = min( Σ sᵢ · δ(t, tᵢ), F_max )
             δ(t, tᵢ) = exp( -λ · Δt · (1 - γc · min(nc, 1)) )

Persistent:  P(t) = 0                               if ρ(t) < ρ_min
             P(t) = min( Σ sᵢ · ω(t - tᵢ), P_max ) if ρ(t) ≥ ρ_min

Composition: P_raw = P₀(E) + F(t) + P(t)
             P_oracle = σ( a · logit(clamp(P_raw)) + b )  constrained to [0.02, 0.98]
```

## Oracle States

| State | Condition |
|-------|-----------|
| `BASELINE` | ρ(t) < ρ_min AND F(t) < 0.05 |
| `SHOCK_ACTIVE` | F(t) ≥ 0.05 AND ρ(t) < ρ_min |
| `BUILDING` | ρ(t) ≥ ρ_min AND P(t) < 0.40 |
| `SUSTAINED` | P(t) ≥ 0.40 for ≥ 14 days |
| `CONTESTED` | \|P_oracle - P_tier1\| > 0.30 for ≥ 3 days |

`CONTESTED` is the paper's distinctive contribution: when primary source evidence and media narrative diverge significantly, the oracle surfaces information warfare risk rather than forcing a probability output.

## Installation

```bash
git clone https://github.com/guruprasad-venkatakrishnan/world-as-oracle.git
cd world-as-oracle
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Quick Start

```python
from oracle import WorldOracle, Article, SourceTier, EventCategory
from oracle.impact_scorer import create_article_from_dict

# Create oracle for a regulatory event
oracle = WorldOracle(
    event_id="fed_rate_jun2023",
    event_description="Federal Reserve raises rates at June 14, 2023 FOMC",
    resolution_criteria="Fed funds rate increases by 25bps or more",
    category=EventCategory.CENTRAL_BANK,
)

# Set prior: P₀ = α·H(E) + β·S(E) + γ·M(E)
oracle.set_prior(
    historical_base_rate=0.65,
    structural_prior=0.60,
    market_implied_prior=0.72,
)

# Ingest articles (pre-scored or auto-scored via Claude)
articles = [
    create_article_from_dict({
        "article_id": "fomc_minutes",
        "source_name": "Federal Reserve — FOMC Minutes",
        "tier": 1,
        "publication_time": 1684886400.0,
        "headline": "Several FOMC members see case for pausing",
        "content_summary": "Minutes show internal debate about pausing hikes",
        "raw_impact": 7,
        "direction": -1,
    })
]
oracle.ingest_articles(articles, auto_score_independence=True)

# Compute
import time
output = oracle.compute(current_time=time.time())
print(output)
# Oracle [BUILDING] P=0.523 CI=[0.412, 0.634] Sources=1 (prior=0.623 + shock=-0.087 + persistent=-0.013)
```

## Running the Examples

```bash
# Adani Group 40-month retrospective (Section 5.3)
python examples/adani_retrospective.py

# Federal Reserve consistency check (Section 5.4)
python examples/fed_rate_decision.py
```

## Running the Tests

```bash
python -m pytest tests/ -v
```

Expected output: All 16 tests pass.

## Source Tiers and Credibility Multipliers

| Tier | Source Type | c(τ) |
|------|-------------|-------|
| 1 | Court filings, regulatory orders, central bank statements | 1.00 |
| 2 | Local financial press (Economic Times, Business Standard) | 0.90 |
| 3 | Regional international press (Nikkei Asia, FT) | 0.75 |
| 4 | International wire (Reuters, Bloomberg) | 0.55 |
| 5 | Capital flow signals (FII positions, CDS spreads, bond yields) | **1.50** |
| 6 | Unverified single-source, social media | 0.10 |

Tier 5 receives `c(τ) > 1.0` because real capital deployment is a stronger epistemic signal than stated opinion. GQG Partners investing $1.87B in Adani is worth more than any number of editorial endorsements.

## Event Probability Swap Integration

The oracle output serves as the floating rate for event probability swaps:

```python
from oracle.models import SwapContract
from oracle.mtm import SwapLedger, initial_margin_schedule

swap = SwapContract(
    contract_id="swap_001",
    event_id="fed_rate_jun2023",
    notional=1_000_000,
    fixed_probability=0.52,
)

# Deterministic Margin Model (Venkatakrishnan, 2025)
# Total margin = (P_max - P_min) × N = 0.96 × N regardless of P_fixed
margin = initial_margin_schedule(swap.fixed_probability, swap.notional)
# {'fixed_payer_initial_margin': 500000.0, 'floating_payer_initial_margin': 460000.0, ...}

ledger = SwapLedger(swap)
record = ledger.process_oracle_output(oracle_output)
settlement = ledger.settle(outcome=0)
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `F_max` | 0.20 | Fast shock ceiling — single event cannot dominate |
| `P_max` | 0.80 | Persistent signal ceiling |
| `ρ_min` | 3 | Minimum independent sources for persistence gate |
| `θ_ind` | 0.40 | Independence threshold |
| `θ_imp` | 3.00 | Impact threshold |
| `γ_c` | 0.95 | Confirmation dampening — decay arrested on confirmation |
| `λ` (central bank) | 0.40/day | Fast shock decay rate |
| `W` (central bank) | 14 days | Persistence window |

All parameters are documented in Section 3 of the paper and configurable via `oracle/config.py`.

## Citation

```bibtex
@article{guruprasad2026oracle,
  title={The World as Oracle: Adversarial-Resistant AI Probability Estimation for Event-Driven Financial Instruments},
  author={Guruprasad Venkatakrishnan},
  institution={Verslan / predictmarkets.finance},
  year={2026},
  note={Working Paper}
}
```

## License

MIT License — Verslan Private Limited, 2026.
