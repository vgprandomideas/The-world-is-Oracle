"""
Probability Volatility Surface — σ(P, t)

Models how fast and how far the oracle probability can move.
This is the direct analogue of implied volatility for binary event derivatives.

Three volatility regimes:
  QUIET    — low signal density, oracle stable, σ ≈ 2-5%/day
  ACTIVE   — moderate signals, oracle moving, σ ≈ 5-15%/day
  CRISIS   — high-impact signals, oracle volatile, σ ≈ 15-40%/day

Vol surface inputs:
  P        — current probability level (vol smile: higher near 0/1)
  t        — time to resolution (vol term structure: rises near resolution)
  ρ        — signal density (event vol: more signals → higher vol)
  state    — oracle state (CONTESTED → vol spike)

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

import math
import numpy as np
from typing import List, Optional
from oracle.models import OracleState


# Base volatility per category (daily, in probability points)
CATEGORY_BASE_VOL = {
    "central_bank":     0.04,   # Tight — very liquid prediction market exists
    "geopolitical":     0.12,   # Wide — high uncertainty
    "corporate_legal":  0.08,   # Medium
    "electoral":        0.06,
    "macro_data":       0.03,   # Tight — data releases binary
    "sovereign_credit": 0.10,
    "crypto_protocol":  0.15,   # Widest — high regime uncertainty
}

STATE_VOL_MULTIPLIER = {
    OracleState.BASELINE:           1.0,
    OracleState.SHOCK_ACTIVE:       2.5,   # Fast shock → vol spike
    OracleState.BUILDING:           1.5,
    OracleState.SUSTAINED:          1.2,
    OracleState.CONTESTED:          3.5,   # Max vol — information warfare
    OracleState.INSUFFICIENT:       0.8,
    OracleState.DEGRADED:           4.0,
}


def probability_smile(p: float) -> float:
    """
    Vol smile: σ is higher when P is near 0 or 1.
    Near-certain events have high vol in probability space
    (small new evidence causes large percentage move).

    smile(P) = 1 / (4 · P · (1-P))^0.5 — normalised to 1.0 at P=0.5
    """
    p = max(0.02, min(0.98, p))
    raw = 1.0 / math.sqrt(4 * p * (1 - p))
    return raw  # ≈ 1.0 at P=0.5, → ∞ as P→0 or P→1


def time_to_resolution_vol(days_to_resolution: Optional[float]) -> float:
    """
    Vol term structure: rises as resolution approaches (pin risk).
    Far from resolution: vol is stable (flat term structure).
    Near resolution: vol spikes (binary gamma effect).

    If days_to_resolution is None: assume 90 days (mid-cycle).
    """
    if days_to_resolution is None:
        return 1.0
    if days_to_resolution <= 0:
        return 5.0  # Resolution day — maximum vol
    if days_to_resolution <= 1:
        return 3.0  # Day before
    if days_to_resolution <= 7:
        return 2.0  # Week before
    if days_to_resolution <= 30:
        return 1.3
    return 1.0     # > 30 days — base vol


def signal_density_vol(n_signals_24h: int) -> float:
    """
    More signals → more vol (information arrival rate).
    """
    if n_signals_24h == 0:
        return 0.7
    if n_signals_24h <= 2:
        return 1.0
    if n_signals_24h <= 5:
        return 1.4
    if n_signals_24h <= 10:
        return 1.8
    return 2.5   # >10 signals in 24h = crisis mode


def compute_vol_surface(
    p: float,
    category: str,
    state: OracleState,
    days_to_resolution: Optional[float] = None,
    n_signals_24h: int = 0,
    posterior_history: Optional[List[float]] = None,
) -> dict:
    """
    Full probability volatility surface computation.

    Returns annualised and daily vol in probability points,
    plus the vol regime and surface components.
    """
    # Base vol from category
    base_vol = CATEGORY_BASE_VOL.get(category, 0.08)

    # Apply smile (level adjustment)
    smile_mult = min(probability_smile(p), 4.0)  # Cap at 4x

    # Apply state multiplier
    state_mult = STATE_VOL_MULTIPLIER.get(state, 1.0)

    # Apply term structure
    term_mult = time_to_resolution_vol(days_to_resolution)

    # Apply signal density
    density_mult = signal_density_vol(n_signals_24h)

    # Realised vol from history (EWMA if available)
    realised_vol = None
    if posterior_history and len(posterior_history) >= 5:
        returns = np.diff(posterior_history)
        realised_vol = float(np.std(returns))
        # Blend implied and realised (60/40)
        implied_daily = base_vol * smile_mult * state_mult * density_mult
        daily_vol = 0.60 * implied_daily + 0.40 * realised_vol
    else:
        daily_vol = base_vol * smile_mult * state_mult * density_mult

    # Apply term structure on top
    daily_vol *= term_mult

    # Cap at sensible limits
    daily_vol = min(daily_vol, 0.45)  # Max 45pp/day
    daily_vol = max(daily_vol, 0.005) # Min 0.5pp/day

    # Annualised vol (× √252)
    annual_vol = daily_vol * math.sqrt(252)

    # Vol regime classification
    if daily_vol < 0.05:
        regime = "QUIET"
    elif daily_vol < 0.15:
        regime = "ACTIVE"
    else:
        regime = "CRISIS"

    return {
        "daily_vol": round(daily_vol, 4),
        "annual_vol": round(annual_vol, 4),
        "regime": regime,
        "components": {
            "base_vol": base_vol,
            "smile_multiplier": round(smile_mult, 3),
            "state_multiplier": state_mult,
            "term_multiplier": term_mult,
            "density_multiplier": density_mult,
        },
        "realised_vol": round(realised_vol, 4) if realised_vol else None,
        "vol_1d_range": (
            round(max(0.02, p - 1.645 * daily_vol), 4),
            round(min(0.98, p + 1.645 * daily_vol), 4),
        ),
        "vol_7d_range": (
            round(max(0.02, p - 1.645 * daily_vol * math.sqrt(7)), 4),
            round(min(0.98, p + 1.645 * daily_vol * math.sqrt(7)), 4),
        ),
    }


def compute_greeks(
    p: float,
    notional: float,
    daily_vol: float,
    days_to_resolution: float,
    fixed_probability: float,
) -> dict:
    """
    Greeks for event probability swap / binary option.

    Delta  = ∂V/∂P    — P&L per 1pp move in oracle probability
    Gamma  = ∂²V/∂P²  — convexity (acceleration of delta)
    Theta  = ∂V/∂t    — daily time decay
    Vega   = ∂V/∂σ    — sensitivity to oracle volatility

    For a swap: V = (P_oracle - P_fixed) × N
    Delta = N (constant for linear swap)

    For a binary option (pays N if P_oracle > strike at resolution):
    Uses normal approximation of binary option pricing.
    """
    T = max(days_to_resolution / 365.0, 1/365)  # In years
    sigma = daily_vol * math.sqrt(252)           # Annualised

    # Normal approximation for binary event option
    # d = (logit(P) - logit(K)) / (sigma * sqrt(T))
    try:
        d1 = (logit_safe(p) - logit_safe(fixed_probability)) / (sigma * math.sqrt(T) + 1e-9)
    except:
        d1 = 0.0

    from scipy.stats import norm

    # Binary call: pays N if oracle > strike at resolution
    # Price = N × N(d1)  where N is standard normal CDF
    binary_price = notional * norm.cdf(d1)

    # Delta: sensitivity of swap P&L to 1pp move in P
    # Swap delta = N (constant)
    swap_delta = notional  # Per unit probability

    # Binary option delta
    binary_delta = notional * norm.pdf(d1) / (sigma * math.sqrt(T) * p * (1-p) + 1e-9)

    # Gamma: second derivative — acceleration of binary delta
    gamma = -binary_delta * d1 / (sigma * math.sqrt(T) + 1e-9)

    # Theta: daily time decay (negative for long options)
    theta = -notional * norm.pdf(d1) * sigma / (2 * math.sqrt(T) * 365 + 1e-9)

    # Vega: sensitivity to 1% change in oracle vol
    vega = notional * norm.pdf(d1) * math.sqrt(T) * 0.01  # Per 1% vol change

    return {
        "swap_delta": round(swap_delta, 2),
        "binary_delta": round(binary_delta, 4),
        "gamma": round(gamma, 6),
        "theta_daily": round(theta, 2),
        "vega_per_1pct_vol": round(vega, 2),
        "binary_option_price": round(binary_price, 2),
        "d1": round(d1, 4),
        "interpretation": {
            "swap_delta": f"${swap_delta:,.0f} P&L per 1pp oracle move",
            "gamma": f"Delta changes by {abs(gamma):.2f} per 1pp oracle move",
            "theta": f"${abs(theta):,.2f}/day time decay",
            "vega": f"${abs(vega):,.2f} per 1% vol change",
        }
    }


def logit_safe(p: float) -> float:
    p = max(0.02, min(0.98, p))
    return math.log(p / (1 - p))
