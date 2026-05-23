"""
Data models for the World-as-Oracle probability estimation system.
Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class SourceTier(Enum):
    TIER1_PRIMARY = 1       # Court filings, regulatory orders, central bank statements
    TIER2_LOCAL_PRESS = 2   # Economic Times, Business Standard, Nikkei Asia
    TIER3_REGIONAL = 3      # FT, Nikkei (domain-adjusted)
    TIER4_WIRE = 4          # Reuters, Bloomberg (geopolitical-adjusted)
    TIER5_CAPITAL_FLOW = 5  # FII positions, CDS spreads, bond yields — real money
    TIER6_UNVERIFIED = 6    # Anonymous, social media, single-source unverified


CREDIBILITY_MULTIPLIERS = {
    SourceTier.TIER1_PRIMARY:    1.00,
    SourceTier.TIER2_LOCAL_PRESS: 0.90,
    SourceTier.TIER3_REGIONAL:   0.75,
    SourceTier.TIER4_WIRE:       0.55,
    SourceTier.TIER5_CAPITAL_FLOW: 1.50,  # Real money > stated opinion
    SourceTier.TIER6_UNVERIFIED: 0.10,
}


class OracleState(Enum):
    BASELINE         = "BASELINE"          # Prior dominates; no active signal
    SHOCK_ACTIVE     = "SHOCK_ACTIVE"      # Fast shock fired; threshold not yet crossed
    BUILDING         = "BUILDING"          # Persistent signal accumulating
    SUSTAINED        = "SUSTAINED"         # Strong persistent signal maintained ≥14 days
    CONTESTED        = "CONTESTED"         # Primary source vs media narrative diverge >30pp
    INSUFFICIENT     = "INSUFFICIENT_SIGNAL"
    DEGRADED         = "ORACLE_DEGRADED"


class EventCategory(Enum):
    CENTRAL_BANK     = "central_bank"
    GEOPOLITICAL     = "geopolitical"
    CORPORATE_LEGAL  = "corporate_legal"
    ELECTORAL        = "electoral"
    MACRO_DATA       = "macro_data"
    SOVEREIGN_CREDIT = "sovereign_credit"
    CRYPTO_PROTOCOL  = "crypto_protocol"


@dataclass
class Article:
    """A single ingested article or signal."""
    article_id: str
    source_name: str
    tier: SourceTier
    publication_time: float          # Unix timestamp
    headline: str
    content_summary: str
    url: str = ""

    # Set by independence scorer
    independence_score: float = 0.0  # ι(a_i) ∈ [0, 1]
    primary_source_refs: list = field(default_factory=list)

    # Set by impact scorer (Layer 3)
    raw_impact: float = 0.0          # r_i ∈ [1, 10]
    direction: int = 0               # d_i ∈ {+1, -1, 0}
    reasoning_chain: str = ""
    counter_evidence: str = ""

    # Derived
    @property
    def credibility(self) -> float:
        return CREDIBILITY_MULTIPLIERS[self.tier]

    @property
    def signed_impact(self) -> float:
        """s_i = r_i · c(τ) · d_i"""
        return self.raw_impact * self.credibility * self.direction


@dataclass
class OracleOutput:
    """Full oracle output for a given moment in time."""
    event_id: str
    timestamp: float
    probability: float               # P_oracle(t) ∈ [0.02, 0.98]
    ci_lower: float
    ci_upper: float
    state: OracleState

    # Signal decomposition
    prior: float
    fast_shock: float                # F(t)
    persistent_signal: float         # P(t)
    p_raw: float                     # Before Platt scaling

    # Evidence metadata
    independent_source_count: int
    total_articles_processed: int

    # Audit
    audit_trail: list = field(default_factory=list)

    @property
    def ci_width(self) -> float:
        return self.ci_upper - self.ci_lower

    def __str__(self):
        return (
            f"Oracle [{self.state.value}] "
            f"P={self.probability:.3f} "
            f"CI=[{self.ci_lower:.3f}, {self.ci_upper:.3f}] "
            f"Sources={self.independent_source_count} "
            f"(prior={self.prior:.3f} + shock={self.fast_shock:.3f} + persistent={self.persistent_signal:.3f})"
        )


@dataclass
class SwapContract:
    """An event probability swap contract."""
    contract_id: str
    event_id: str
    notional: float
    fixed_probability: float         # P_fixed agreed at inception
    p_min: float = 0.02
    p_max: float = 0.98

    @property
    def initial_margin_fixed_payer(self) -> float:
        """IM_A = max(P_fixed - P_min, 0) × N"""
        return max(self.fixed_probability - self.p_min, 0) * self.notional

    @property
    def initial_margin_floating_payer(self) -> float:
        """IM_B = max(P_max - P_fixed, 0) × N"""
        return max(self.p_max - self.fixed_probability, 0) * self.notional

    @property
    def total_margin(self) -> float:
        return self.initial_margin_fixed_payer + self.initial_margin_floating_payer

    def variation_margin(self, p_prev: float, p_current: float,
                         long_floating: bool = True) -> float:
        """VM(t) = (P_oracle(t) - P_oracle(t-1)) × N × δ_direction"""
        delta = +1 if long_floating else -1
        return (p_current - p_prev) * self.notional * delta

    def terminal_settlement(self, p_prev: float, outcome: int,
                            long_floating: bool = True) -> float:
        """VM_terminal = (O_E - P_oracle(T-1)) × N × δ_direction"""
        delta = +1 if long_floating else -1
        return (outcome - p_prev) * self.notional * delta
