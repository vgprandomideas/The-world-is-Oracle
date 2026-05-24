"""
World-as-Oracle: Adversarial-Resistant AI Probability Estimation
for Event-Driven Financial Instruments.

Paper: Guruprasad Venkatakrishnan (2026)
Verslan / predictmarkets.finance / verslan.xyz

Architecture: 7-layer temporal persistence oracle
"""

from oracle.models import (
    Article, OracleOutput, OracleState, OracleState,
    SourceTier, EventCategory, SwapContract,
    CREDIBILITY_MULTIPLIERS,
)
from oracle.oracle import WorldOracle
from oracle.deterministic_car import run_deterministic_car
from oracle.portfolio import PortfolioPosition, analyze_portfolio_risk
from oracle.config import CATEGORY_CONFIGS, CategoryConfig
from oracle.mtm import SwapLedger, initial_margin_schedule
from oracle.probability import PlattScaler, BrierScoreTracker

__version__ = "2.0.0"
__author__ = "Guruprasad Venkatakrishnan"
__affiliation__ = "Verslan / predictmarkets.finance / verslan.xyz"

__all__ = [
    "WorldOracle",
    "Article",
    "OracleOutput",
    "OracleState",
    "SourceTier",
    "EventCategory",
    "SwapContract",
    "SwapLedger",
    "PortfolioPosition",
    "PlattScaler",
    "BrierScoreTracker",
    "run_deterministic_car",
    "analyze_portfolio_risk",
    "CATEGORY_CONFIGS",
    "CREDIBILITY_MULTIPLIERS",
    "initial_margin_schedule",
]
