"""
Event category configuration.
Each category has its own temporal filter parameters as specified in Section 6.
Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

from dataclasses import dataclass
from oracle.models import EventCategory


@dataclass
class CategoryConfig:
    """Full parameter set for one event category."""
    category: EventCategory
    name: str

    # Temporal filter parameters (Section 3.5 / Section 6)
    decay_rate_lambda: float         # λ per day (no confirmation)
    persistence_window_days: int     # W: rolling window for persistent signal
    f_max: float                     # Fast shock ceiling (probability points)
    p_max: float                     # Persistent signal ceiling
    rho_min: int                     # Minimum independent sources for persistence gate

    # Independence and impact thresholds (Section 3.3 / 3.6)
    theta_ind: float = 0.40          # Independence threshold ι(a_i) ≥ θ_ind
    theta_imp: float = 3.00          # Impact threshold |s_i| ≥ θ_imp
    gamma_c: float = 0.95            # Confirmation dampening factor

    # Calibration targets (Appendix C)
    target_calibration_error: float = 0.08
    min_training_events: int = 300

    # Prior weights α, β, γ (Section 3.2)
    alpha: float = 0.60              # Historical base rate weight
    beta: float = 0.25               # Structural prior weight
    gamma: float = 0.15              # Market-implied prior weight

    # CONTESTED state threshold
    contested_divergence_threshold: float = 0.30   # 30 percentage points
    contested_duration_days: int = 3


CATEGORY_CONFIGS = {
    EventCategory.CENTRAL_BANK: CategoryConfig(
        category=EventCategory.CENTRAL_BANK,
        name="Central Bank Rate Decisions",
        decay_rate_lambda=0.40,
        persistence_window_days=14,
        f_max=0.20,
        p_max=0.80,
        rho_min=3,
        target_calibration_error=0.04,
        min_training_events=500,
        alpha=0.60,
        beta=0.25,
        gamma=0.15,
    ),
    EventCategory.GEOPOLITICAL: CategoryConfig(
        category=EventCategory.GEOPOLITICAL,
        name="Geopolitical Developments",
        decay_rate_lambda=0.25,
        persistence_window_days=7,
        f_max=0.20,
        p_max=0.80,
        rho_min=3,
        target_calibration_error=0.10,
        min_training_events=400,
        alpha=0.35,
        beta=0.50,
        gamma=0.15,
    ),
    EventCategory.CORPORATE_LEGAL: CategoryConfig(
        category=EventCategory.CORPORATE_LEGAL,
        name="Corporate Regulatory Actions",
        decay_rate_lambda=0.30,
        persistence_window_days=7,
        f_max=0.20,
        p_max=0.80,
        rho_min=3,
        target_calibration_error=0.08,
        min_training_events=300,
        alpha=0.50,
        beta=0.40,
        gamma=0.10,   # Market prior down-weighted per Bartlett & O'Hara (2026)
    ),
    EventCategory.ELECTORAL: CategoryConfig(
        category=EventCategory.ELECTORAL,
        name="Electoral Outcomes",
        decay_rate_lambda=0.20,
        persistence_window_days=21,
        f_max=0.20,
        p_max=0.80,
        rho_min=3,
        target_calibration_error=0.06,
        min_training_events=200,
        alpha=0.55,
        beta=0.30,
        gamma=0.15,
    ),
    EventCategory.MACRO_DATA: CategoryConfig(
        category=EventCategory.MACRO_DATA,
        name="Macroeconomic Data Releases",
        decay_rate_lambda=0.40,
        persistence_window_days=3,
        f_max=0.20,
        p_max=0.80,
        rho_min=2,
        alpha=0.65,
        beta=0.25,
        gamma=0.10,
    ),
    EventCategory.SOVEREIGN_CREDIT: CategoryConfig(
        category=EventCategory.SOVEREIGN_CREDIT,
        name="Sovereign Credit Events",
        decay_rate_lambda=0.20,
        persistence_window_days=14,
        f_max=0.20,
        p_max=0.80,
        rho_min=3,
        target_calibration_error=0.08,
        min_training_events=200,
        alpha=0.40,
        beta=0.45,
        gamma=0.15,
    ),
    EventCategory.CRYPTO_PROTOCOL: CategoryConfig(
        category=EventCategory.CRYPTO_PROTOCOL,
        name="Crypto Protocol Events",
        decay_rate_lambda=0.50,
        persistence_window_days=3,
        f_max=0.20,
        p_max=0.80,
        rho_min=2,
        target_calibration_error=0.03,
        min_training_events=1000,
        alpha=0.20,   # On-chain data dominant
        beta=0.70,
        gamma=0.10,
    ),
}
