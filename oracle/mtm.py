"""
Mark-to-Market and Swap Settlement — Sections 3.12 and 8.

Implements the full event probability swap settlement mechanics
including the Deterministic Margin Model (DMM) from Venkatakrishnan (2025).

Daily VM:      VM(t) = (P_oracle(t) - P_oracle(t-1)) × N × δ_direction
Terminal VM:   VM_terminal = (O_E - P_oracle(T-1)) × N × δ_direction
Initial Margin:
  IM_A = max(P_fixed - P_min, 0) × N   [fixed-rate payer]
  IM_B = max(P_max - P_fixed, 0) × N   [floating-rate payer]

VM is suspended when oracle state = CONTESTED or ORACLE_DEGRADED.

Guruprasad Venkatakrishnan — Verslan / predictmarkets.finance / verslan.xyz
"""

from typing import List, Optional
from oracle.models import OracleOutput, OracleState, SwapContract


SUSPEND_STATES = {OracleState.CONTESTED, OracleState.DEGRADED}


def daily_variation_margin(
    p_current: float,
    p_previous: float,
    notional: float,
    long_floating: bool = True,
    oracle_state: OracleState = OracleState.BASELINE,
) -> Optional[float]:
    """
    VM(t) = (P_oracle(t) - P_oracle(t-1)) × N × δ_direction

    Returns None if oracle state requires VM suspension.

    delta_direction = +1 for long floating (receives floating, pays fixed)
    delta_direction = -1 for short floating (pays floating, receives fixed)
    """
    if oracle_state in SUSPEND_STATES:
        return None  # VM suspended

    delta = +1.0 if long_floating else -1.0
    return (p_current - p_previous) * notional * delta


def terminal_settlement(
    outcome: int,
    p_previous: float,
    notional: float,
    long_floating: bool = True,
) -> float:
    """
    VM_terminal = (O_E - P_oracle(T-1)) × N × δ_direction

    outcome ∈ {0, 1} — verified event resolution
    """
    delta = +1.0 if long_floating else -1.0
    return (outcome - p_previous) * notional * delta


def initial_margin_schedule(
    p_fixed: float,
    notional: float,
    p_min: float = 0.02,
    p_max: float = 0.98,
) -> dict:
    """
    DMM initial margin — fully covers all admissible oracle outcomes.

    IM_A = max(P_fixed - P_min, 0) × N
    IM_B = max(P_max - P_fixed, 0) × N

    Total margin = (P_max - P_min) × N = 0.96 × N regardless of P_fixed.
    This is the unique property that makes event probability swaps DMM-eligible:
    probability is bounded by construction, so IM is deterministic at inception.
    """
    im_a = max(p_fixed - p_min, 0.0) * notional
    im_b = max(p_max - p_fixed, 0.0) * notional

    return {
        "fixed_payer_initial_margin": round(im_a, 2),
        "floating_payer_initial_margin": round(im_b, 2),
        "total_margin": round(im_a + im_b, 2),
        "margin_efficiency": round((im_a + im_b) / notional, 4),
        "max_loss_fixed_payer": round(im_a, 2),
        "max_loss_floating_payer": round(im_b, 2),
    }


class SwapLedger:
    """
    Tracks daily variation margin flows for an event probability swap.
    Implements the DMM settlement logic from Venkatakrishnan (2025).
    """

    def __init__(self, contract: SwapContract):
        self.contract = contract
        self._vm_history: List[dict] = []
        self._total_vm_party_a: float = 0.0  # Cumulative VM received by fixed payer
        self._p_previous: Optional[float] = None
        self._suspended_days: int = 0

    def process_oracle_output(
        self,
        output: OracleOutput,
        long_floating_is_party_a: bool = False,
    ) -> dict:
        """
        Process one oracle output and compute daily variation margin.
        Returns a settlement record.
        """
        suspended = output.state in SUSPEND_STATES

        if self._p_previous is None:
            # First day — no VM, just record
            self._p_previous = output.probability
            return {
                "date": output.timestamp,
                "p_oracle": output.probability,
                "p_previous": None,
                "vm_party_a": 0.0,
                "state": output.state.value,
                "suspended": False,
                "note": "Contract inception",
            }

        vm = daily_variation_margin(
            p_current=output.probability,
            p_previous=self._p_previous,
            notional=self.contract.notional,
            long_floating=long_floating_is_party_a,
            oracle_state=output.state,
        )

        if vm is None:
            self._suspended_days += 1
            record = {
                "date": output.timestamp,
                "p_oracle": output.probability,
                "p_previous": self._p_previous,
                "vm_party_a": 0.0,
                "state": output.state.value,
                "suspended": True,
                "note": f"VM suspended: {output.state.value}",
            }
        else:
            self._total_vm_party_a += vm
            record = {
                "date": output.timestamp,
                "p_oracle": output.probability,
                "p_previous": self._p_previous,
                "vm_party_a": round(vm, 2),
                "cumulative_vm_party_a": round(self._total_vm_party_a, 2),
                "state": output.state.value,
                "suspended": False,
            }
            self._p_previous = output.probability

        self._vm_history.append(record)
        return record

    def settle(self, outcome: int) -> dict:
        """Final settlement at event resolution."""
        if self._p_previous is None:
            return {"error": "No oracle history to settle against"}

        terminal_vm = terminal_settlement(
            outcome=outcome,
            p_previous=self._p_previous,
            notional=self.contract.notional,
            long_floating=True,
        )
        self._total_vm_party_a += terminal_vm

        return {
            "contract_id": self.contract.contract_id,
            "outcome": outcome,
            "p_final": self._p_previous,
            "terminal_vm": round(terminal_vm, 2),
            "total_vm_party_a": round(self._total_vm_party_a, 2),
            "suspended_days": self._suspended_days,
            "settlement_complete": True,
        }

    def summary(self) -> dict:
        return {
            "contract_id": self.contract.contract_id,
            "event_id": self.contract.event_id,
            "notional": self.contract.notional,
            "p_fixed": self.contract.fixed_probability,
            "days_tracked": len(self._vm_history),
            "suspended_days": self._suspended_days,
            "cumulative_vm_party_a": round(self._total_vm_party_a, 2),
            "initial_margin": initial_margin_schedule(
                self.contract.fixed_probability,
                self.contract.notional,
            ),
        }
