"""
Counterfactual Adversarial Reasoning (CAR) — Core Framework

The seven-step protocol that AI cannot do autonomously because
it was never trained to ask: "Who benefits from this NOT resolving?"

Three key innovations:
  1. Obstruction Incentive (OI) — U(actor|NO) - U(actor|YES)
  2. Structural Silence Score (SS) — high OI + absent from signal chain
  3. Actor Fracture — split compound actors into sub-agents

Guruprasad Venkatakrishnan (2026) — Verslan / predictmarkets.finance
"""

import math
import json
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class BlockingMechanism(Enum):
    VETO_POWER        = "veto_power"
    RESOURCE_CONTROL  = "resource_control"
    PROXY_INFLUENCE   = "proxy_influence"
    INFO_CONTROL      = "info_control"
    FINANCIAL_FLOW    = "financial_flow"
    MILITARY          = "military"


class SupplyType(Enum):
    SUBSTITUTIONAL  = "substitutional"   # Replaces disrupted supply → reduces urgency
    ADDITIVE        = "additive"         # Added on top → no urgency change
    COMPLEMENTARY   = "complementary"    # Profitable BECAUSE of disruption → obstructs


@dataclass
class Actor:
    name: str
    actor_type: str                          # "state", "corporate", "multilateral"
    utility_yes: float                       # Payoff if event resolves YES [0-10]
    utility_no: float                        # Payoff if event resolves NO  [0-10]
    blocking_mechanisms: List[BlockingMechanism] = field(default_factory=list)
    presence_in_signal_chain: float = 0.5   # 0 = silent, 1 = dominant in news
    description: str = ""

    # Sub-actors for fractured entities
    sub_actors: List["Actor"] = field(default_factory=list)

    @property
    def obstruction_incentive(self) -> float:
        """OI(A,E) = max(0, U(NO) - U(YES))"""
        return max(0.0, self.utility_no - self.utility_yes)

    @property
    def blocking_power(self) -> float:
        """BP based on blocking mechanism portfolio"""
        weights = {
            BlockingMechanism.VETO_POWER:       1.00,
            BlockingMechanism.RESOURCE_CONTROL: 0.90,
            BlockingMechanism.PROXY_INFLUENCE:  0.70,
            BlockingMechanism.INFO_CONTROL:     0.50,
            BlockingMechanism.FINANCIAL_FLOW:   0.80,
            BlockingMechanism.MILITARY:         0.95,
        }
        if not self.blocking_mechanisms:
            return 0.20
        raw = sum(weights[m] for m in self.blocking_mechanisms)
        # Diminishing returns: each additional mechanism adds less
        return min(0.99, 1.0 - (1.0 / (1.0 + raw)))

    @property
    def structural_silence_score(self) -> float:
        """
        SS(A,E) = OI × BP × (1 - Presence)

        High OI + high BP + low presence = STRUCTURALLY SILENT OBSTRUCTOR
        This is the key signal standard AI completely misses.
        """
        return self.obstruction_incentive * self.blocking_power * (1.0 - self.presence_in_signal_chain)

    @property
    def is_fractured(self) -> bool:
        return len(self.sub_actors) > 0

    def fracture_divergence(self) -> float:
        """How much do sub-actors disagree? 0 = unified, 1 = maximally split"""
        if not self.sub_actors:
            return 0.0
        utilities = [s.utility_yes for s in self.sub_actors]
        return (max(utilities) - min(utilities)) / 10.0

    def dominant_sub_actor(self) -> Optional["Actor"]:
        """Which sub-actor's incentive dominates?"""
        if not self.sub_actors:
            return None
        return max(self.sub_actors, key=lambda a: a.blocking_power)


@dataclass
class CARAnalysis:
    event_id: str
    event_description: str
    resolution_criteria: str

    actors: List[Actor] = field(default_factory=list)
    supply_signals: List[dict] = field(default_factory=list)

    # Core outputs
    adversarial_haircut: float = 0.0
    dominant_obstructor: Optional[Actor] = None
    fractured_actors: List[Actor] = field(default_factory=list)
    structural_silence_warning: bool = False

    def run(self) -> "CARAnalysis":
        """Execute the full 7-step CAR protocol."""

        # Step 1: Actor enumeration — done at init

        # Step 2 & 3: Identify obstructors
        obstructors = [a for a in self.actors if a.obstruction_incentive > 0]

        # Step 4: Structural silence audit
        high_ss = [a for a in obstructors if a.structural_silence_score > 0.3]
        if high_ss:
            self.structural_silence_warning = True
            self.dominant_obstructor = max(high_ss, key=lambda a: a.structural_silence_score)

        # Step 5: Actor fracture analysis
        self.fractured_actors = [a for a in self.actors if a.is_fractured]

        # Step 6: Compute adversarial haircut
        total_obstruction = sum(
            a.structural_silence_score
            for a in self.actors
            if a.obstruction_incentive > 0
        )
        # Normalise: sigmoid-like compression so haircut stays in [0, 0.45]
        self.adversarial_haircut = min(0.45, total_obstruction / (total_obstruction + 1.5))

        # Step 7: Regime tree weights (for fractured actors)
        for actor in self.fractured_actors:
            actor._regime_divergence = actor.fracture_divergence()

        return self

    def apply_to_probability(self, p_naive: float) -> float:
        """
        P_oracle_CAR = P_naive × (1 - AH)

        The adversarial haircut systematically discounts naive estimates
        when structurally silent obstructors are present.
        """
        return max(0.02, min(0.98, p_naive * (1.0 - self.adversarial_haircut)))

    def report(self) -> dict:
        """Structured CAR report — training signal for the model."""
        return {
            "event_id": self.event_id,
            "event_description": self.event_description,
            "step1_actors": [
                {
                    "name": a.name,
                    "type": a.actor_type,
                    "utility_yes": a.utility_yes,
                    "utility_no": a.utility_no,
                    "obstruction_incentive": round(a.obstruction_incentive, 3),
                    "blocking_power": round(a.blocking_power, 3),
                    "presence_in_signal_chain": a.presence_in_signal_chain,
                    "structural_silence_score": round(a.structural_silence_score, 3),
                }
                for a in self.actors
            ],
            "step2_obstructors": [
                a.name for a in self.actors if a.obstruction_incentive > 0
            ],
            "step3_dominant_obstructor": {
                "name": self.dominant_obstructor.name,
                "structural_silence_score": round(self.dominant_obstructor.structural_silence_score, 3),
                "description": self.dominant_obstructor.description,
            } if self.dominant_obstructor else None,
            "step4_silence_warning": self.structural_silence_warning,
            "step5_fractured_actors": [
                {
                    "name": a.name,
                    "divergence": round(a.fracture_divergence(), 3),
                    "sub_actors": [s.name for s in a.sub_actors],
                    "dominant_sub_actor": a.dominant_sub_actor().name if a.dominant_sub_actor() else None,
                }
                for a in self.fractured_actors
            ],
            "step6_adversarial_haircut": round(self.adversarial_haircut, 4),
            "step7_regime_trees": [
                {
                    "actor": a.name,
                    "regime_divergence": round(a.fracture_divergence(), 3),
                    "sub_actor_payoffs": [
                        {"name": s.name, "utility_yes": s.utility_yes,
                         "utility_no": s.utility_no}
                        for s in a.sub_actors
                    ]
                }
                for a in self.fractured_actors
            ],
        }
