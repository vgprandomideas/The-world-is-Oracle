"""
Paper-backed CAR presets for deterministic scenario analysis.
"""

from copy import deepcopy


CAR_PRESETS = {
    "hormuz_2027": {
        "title": "Hormuz 2027",
        "question": "Will the Strait of Hormuz return to full commercial operation by December 2027?",
        "context": (
            "US-Iran ceasefire has intermittently held, Maersk test transits resumed under escort, "
            "but shipping volume remains impaired and several actors continue to profit from duration."
        ),
        "naive_probability": 0.52,
        "expected_outcome": "Stable disruption with low strategic resolution probability.",
        "seed_event": {
            "event_id": "hormuz_2027",
            "description": "Strait of Hormuz returns to full commercial operation by December 2027",
            "resolution": "YES if unrestricted commercial traffic returns to near-normal baseline by 2027-12-31.",
            "category": "geopolitical",
            "historical_base_rate": 0.28,
            "structural_prior": 0.43,
            "market_implied_prior": 0.52,
        },
        "actors": [
            {
                "actor": "Russia",
                "actor_class": "obstructor",
                "utility_yes": 3.0,
                "utility_no": 9.5,
                "blocking_power": 0.85,
                "structural_silence_score": 4.52,
                "one_line_summary": "Silent beneficiary of disruption via higher oil revenues and durable leverage.",
            },
            {
                "actor": "China",
                "actor_class": "fence_sitter",
                "utility_yes": 6.5,
                "utility_no": 6.0,
                "blocking_power": 0.85,
                "structural_silence_score": 0.0,
                "structural_contradiction": "Calls for reopening while preserving optionality by blocking multilateral coercion.",
                "collapse_trigger": "Commits openly to one side of the corridor security regime.",
                "one_line_summary": "High-power fence sitter whose choice can tip the equilibrium.",
            },
            {
                "actor": "Pakistan",
                "actor_class": "nuclear_rentier",
                "utility_yes": 4.5,
                "utility_no": 5.5,
                "blocking_power": 0.72,
                "structural_silence_score": 0.35,
                "structural_contradiction": "Selling strategic reassurance to Saudi Arabia while mediating Iran.",
                "collapse_trigger": "Its contradictory alignments become fully exposed to both patrons.",
                "one_line_summary": "Extracts relevance from prolonged crisis while juggling incompatible patrons.",
            },
            {
                "actor": "Houthis",
                "actor_class": "vendetta_architect",
                "utility_yes": 3.0,
                "utility_no": 7.0,
                "blocking_power": 0.70,
                "structural_silence_score": 2.10,
                "one_line_summary": "Vendetta actor whose payoff is not exhausted by principal-state bargaining.",
            },
            {
                "actor": "United States",
                "actor_class": "credibility_depleted_resolver",
                "utility_yes": 6.0,
                "utility_no": 3.5,
                "blocking_power": 0.90,
                "structural_silence_score": 0.0,
                "one_line_summary": "Wants reopening but carries credibility damage from prior escalation.",
            },
            {
                "actor": "India",
                "actor_class": "fractured_resolver",
                "utility_yes": 7.0,
                "utility_no": 5.5,
                "blocking_power": 0.65,
                "structural_silence_score": 0.0,
                "one_line_summary": "Commercially motivated resolver with internal divergence across government and refiners.",
            },
            {
                "actor": "Iran",
                "actor_class": "wounded_opportunist",
                "utility_yes": 5.0,
                "utility_no": 4.0,
                "blocking_power": 0.80,
                "structural_silence_score": 0.0,
                "one_line_summary": "Can live with managed normalisation if it preserves deterrence and face.",
            },
        ],
    },
    "adani_2026": {
        "title": "Adani 2026",
        "question": "Will the Adani Group face severe US enforcement action by December 2026?",
        "context": (
            "Negative narrative bursts persisted across Western wire coverage, but real-money longs, "
            "continued lending, and narrowed primary legal scope contradicted the implied severity."
        ),
        "naive_probability": 0.72,
        "expected_outcome": "Narrative intensity exceeds underlying enforcement severity.",
        "seed_event": {
            "event_id": "adani_2026",
            "description": "Adani Group faces severe US enforcement action by December 2026",
            "resolution": "YES if material US enforcement meaningfully impairs group financing or operations by 2026-12-31.",
            "category": "corporate_legal",
            "historical_base_rate": 0.22,
            "structural_prior": 0.34,
            "market_implied_prior": 0.72,
        },
        "actors": [
            {
                "actor": "GQG Partners",
                "actor_class": "capital_flow_resolver",
                "utility_yes": 1.5,
                "utility_no": 8.5,
                "blocking_power": 0.55,
                "structural_silence_score": 1.10,
                "one_line_summary": "Real-money long that silently profits from the narrative being wrong.",
            },
            {
                "actor": "Adani Group",
                "actor_class": "obstructor",
                "utility_yes": 1.0,
                "utility_no": 9.0,
                "blocking_power": 0.75,
                "structural_silence_score": 2.20,
                "one_line_summary": "Core beneficiary of non-resolution with financing incentives to survive the narrative.",
            },
            {
                "actor": "Indian Banks",
                "actor_class": "balance_sheet_resolver",
                "utility_yes": 2.5,
                "utility_no": 7.0,
                "blocking_power": 0.60,
                "structural_silence_score": 0.85,
                "one_line_summary": "Lenders reveal confidence through continued funding rather than commentary.",
            },
            {
                "actor": "US DOJ",
                "actor_class": "institutional_resolver",
                "utility_yes": 7.0,
                "utility_no": 4.0,
                "blocking_power": 0.78,
                "structural_silence_score": 0.0,
                "one_line_summary": "Primary legal actor with narrower incentives than headline coverage implied.",
            },
            {
                "actor": "Hindenburg",
                "actor_class": "narrative_amplifier",
                "utility_yes": 8.0,
                "utility_no": 1.0,
                "blocking_power": 0.45,
                "structural_silence_score": 0.0,
                "one_line_summary": "Benefits from narrative escalation but does not benefit from non-resolution.",
            },
        ],
    },
    "fed_jun_2023": {
        "title": "Fed June 2023",
        "question": "Will the Federal Reserve hold rates at the June 2023 FOMC meeting?",
        "context": (
            "Standardised macro event with liquid price discovery, abundant primary data, and no strategically silent obstructor."
        ),
        "naive_probability": 0.52,
        "expected_outcome": "No CAR haircut: adversarial layer should stay quiet when the event does not need it.",
        "seed_event": {
            "event_id": "fed_jun_2023",
            "description": "Federal Reserve holds rates at the June 2023 FOMC meeting",
            "resolution": "YES if the target range is unchanged at the June 2023 decision.",
            "category": "central_bank",
            "historical_base_rate": 0.48,
            "structural_prior": 0.50,
            "market_implied_prior": 0.52,
        },
        "actors": [
            {
                "actor": "Federal Reserve",
                "actor_class": "genuine_resolver",
                "utility_yes": 6.0,
                "utility_no": 4.0,
                "blocking_power": 0.95,
                "structural_silence_score": 0.0,
                "one_line_summary": "Primary policy actor with transparent institutional incentives.",
            },
            {
                "actor": "Treasury Market",
                "actor_class": "data_anchor",
                "utility_yes": 5.4,
                "utility_no": 5.0,
                "blocking_power": 0.60,
                "structural_silence_score": 0.0,
                "one_line_summary": "Liquid price discovery adds signal without strategic silence.",
            },
            {
                "actor": "CPI Path",
                "actor_class": "data_anchor",
                "utility_yes": 5.2,
                "utility_no": 5.0,
                "blocking_power": 0.45,
                "structural_silence_score": 0.0,
                "one_line_summary": "Macro data informs the decision but is not an adversarial actor.",
            },
            {
                "actor": "Labor Market",
                "actor_class": "data_anchor",
                "utility_yes": 5.1,
                "utility_no": 5.0,
                "blocking_power": 0.45,
                "structural_silence_score": 0.0,
                "one_line_summary": "Signal source, not manipulative strategic participant.",
            },
        ],
    },
}


def list_car_presets():
    presets = []
    for key, preset in CAR_PRESETS.items():
        presets.append(
            {
                "key": key,
                "title": preset["title"],
                "question": preset["question"],
                "naive_probability": preset["naive_probability"],
                "actor_count": len(preset["actors"]),
                "event_id": preset["seed_event"]["event_id"],
                "category": preset["seed_event"]["category"],
            }
        )
    return presets


def get_car_preset(key):
    preset = CAR_PRESETS.get(key)
    if not preset:
        return None
    return deepcopy(preset)


def get_preset_for_event_id(event_id):
    for preset in CAR_PRESETS.values():
        if preset["seed_event"]["event_id"] == event_id:
            return deepcopy(preset)
    return None
