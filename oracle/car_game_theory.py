"""
CAR Game Theory Engine — Stage 3

Automatically constructs the game theory model from actor profiles.
No human needs to inject the payoff matrix.
The engine infers it from historical actions.

Three outputs:
  1. Payoff matrix — utility values for all actors in all resolution states
  2. Dominant strategy detection — who wins in BOTH scenarios
  3. Equilibrium classification — what state the system will settle into

Guruprasad Venkatakrishnan (2026)
Verslan / predictmarkets.finance / verslan.xyz
"""

import json
import math
from typing import List, Dict, Optional
import anthropic

client = anthropic.Anthropic()


GAME_THEORY_SYSTEM = """You are a CAR Game Theory Engine.

Given a set of actor profiles with verified utility values,
you automatically construct:
  1. The payoff matrix
  2. Dominant strategy detection
  3. Nash equilibrium identification
  4. Equilibrium classification

You think like a strategic intelligence analyst who has read
all the relevant history and understands that:

RULE 1: Actors with dominant strategies (win in ALL scenarios)
        are the most dangerous. They have no incentive to cooperate.
        Russia on Hormuz: earns from disruption + India buys Russian
        in normalcy. Dominant strategy = obstruct indefinitely.

RULE 2: Actors playing multiple parties simultaneously have
        structural contradictions that will eventually collapse.
        When the contradiction is exposed, ALL their positions
        collapse simultaneously.

RULE 3: The equilibrium is not where the principals want to go.
        The equilibrium is where the sum of actor incentives
        lands the system — regardless of stated intentions.

RULE 4: Identify the BLOCKING COALITION — the set of actors
        whose combined obstruction exceeds the resolving actors'
        combined pushing power. If blocking > resolving: no deal.

Output valid JSON only."""


def construct_payoff_matrix(actor_profiles: List[dict], event: str) -> dict:
    """
    Build the full game theory model from actor profiles.
    """
    actors_summary = json.dumps([
        {
            "actor": p.get("actor"),
            "class": p.get("actor_class"),
            "utility_yes": p.get("utility_yes"),
            "utility_no": p.get("utility_no"),
            "obstruction_incentive": p.get("obstruction_incentive", 0),
            "blocking_power": p.get("blocking_power", 0),
            "structural_contradiction": p.get("car_questions", {}).get(
                "Q4_multi_party_selling", {}).get("structural_contradiction", "none"),
            "collapse_trigger": p.get("car_questions", {}).get(
                "Q8_collapse_trigger", {}).get("trigger", "unknown"),
        }
        for p in actor_profiles
    ], indent=2)

    prompt = f"""Event: {event}

Actor profiles with verified utility values:
{actors_summary}

Construct the complete game theory analysis. Return this JSON:
{{
  "event": "{event}",

  "payoff_matrix": {{
    "description": "What each actor gets in each scenario",
    "resolution_YES": {{}},
    "resolution_NO": {{}}
  }},

  "dominant_strategies": [
    {{
      "actor": "...",
      "strategy": "obstruct|resolve|exploit|sit",
      "reason": "wins in YES and NO because...",
      "is_truly_dominant": true
    }}
  ],

  "blocking_coalition": {{
    "members": [],
    "combined_blocking_power": 0.0,
    "assessment": "blocking coalition exceeds resolving coalition"
  }},

  "resolving_coalition": {{
    "members": [],
    "combined_resolving_power": 0.0,
    "assessment": "..."
  }},

  "nash_equilibrium": {{
    "state": "resolution|disruption|managed_disruption|collapse",
    "stability": "stable|unstable|metastable",
    "description": "...",
    "probability": 0.0
  }},

  "structural_contradictions": [
    {{
      "actor": "...",
      "contradiction": "...",
      "collapse_trigger": "...",
      "collapse_probability_12m": 0.0
    }}
  ],

  "adversarial_haircut_total": 0.0,
  "p_naive_adjustment_factor": 0.0,
  "dominant_scenario_2027": "...",
  "dominant_scenario_probability": 0.0,

  "car_final_probability": 0.0,
  "ci_lower": 0.0,
  "ci_upper": 0.0,

  "key_insight": "The one thing standard AI would completely miss about this situation"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2500,
        system=GAME_THEORY_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e), "raw": raw}


def detect_equilibrium_type(game_model: dict) -> str:
    """
    Classify the equilibrium type from the game model.
    This determines how to interpret the CAR probability.
    """
    ne = game_model.get("nash_equilibrium", {})
    state = ne.get("state", "unknown")
    stability = ne.get("stability", "unknown")

    blocking = game_model.get("blocking_coalition", {}).get("combined_blocking_power", 0)
    resolving = game_model.get("resolving_coalition", {}).get("combined_resolving_power", 0)

    if blocking > resolving and stability == "stable":
        return "STABLE_DISRUPTION"
    elif blocking > resolving and stability == "metastable":
        return "MANAGED_DISRUPTION"
    elif resolving > blocking:
        return "RESOLUTION_POSSIBLE"
    elif stability == "unstable":
        return "VOLATILE_UNKNOWN"
    else:
        return "DEADLOCK"


def compute_final_car_probability(
    p_naive: float,
    game_model: dict,
) -> dict:
    """
    Apply game theory adjustments to get final P_CAR.
    """
    # Get adversarial haircut from game model
    ah = game_model.get("adversarial_haircut_total", 0)
    ah = min(0.45, ah)

    # Get structural contradiction risk
    contradictions = game_model.get("structural_contradictions", [])
    contradiction_discount = sum(
        c.get("collapse_probability_12m", 0) * 0.1
        for c in contradictions
    )
    contradiction_discount = min(0.15, contradiction_discount)

    # Apply adjustments
    p_car = p_naive * (1 - ah) * (1 - contradiction_discount)
    p_car = max(0.02, min(0.98, p_car))

    # Compute CI from Nash equilibrium stability
    equilibrium_type = detect_equilibrium_type(game_model)
    ci_width = {
        "STABLE_DISRUPTION": 0.08,
        "MANAGED_DISRUPTION": 0.14,
        "RESOLUTION_POSSIBLE": 0.12,
        "VOLATILE_UNKNOWN": 0.20,
        "DEADLOCK": 0.10,
    }.get(equilibrium_type, 0.12)

    return {
        "p_naive": round(p_naive, 3),
        "adversarial_haircut": round(ah, 3),
        "contradiction_discount": round(contradiction_discount, 3),
        "p_car": round(p_car, 3),
        "ci_lower": round(max(0.02, p_car - ci_width), 3),
        "ci_upper": round(min(0.98, p_car + ci_width), 3),
        "equilibrium_type": equilibrium_type,
        "dominant_scenario": game_model.get("dominant_scenario_2027", "Unknown"),
        "dominant_scenario_probability": game_model.get("dominant_scenario_probability", 0),
    }


def construct_payoff_matrix_deterministic(actor_profiles: list, event: str) -> dict:
    """
    Deterministic game theory model — no API key needed.
    Computes payoff matrix mathematically from actor profile scores.
    """
    import math

    obstructors = [p for p in actor_profiles if p.get("obstruction_incentive", 0) > 1.5]
    fence_sitters = [p for p in actor_profiles if p.get("actor_class") == "fence_sitter"]
    resolvers = [p for p in actor_profiles
                 if p.get("utility_yes", 0) > p.get("utility_no", 0)
                 and p.get("obstruction_incentive", 0) < 1.0]

    # Blocking coalition power
    blocking_power = sum(
        p.get("blocking_power", 0) * min(1.0, p.get("obstruction_incentive", 0) / 5.0)
        for p in obstructors
    )
    blocking_power = min(0.99, blocking_power)

    # Resolving coalition power
    resolving_power = sum(
        p.get("blocking_power", 0) * (p.get("utility_yes", 5) - p.get("utility_no", 5)) / 10.0
        for p in resolvers
    )
    resolving_power = min(0.99, resolving_power)

    # Total structural silence
    total_ss = sum(p.get("structural_silence_score", 0) for p in actor_profiles)
    adversarial_haircut = min(0.45, total_ss / (total_ss + 1.5))

    # Nash equilibrium
    if blocking_power > resolving_power * 1.3:
        eq_state = "stable_disruption"
        eq_stability = "stable"
        eq_prob = 0.65
    elif blocking_power > resolving_power:
        eq_state = "managed_disruption"
        eq_stability = "metastable"
        eq_prob = 0.50
    else:
        eq_state = "resolution_possible"
        eq_stability = "unstable"
        eq_prob = 0.35

    # Dominant strategies
    dominant = []
    for p in actor_profiles:
        if p.get("utility_no", 0) > 6.0 and p.get("utility_yes", 0) > 5.0:
            dominant.append({
                "actor": p.get("actor"),
                "strategy": "dominant_obstruct",
                "reason": f"Wins in both YES ({p.get('utility_yes')}) and NO ({p.get('utility_no')})",
                "is_truly_dominant": True
            })

    # Structural contradictions
    contradictions = []
    for p in actor_profiles:
        sc = p.get("car_questions", {}).get("Q4_multi_party_selling", {}).get("structural_contradiction", "")
        if sc and sc != "none":
            contradictions.append({
                "actor": p.get("actor"),
                "contradiction": sc,
                "collapse_trigger": p.get("car_questions", {}).get("Q8_collapse_trigger", {}).get("trigger", "unknown"),
                "collapse_probability_12m": 0.20
            })

    # Build payoff matrix
    payoff_yes = {p.get("actor"): p.get("utility_yes", 5.0) for p in actor_profiles}
    payoff_no  = {p.get("actor"): p.get("utility_no", 5.0) for p in actor_profiles}

    dominant_scenario = (
        "Stable Disrupted Equilibrium: Blocking coalition dominates. "
        "Process continues indefinitely. Key actors profit from duration."
        if blocking_power > resolving_power else
        "Resolution pathway exists but contested. Outcome depends on China fence-sitter decision."
    )

    key_insight = (
        f"Standard AI assigns positive weight to all stated mediator positions. "
        f"CAR detects {len(obstructors)} structural obstructors with combined "
        f"blocking power {blocking_power:.2f} vs resolving power {resolving_power:.2f}. "
        f"The blocking coalition exceeds the resolving coalition. "
        f"No deal is the equilibrium, not the exception."
    )

    return {
        "event": event,
        "payoff_matrix": {"resolution_YES": payoff_yes, "resolution_NO": payoff_no},
        "dominant_strategies": dominant,
        "blocking_coalition": {
            "members": [p.get("actor") for p in obstructors],
            "combined_blocking_power": round(blocking_power, 3),
            "assessment": f"Blocking coalition power {blocking_power:.2f}"
        },
        "resolving_coalition": {
            "members": [p.get("actor") for p in resolvers],
            "combined_resolving_power": round(resolving_power, 3),
            "assessment": f"Resolving coalition power {resolving_power:.2f}"
        },
        "nash_equilibrium": {
            "state": eq_state,
            "stability": eq_stability,
            "description": dominant_scenario,
            "probability": eq_prob
        },
        "structural_contradictions": contradictions,
        "adversarial_haircut_total": round(adversarial_haircut, 3),
        "p_naive_adjustment_factor": round(1 - adversarial_haircut, 3),
        "dominant_scenario_2027": dominant_scenario,
        "dominant_scenario_probability": round(eq_prob, 2),
        "car_final_probability": 0.0,
        "ci_lower": 0.0,
        "ci_upper": 0.0,
        "key_insight": key_insight,
        "method": "deterministic_car"
    }
