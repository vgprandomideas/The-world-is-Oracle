"""
Deterministic CAR analysis that works without external model calls.
"""

from copy import deepcopy


def _clamp(value, low, high):
    return max(low, min(high, value))


def _derive_actor_class(profile):
    actor_class = profile.get("actor_class")
    if actor_class:
        return actor_class
    if profile.get("structural_silence_score", 0.0) >= 2.0:
        return "obstructor"
    if profile.get("utility_yes", 5.0) > profile.get("utility_no", 5.0):
        return "genuine_resolver"
    return "mixed_actor"


def normalize_actor_profiles(actor_profiles):
    normalized = []
    for raw in actor_profiles:
        profile = deepcopy(raw)
        profile["actor"] = profile.get("actor", "Unknown actor")
        profile["utility_yes"] = float(profile.get("utility_yes", 5.0))
        profile["utility_no"] = float(profile.get("utility_no", 5.0))
        profile["blocking_power"] = _clamp(float(profile.get("blocking_power", 0.5)), 0.0, 1.0)

        obstruction = profile.get("obstruction_incentive")
        if obstruction is None:
            obstruction = max(0.0, profile["utility_no"] - profile["utility_yes"])
        profile["obstruction_incentive"] = round(float(obstruction), 3)

        resolving = profile.get("resolving_incentive")
        if resolving is None:
            resolving = max(0.0, profile["utility_yes"] - profile["utility_no"])
        profile["resolving_incentive"] = round(float(resolving), 3)

        silence = profile.get("structural_silence_score")
        if silence is None:
            presence = _clamp(float(profile.get("presence_in_signal_chain", 0.5)), 0.0, 1.0)
            silence = profile["obstruction_incentive"] * profile["blocking_power"] * (1.0 - presence)
        profile["structural_silence_score"] = round(float(silence), 3)

        profile["actor_class"] = _derive_actor_class(profile)
        profile["structural_contradiction"] = profile.get("structural_contradiction", "")
        profile["collapse_trigger"] = profile.get("collapse_trigger", "")
        profile["one_line_summary"] = profile.get("one_line_summary", "")
        normalized.append(profile)
    return normalized


def build_deterministic_game_model(actor_profiles, question):
    profiles = normalize_actor_profiles(actor_profiles)
    obstructors = [p for p in profiles if p["obstruction_incentive"] >= 1.0]
    resolvers = [p for p in profiles if p["resolving_incentive"] > 0.0]

    blocking_power = sum(
        p["blocking_power"] * min(1.0, p["obstruction_incentive"] / 5.0)
        for p in obstructors
    )
    resolving_power = sum(
        p["blocking_power"] * min(1.0, p["resolving_incentive"] / 5.0)
        for p in resolvers
    )
    blocking_power = round(min(0.99, blocking_power), 3)
    resolving_power = round(min(0.99, resolving_power), 3)

    total_ss = sum(p["structural_silence_score"] for p in profiles)
    adversarial_haircut = round(min(0.45, total_ss / (total_ss + 1.5)) if total_ss > 0 else 0.0, 3)

    if blocking_power >= resolving_power + 0.20:
        eq_state = "stable_disruption"
        eq_stability = "stable"
        eq_prob = 0.70
        dominant_scenario = "Blocking coalition dominates and duration remains rational for key actors."
    elif blocking_power > resolving_power:
        eq_state = "managed_disruption"
        eq_stability = "metastable"
        eq_prob = 0.55
        dominant_scenario = "Managed disruption persists while fence sitters preserve optionality."
    elif resolving_power >= blocking_power + 0.15:
        eq_state = "resolution_possible"
        eq_stability = "unstable"
        eq_prob = 0.40
        dominant_scenario = "Resolution pathway exists if resolvers stay aligned and contradictions do not re-open."
    else:
        eq_state = "deadlock"
        eq_stability = "metastable"
        eq_prob = 0.50
        dominant_scenario = "System remains balanced enough that small actor moves can swing the equilibrium."

    contradictions = []
    for profile in profiles:
        contradiction = profile.get("structural_contradiction", "").strip()
        if contradiction:
            contradictions.append(
                {
                    "actor": profile["actor"],
                    "contradiction": contradiction,
                    "collapse_trigger": profile.get("collapse_trigger", "Unknown"),
                    "collapse_probability_12m": round(min(0.35, 0.08 + profile["blocking_power"] * 0.2), 3),
                }
            )

    dominant_strategies = []
    for profile in profiles:
        if profile["utility_no"] >= 6.0 and profile["utility_yes"] >= 5.0:
            dominant_strategies.append(
                {
                    "actor": profile["actor"],
                    "strategy": "dominant_obstruct",
                    "reason": (
                        f"Wins materially in both outcomes: YES={profile['utility_yes']:.1f}, "
                        f"NO={profile['utility_no']:.1f}"
                    ),
                    "is_truly_dominant": True,
                }
            )

    profiles_sorted = sorted(
        profiles,
        key=lambda item: (
            item["structural_silence_score"],
            item["obstruction_incentive"] * item["blocking_power"],
        ),
        reverse=True,
    )
    top_drag = profiles_sorted[0]["actor"] if profiles_sorted else "No actor"

    return {
        "event": question,
        "payoff_matrix": {
            "resolution_YES": {p["actor"]: p["utility_yes"] for p in profiles},
            "resolution_NO": {p["actor"]: p["utility_no"] for p in profiles},
        },
        "dominant_strategies": dominant_strategies,
        "blocking_coalition": {
            "members": [p["actor"] for p in obstructors],
            "combined_blocking_power": blocking_power,
            "assessment": f"Blocking power {blocking_power:.2f}",
        },
        "resolving_coalition": {
            "members": [p["actor"] for p in resolvers],
            "combined_resolving_power": resolving_power,
            "assessment": f"Resolving power {resolving_power:.2f}",
        },
        "nash_equilibrium": {
            "state": eq_state,
            "stability": eq_stability,
            "description": dominant_scenario,
            "probability": eq_prob,
        },
        "structural_contradictions": contradictions,
        "adversarial_haircut_total": adversarial_haircut,
        "p_naive_adjustment_factor": round(1.0 - adversarial_haircut, 3),
        "dominant_scenario_2027": dominant_scenario,
        "dominant_scenario_probability": eq_prob,
        "key_insight": (
            f"Standard AI would over-weight visible commentary. Deterministic CAR identifies "
            f"{top_drag} as the strongest strategic drag through silence, payoff asymmetry, or both."
        ),
        "actors": profiles,
    }


def compute_final_car_probability(p_naive, game_model):
    p_naive = _clamp(float(p_naive), 0.02, 0.98)
    haircut = min(0.45, float(game_model.get("adversarial_haircut_total", 0.0)))
    contradiction_discount = min(
        0.15,
        sum(item.get("collapse_probability_12m", 0.0) * 0.1 for item in game_model.get("structural_contradictions", [])),
    )
    p_car = _clamp(p_naive * (1.0 - haircut) * (1.0 - contradiction_discount), 0.02, 0.98)

    state = game_model.get("nash_equilibrium", {}).get("state", "deadlock")
    ci_width = {
        "stable_disruption": 0.08,
        "managed_disruption": 0.12,
        "resolution_possible": 0.10,
        "deadlock": 0.14,
    }.get(state, 0.12)

    return {
        "p_naive": round(p_naive, 3),
        "adversarial_haircut": round(haircut, 3),
        "contradiction_discount": round(contradiction_discount, 3),
        "p_car": round(p_car, 3),
        "ci_lower": round(_clamp(p_car - ci_width, 0.02, 0.98), 3),
        "ci_upper": round(_clamp(p_car + ci_width, 0.02, 0.98), 3),
        "equilibrium_type": state.upper(),
        "dominant_scenario": game_model.get("dominant_scenario_2027", ""),
        "dominant_scenario_probability": game_model.get("dominant_scenario_probability", 0.0),
    }


def _neutralize_actor(profile):
    neutral = deepcopy(profile)
    midpoint = (neutral["utility_yes"] + neutral["utility_no"]) / 2.0
    neutral["utility_yes"] = midpoint
    neutral["utility_no"] = midpoint
    neutral["obstruction_incentive"] = 0.0
    neutral["resolving_incentive"] = 0.0
    neutral["structural_silence_score"] = 0.0
    neutral["structural_contradiction"] = ""
    neutral["collapse_trigger"] = ""
    return neutral


def compute_marginal_actor_impacts(question, p_naive, actor_profiles):
    base_model = build_deterministic_game_model(actor_profiles, question)
    base_final = compute_final_car_probability(p_naive, base_model)
    base_p = base_final["p_car"]
    impacts = []

    for index, profile in enumerate(normalize_actor_profiles(actor_profiles)):
        adjusted = normalize_actor_profiles(actor_profiles)
        adjusted[index] = _neutralize_actor(profile)
        alt_model = build_deterministic_game_model(adjusted, question)
        alt_final = compute_final_car_probability(p_naive, alt_model)
        impacts.append(
            {
                "actor": profile["actor"],
                "base_class": profile["actor_class"],
                "delta_p_car": round(alt_final["p_car"] - base_p, 3),
                "delta_haircut": round(base_final["adversarial_haircut"] - alt_final["adversarial_haircut"], 3),
                "base_structural_silence": profile["structural_silence_score"],
            }
        )

    impacts.sort(key=lambda item: item["delta_p_car"], reverse=True)
    return impacts


def run_deterministic_car(question, p_naive, actor_profiles):
    game_model = build_deterministic_game_model(actor_profiles, question)
    final = compute_final_car_probability(p_naive, game_model)
    marginal_impacts = compute_marginal_actor_impacts(question, p_naive, actor_profiles)

    return {
        "question": question,
        "actor_profiles": game_model["actors"],
        "game_model": game_model,
        "final": final,
        "marginal_impacts": marginal_impacts,
    }
