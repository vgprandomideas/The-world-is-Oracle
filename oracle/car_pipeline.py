"""
CAR Pipeline — The Complete Automated System

Takes any geopolitical question and returns a CAR-grounded
probability estimate by automatically:

1. Extracting actors from the question
2. Fetching their historical actions
3. Building behavioural fingerprints (9 CAR questions)
4. Constructing payoff matrix and game theory model
5. Detecting equilibrium type
6. Computing P_CAR with adversarial haircut

This is the system that teaches AI to ask adversarial questions
AUTOMATICALLY — without a human injecting them.

Usage:
  result = run_car_pipeline(
      question="Will the Strait of Hormuz return to full commercial
                operation by December 2027?",
      known_actors=["Russia", "China", "Pakistan", "India",
                    "United States", "Iran", "Israel", "Houthis"],
      p_naive=0.52,
  )

Guruprasad Venkatakrishnan (2026)
Verslan / predictmarkets.finance / verslan.xyz
"""

import json
import time
import anthropic
from oracle.car_actor_profiler import (
    profile_actor, CAR_QUESTIONS, PAKISTAN_VERIFIED_ACTIONS
)
from oracle.car_game_theory import (
    construct_payoff_matrix, detect_equilibrium_type,
    compute_final_car_probability
)

client = anthropic.Anthropic()

# ── Stage 1: Actor Action Extraction ─────────────────────────────────────────

ACTION_EXTRACTOR_SYSTEM = """You are a CAR Action Extractor.

Given a geopolitical actor and event context, extract their
VERIFIED HISTORICAL ACTIONS — not statements or opinions.

CRITICAL RULES:
1. Actions only: military deployments, treaty signings, votes,
   financial flows, sanctions, trade deals, coups, elections
2. NOT statements: speeches, press releases, interviews
3. NOT opinions: analyst commentary, media characterisation
4. Include DATES and SOURCES for every action
5. Include actions that CONTRADICT the actor's stated position —
   these are the most valuable for CAR

The goal: build the action record that reveals the actor's TRUE
utility function — what they actually want vs what they say they want.

Output valid JSON only."""


def extract_actor_actions(
    actor: str,
    event_context: str,
    years_back: int = 10,
) -> list:
    """
    Extract verified historical actions for an actor.
    In production: connects to GDELT, UN voting records,
    financial flow databases, SIPRI arms data.
    For now: uses Claude's knowledge with strict action-only filter.
    """
    prompt = f"""Actor: {actor}
Event context: {event_context}
Time period: Last {years_back} years

Extract their verified ACTIONS (not statements) relevant to
understanding their payoff structure for this event.

Focus on:
- Military deployments and pacts
- Financial transactions (who they pay, who pays them)
- Votes at UN Security Council or General Assembly
- Economic sanctions applied or endured
- Treaty signings and violations
- Actions that CONTRADICT their stated position

Return JSON array:
[
  {{
    "date": "YYYY-MM-DD",
    "type": "military|financial|treaty|vote|economic|political",
    "action": "Specific verifiable action description",
    "source": "Institutional source (UN, IMF, DoD, etc.)",
    "contradicts_stated_position": true,
    "reveals": "What this action reveals about their true utility function"
  }}
]

Maximum 15 most important actions. Verified only."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=ACTION_EXTRACTOR_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except Exception:
        return []


# ── Stage 2: P_naive Estimator ────────────────────────────────────────────────

def estimate_naive_probability(question: str, context: str) -> float:
    """
    Standard AI estimate — NO adversarial reasoning.
    This is the baseline that CAR will improve on.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        system="You estimate geopolitical probabilities. Be concise. Output JSON only: {\"probability\": 0.XX, \"reasoning\": \"...\"}",
        messages=[{"role": "user", "content": f"Question: {question}\nContext: {context}\nEstimate the probability this resolves YES."}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        data = json.loads(raw)
        return float(data.get("probability", 0.5))
    except Exception:
        return 0.5


# ── Stage 3: Full CAR Pipeline ────────────────────────────────────────────────

def run_car_pipeline(
    question: str,
    known_actors: list,
    p_naive: float = None,
    context: str = "",
    use_prebuilt_profiles: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Full automated CAR pipeline.

    question: The geopolitical probability question
    known_actors: List of actor names to profile
    p_naive: Optional pre-computed naive estimate
             (if None, system computes it automatically)
    context: Additional context about the event
    """

    if verbose:
        print(f"\n{'='*65}")
        print(f"CAR PIPELINE")
        print(f"Question: {question}")
        print(f"{'='*65}")

    # Stage 1: Naive estimate
    if p_naive is None:
        if verbose:
            print("\n[Stage 1] Computing naive probability (no adversarial reasoning)...")
        p_naive = estimate_naive_probability(question, context)
        if verbose:
            print(f"  Naive P = {p_naive:.1%} (standard AI — no CAR applied)")
    else:
        if verbose:
            print(f"\n[Stage 1] Naive P = {p_naive:.1%} (provided)")

    # Stage 2: Extract actions + build profiles for each actor
    if verbose:
        print(f"\n[Stage 2] Building actor profiles from historical actions...")

    actor_profiles = []

    # Prebuilt profiles for speed (production: all from real-time data)
    prebuilt = {
        "Pakistan": {
            "actor": "Pakistan",
            "actor_class": "nuclear_rentier",
            "utility_yes": 4.5,
            "utility_no": 5.5,
            "obstruction_incentive": 1.0,
            "blocking_power": 0.72,
            "structural_silence_score": 0.35,
            "adversarial_haircut_contribution": 0.08,
            "car_questions": {
                "Q4_multi_party_selling": {
                    "structural_contradiction": "Saudi nuclear shield + Iran mediator simultaneously — incompatible when exposed"
                },
                "Q8_collapse_trigger": {
                    "trigger": "Iran discovers full extent of Saudi nuclear pact",
                    "proximity": "near"
                },
                "Q9_india_equivalent": {
                    "comparator": "India",
                    "envy_blocking_factor": 0.85,
                    "evidence": "Blocks India-led solutions in Islamic world forums; inserts self whenever India might get credit"
                }
            },
            "one_line_summary": "Broken nuclear rentier selling deterrence to Saudi and backchannel to Iran simultaneously, driven by India envy"
        },
        "Russia": {
            "actor": "Russia",
            "actor_class": "obstructor",
            "utility_yes": 3.0,
            "utility_no": 9.5,
            "obstruction_incentive": 6.5,
            "blocking_power": 0.85,
            "structural_silence_score": 4.52,
            "adversarial_haircut_contribution": 0.45,
            "car_questions": {
                "Q3_who_benefits_from_no": {
                    "finding": "Earns $150M/day extra from Hormuz closure. Oil price: $44→$100",
                    "obstruction_incentive": 6.5,
                    "presence_in_signal_chain": 0.08,
                    "structural_silence_score": 4.52
                }
            },
            "one_line_summary": "Dominant obstructor earning $150M/day from disruption while saying nothing publicly"
        },
        "China": {
            "actor": "China",
            "actor_class": "fence_sitter",
            "utility_yes": 6.5,
            "utility_no": 6.0,
            "obstruction_incentive": 0.0,
            "blocking_power": 0.85,
            "structural_silence_score": 0.0,
            "adversarial_haircut_contribution": 0.0,
            "car_questions": {
                "Q6_sovereignty_check": {
                    "sovereignty_quotient": 0.90,
                    "is_captured_state": False
                }
            },
            "one_line_summary": "High-power fence sitter: calls for Hormuz opening while blocking UN mechanism and enjoying non-hostile oil access"
        },
        "Houthis": {
            "actor": "Houthis",
            "actor_class": "vendetta_architect",
            "utility_yes": 3.0,
            "utility_no": 7.0,
            "obstruction_incentive": 4.0,
            "blocking_power": 0.70,
            "structural_silence_score": 2.1,
            "adversarial_haircut_contribution": 0.15,
            "car_questions": {
                "Q7_shakuni_check": {
                    "vendetta_score": 5.99,
                    "historical_grievance": "Yemen war 2015-2025 — Saudi bombing, blockade, starvation. Thousands dead.",
                    "martyrdom_acceptance": 0.95,
                    "is_vendetta_architect": True
                }
            },
            "one_line_summary": "Vendetta architect — ceasefire-proof, dice made from Yemen's dead, attacks regardless of Iran-US agreements"
        },
    }

    for actor in known_actors:
        if verbose:
            print(f"  Profiling: {actor}...")

        if use_prebuilt_profiles and actor in prebuilt:
            profile = prebuilt[actor]
            if verbose:
                print(f"    Class: {profile['actor_class']}")
                print(f"    OI: {profile['obstruction_incentive']:.1f}  BP: {profile['blocking_power']:.2f}  SS: {profile['structural_silence_score']:.2f}")
        else:
            # Live profiling from actions
            actions = extract_actor_actions(actor, question + " " + context)
            time.sleep(0.5)
            profile = profile_actor(actor, question, actions)
            if verbose and "error" not in profile:
                print(f"    Class: {profile.get('actor_class', 'unknown')}")

        actor_profiles.append(profile)

    # Stage 3: Game theory model
    if verbose:
        print(f"\n[Stage 3] Constructing game theory model...")
    game_model = construct_payoff_matrix(actor_profiles, question)

    if verbose and "error" not in game_model:
        eq = game_model.get("nash_equilibrium", {})
        print(f"  Nash Equilibrium: {eq.get('state', '?')} ({eq.get('stability', '?')})")
        dominant = [d["actor"] for d in game_model.get("dominant_strategies", [])
                   if d.get("is_truly_dominant")]
        if dominant:
            print(f"  Dominant strategies: {dominant}")
        blocking = game_model.get("blocking_coalition", {})
        resolving = game_model.get("resolving_coalition", {})
        print(f"  Blocking coalition power: {blocking.get('combined_blocking_power', 0):.2f}")
        print(f"  Resolving coalition power: {resolving.get('combined_resolving_power', 0):.2f}")

    # Stage 4: Final CAR probability
    if verbose:
        print(f"\n[Stage 4] Computing P_CAR...")

    final = compute_final_car_probability(p_naive, game_model)

    if verbose:
        print(f"\n{'='*65}")
        print(f"RESULT")
        print(f"{'='*65}")
        print(f"Naive P:              {final['p_naive']:.1%}  (standard AI)")
        print(f"Adversarial Haircut:  {final['adversarial_haircut']:.1%}")
        print(f"Contradiction Disc:   {final['contradiction_discount']:.1%}")
        print(f"P_CAR:                {final['p_car']:.1%}  (with adversarial reasoning)")
        print(f"CI:                   [{final['ci_lower']:.1%}, {final['ci_upper']:.1%}]")
        print(f"Equilibrium:          {final['equilibrium_type']}")
        print(f"\nDominant 2027 scenario ({final['dominant_scenario_probability']:.0%}):")
        print(f"  {final['dominant_scenario']}")

        if "key_insight" in game_model:
            print(f"\nKey insight standard AI misses:")
            print(f"  {game_model['key_insight']}")

    return {
        "question": question,
        "actor_profiles": actor_profiles,
        "game_model": game_model,
        "final": final,
    }


# ── Training Data Generator ────────────────────────────────────────────────────

GEOPOLITICAL_QUESTIONS = [
    {
        "question": "Will the Strait of Hormuz return to full commercial operation by December 2027?",
        "actors": ["Russia", "China", "Pakistan", "India", "United States", "Iran", "Israel", "Houthis", "Qatar"],
        "p_naive": 0.52,
    },
    {
        "question": "Will OPEC+ maintain current production cuts through 2027?",
        "actors": ["Saudi Arabia", "Russia", "UAE", "Iraq", "United States"],
        "p_naive": 0.55,
    },
    {
        "question": "Will India and Pakistan normalise trade relations within 5 years?",
        "actors": ["India", "Pakistan", "China", "United States"],
        "p_naive": 0.25,
    },
    {
        "question": "Will China reunify Taiwan by force before 2030?",
        "actors": ["China", "United States", "Taiwan", "Japan", "Russia"],
        "p_naive": 0.15,
    },
    {
        "question": "Will a permanent Gaza ceasefire hold through 2027?",
        "actors": ["Israel", "Hamas", "United States", "Egypt", "Qatar", "Iran"],
        "p_naive": 0.35,
    },
]


def generate_car_training_batch(n: int = 5) -> list:
    """
    Generate CAR training data from geopolitical questions.
    Returns (naive_response, car_response, preference: car) triples.
    """
    dataset = []

    for i, q_data in enumerate(GEOPOLITICAL_QUESTIONS[:n]):
        print(f"\n[{i+1}/{n}] Processing: {q_data['question'][:60]}...")

        result = run_car_pipeline(
            question=q_data["question"],
            known_actors=q_data["actors"],
            p_naive=q_data["p_naive"],
            use_prebuilt_profiles=True,
            verbose=False,
        )

        dataset.append({
            "prompt": q_data["question"],
            "naive_response": {
                "probability": q_data["p_naive"],
                "method": "standard_ai",
                "adversarial_check": "NOT PERFORMED",
                "quality": "REJECTED"
            },
            "car_response": {
                "probability": result["final"]["p_car"],
                "ci": [result["final"]["ci_lower"], result["final"]["ci_upper"]],
                "equilibrium": result["final"]["equilibrium_type"],
                "dominant_scenario": result["final"]["dominant_scenario"],
                "key_insight": result["game_model"].get("key_insight", ""),
                "method": "CAR_pipeline",
                "quality": "PREFERRED"
            },
            "preference": "car_response",
            "delta": round(q_data["p_naive"] - result["final"]["p_car"], 3),
            "actors_analysed": q_data["actors"],
        })
        print(f"  Naive: {q_data['p_naive']:.1%} → CAR: {result['final']['p_car']:.1%}")

    return dataset


if __name__ == "__main__":
    import os

    print("CAR PIPELINE — FULL DEMONSTRATION")
    print("Guruprasad Venkatakrishnan (2026)")
    print("Verslan / predictmarkets.finance / verslan.xyz")

    # Full Hormuz demonstration
    result = run_car_pipeline(
        question="Will the Strait of Hormuz return to full commercial operation by December 2027?",
        known_actors=["Russia", "China", "Pakistan", "India",
                      "United States", "Iran", "Houthis"],
        p_naive=0.52,
        context=(
            "US-Iran ceasefire April 2026. Islamabad Talks failed April 11-12. "
            "US naval blockade of Iran from April 13. "
            "Maersk suspended Hormuz transits. 600+ vessels stranded. "
            "Pakistan deployed jets to Saudi Arabia. "
            "India buying Russian crude regardless of US sanctions."
        ),
        verbose=True,
    )

    # Save training data
    print("\n\nGenerating training dataset...")
    dataset = generate_car_training_batch(n=3)

    os.makedirs("training_data", exist_ok=True)
    with open("training_data/car_pipeline_v1.jsonl", "w") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")

    print(f"\nSaved {len(dataset)} training examples")
    print("\nSummary:")
    for ex in dataset:
        print(f"  {ex['prompt'][:55]}...")
        print(f"    {ex['naive_response']['probability']:.1%} → {ex['car_response']['probability']:.1%}  (Δ={ex['delta']:+.2f})")
