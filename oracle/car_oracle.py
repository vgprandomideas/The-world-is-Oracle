"""
CAR Oracle — The Full System

Wraps any LLM and forces Counterfactual Adversarial Reasoning
before outputting any probability estimate.

This is the demonstration of what "non-cringe AI" looks like:
  1. Never outputs a probability without asking "who benefits from NO?"
  2. Always detects structurally silent obstructors
  3. Always classifies supply signals as substitutional/additive/complementary
  4. Always checks for actor fracture before treating a country as unified

Guruprasad Venkatakrishnan (2026) — Verslan / predictmarkets.finance
"""

import json
import anthropic
from oracle.car import CARAnalysis

client = anthropic.Anthropic()

CAR_ORACLE_SYSTEM = """You are a CAR Oracle — a probability estimation system
that uses Counterfactual Adversarial Reasoning before every estimate.

You represent the opposite of how AI normally behaves. Standard AI:
1. Reads available signals
2. Averages them
3. Outputs a probability
4. Sounds confident
5. Misses the actor who benefits from being invisible

You instead:

BEFORE touching any signals, ask these questions IN ORDER:

Q1. WHO ARE ALL THE ACTORS — including those NOT generating signals?
    (Silence is the most important signal. Absence is evidence.)

Q2. FOR EACH ACTOR: what is their payoff if this resolves YES vs NO?
    Use financial flows, voting records, trade data.
    NEVER use stated positions. Watch what they do, not what they say.

Q3. WHO HAS OBSTRUCTION INCENTIVE?
    OI = max(0, U(NO) - U(YES))
    Flag every actor where being absent from the discussion helps them.

Q4. WHO IS STRUCTURALLY SILENT?
    High obstruction incentive + low presence in news = most dangerous
    The actor you never hear about is the one gaming your estimate.

Q5. IS EACH ACTOR INTERNALLY UNIFIED?
    Reliance Industries ≠ Indian Government ≠ Indian State Refiners
    A country is not one actor. A company is not one actor.
    Find the sub-actor whose incentive dominates.

Q6. FOR SUPPLY SIGNALS: is this substitutional, additive, or complementary?
    Venezuela → India is ADDITIVE (India buys Russian AND Venezuelan)
    Russian oil → India is COMPLEMENTARY TO DISRUPTION (profitable BECAUSE of crisis)
    Only substitutional supply reduces resolution urgency.

THEN AND ONLY THEN:
- Apply adversarial haircut to naive estimate
- Output structured JSON with full reasoning chain

Your output format must ALWAYS be:
{
  "step1_actors": [...],
  "step2_payoffs": {...},
  "step3_obstructors": [...],
  "step4_structural_silence": {...},
  "step5_fractures": {...},
  "step6_supply_classification": {...},
  "naive_probability": 0.XX,
  "adversarial_haircut": 0.XX,
  "car_probability": 0.XX,
  "dominant_insight": "The one thing standard AI would miss"
}"""


def naive_estimate(event: str, context: str) -> dict:
    """Standard AI response — no adversarial reasoning."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system="You estimate probabilities for geopolitical events. Be concise.",
        messages=[{
            "role": "user",
            "content": f"What is the probability this event resolves YES?\n\nEvent: {event}\n\nContext: {context}\n\nGive a probability and brief reasoning."
        }]
    )
    return {
        "method": "NAIVE",
        "response": response.content[0].text,
        "adversarial_check": "NONE PERFORMED",
        "quality": "Standard AI — misses structural obstructors"
    }


def car_estimate(event: str, context: str, actors_hint: str = "") -> dict:
    """CAR Oracle response — full adversarial reasoning."""
    user_content = f"""Event: {event}

Context: {context}"""

    if actors_hint:
        user_content += f"\n\nKnown actors to consider: {actors_hint}"

    user_content += "\n\nApply full CAR protocol. Output valid JSON only."

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=CAR_ORACLE_SYSTEM,
        messages=[{"role": "user", "content": user_content}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        parsed = json.loads(raw)
        parsed["method"] = "CAR"
        parsed["quality"] = "Adversarial reasoning applied"
        return parsed
    except Exception:
        return {"method": "CAR", "raw": raw, "parse_error": True}


def run_comparison(event: str, context: str, actors_hint: str = ""):
    """
    Side-by-side comparison: naive AI vs CAR Oracle.
    This is the core demonstration.
    """
    print("=" * 70)
    print(f"EVENT: {event}")
    print("=" * 70)

    print("\n⚠️  NAIVE AI RESPONSE (standard LLM):")
    print("-" * 40)
    naive = naive_estimate(event, context)
    print(naive["response"])
    print(f"\n[Adversarial check: {naive['adversarial_check']}]")

    print("\n\n✅ CAR ORACLE RESPONSE (adversarial reasoning):")
    print("-" * 40)
    car = car_estimate(event, context, actors_hint)

    if "parse_error" not in car:
        print(f"Naive P: {car.get('naive_probability', '?')}")
        print(f"Adversarial Haircut: {car.get('adversarial_haircut', '?')}")
        print(f"CAR P: {car.get('car_probability', '?')}")
        print(f"\nDominant Insight: {car.get('dominant_insight', '')}")

        if "step3_obstructors" in car:
            print(f"\nObstructing actors: {car['step3_obstructors']}")

        if "step4_structural_silence" in car:
            ss = car["step4_structural_silence"]
            if isinstance(ss, dict):
                print(f"Structurally silent: {ss}")
    else:
        print(car.get("raw", "Parse error"))

    print("\n" + "=" * 70)
    return naive, car


if __name__ == "__main__":
    print("CAR ORACLE — DEMONSTRATION")
    print("Counterfactual Adversarial Reasoning")
    print("Guruprasad Venkatakrishnan (2026)")
    print()

    # Test 1: Hormuz — high CAR complexity
    run_comparison(
        event="Strait of Hormuz returns to full commercial operation by Dec 2027",
        context=(
            "US-Iran ceasefire announced April 8 2026. Maersk transited under "
            "military escort May 2026. India conducting bilateral diplomacy. "
            "Pakistan mediating Islamabad talks. Iran agreed to ceasefire. "
            "Traffic still at 5% of pre-crisis levels."
        ),
        actors_hint=(
            "Russia, India (Reliance vs Government vs State Refiners), "
            "China, United States, Iran, Venezuela (supply)"
        )
    )

    print("\n\n")

    # Test 2: OPEC cut — medium CAR complexity
    run_comparison(
        event="OPEC+ agrees to maintain current production cuts through 2027",
        context=(
            "Saudi Arabia pushing for extended cuts. Russia participating in OPEC+. "
            "US shale producers at record output. Global demand growth slowing. "
            "Iraq and UAE have repeatedly exceeded quotas."
        ),
        actors_hint="Saudi Arabia, Russia, UAE, Iraq, US shale producers, China (demand)"
    )
