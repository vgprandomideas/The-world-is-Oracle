"""
CAR Training Data Generator

Generates synthetic training examples teaching AI to:
1. Ask "who benefits from this NOT resolving?" before estimating probability
2. Detect structurally silent obstructors
3. Apply adversarial haircut to naive estimates
4. Decompose fractured actors

Each training example is a (prompt, naive_response, CAR_response) triple.
The CAR response is always superior — this is the preference signal for RLHF.

Two generation modes:
  SYNTHETIC  — Claude generates novel geopolitical scenarios
  EMPIRICAL  — Use verified real-world cases (Hormuz, Adani, etc.)

Guruprasad Venkatakrishnan (2026) — Verslan / predictmarkets.finance
"""

import json
import time
import anthropic
from oracle.car import (
    CARAnalysis, Actor, BlockingMechanism, SupplyType
)

client = anthropic.Anthropic()

# ── Empirical seed cases ─────────────────────────────────────────────────────

def build_hormuz_car() -> CARAnalysis:
    """
    Hormuz 2026 — the founding empirical case for CAR.
    Verified through: Kpler, Washington Institute, UK Parliament,
    FT Podcast, CNBC, Business Standard.
    """
    analysis = CARAnalysis(
        event_id="hormuz_normalisation_2027",
        event_description=(
            "Strait of Hormuz returns to full commercial operation "
            "by December 31, 2027"
        ),
        resolution_criteria=(
            "YES: Daily transits exceed 100 vessels (pre-crisis baseline: 120-140). "
            "NO: Transits remain below 50/day or subject to Iranian control."
        ),
    )

    # Russia — the dominant structurally silent obstructor
    russia = Actor(
        name="Russia",
        actor_type="state",
        utility_yes=3.0,   # Still sells oil to India at discount — ok
        utility_no=9.5,    # $150M/day windfall, Ukraine war funded, sanctions relief
        blocking_mechanisms=[
            BlockingMechanism.VETO_POWER,       # UN SC permanent member
            BlockingMechanism.RESOURCE_CONTROL, # Alternative oil supplier
            BlockingMechanism.PROXY_INFLUENCE,  # Historical Iran relationship
            BlockingMechanism.INFO_CONTROL,     # State media
        ],
        presence_in_signal_chain=0.08,  # Almost never mentioned in Hormuz coverage
        description=(
            "Russia earns $150M/day extra from Hormuz closure. "
            "Co-vetoed UN resolution. Oil price: $44 → $100. "
            "Dominant strategy: wins in both disruption AND normalcy "
            "because India confirmed buying Russian regardless. "
            "STRUCTURALLY SILENT — publishes nothing about Hormuz "
            "while being the primary financial beneficiary."
        ),
    )

    # India — fractured actor
    india_reliance = Actor(
        name="India_Reliance",
        actor_type="corporate",
        utility_yes=8.0,   # Gulf oil cheap, US business safe
        utility_no=3.0,    # Sanctioned if buys Russian, export disrupted
        blocking_mechanisms=[BlockingMechanism.RESOURCE_CONTROL],
        presence_in_signal_chain=0.3,
        description="Chickens out Day 1. US business > Russian oil.",
    )
    india_state_refiners = Actor(
        name="India_StateRefiners",
        actor_type="state_enterprise",
        utility_yes=7.0,   # Cheap Gulf crude
        utility_no=6.0,    # Russian crude at discount — still ok
        blocking_mechanisms=[BlockingMechanism.FINANCIAL_FLOW],
        presence_in_signal_chain=0.2,
        description="Follow government direction. Absorb Reliance's gap.",
    )
    india_government = Actor(
        name="India_Government",
        actor_type="state",
        utility_yes=7.5,   # Gulf crude cheap, energy security
        utility_no=6.0,    # Russian discount maintained, sovereignty asserted
        blocking_mechanisms=[
            BlockingMechanism.PROXY_INFLUENCE,
            BlockingMechanism.FINANCIAL_FLOW,
        ],
        presence_in_signal_chain=0.7,
        description=(
            "Modi: before waiver, during waiver, now also. "
            "Jaishankar got ships through bilaterally. "
            "BUT: credibility with Iran reduced if seen as US-aligned."
        ),
    )

    india = Actor(
        name="India",
        actor_type="state",
        utility_yes=7.5,
        utility_no=5.5,
        blocking_mechanisms=[
            BlockingMechanism.PROXY_INFLUENCE,
            BlockingMechanism.FINANCIAL_FLOW,
        ],
        presence_in_signal_chain=0.75,
        sub_actors=[india_reliance, india_state_refiners, india_government],
        description="Fractured. Private sector complies with US. State maintains Russia.",
    )

    # China — paradoxical actor
    china = Actor(
        name="China",
        actor_type="state",
        utility_yes=6.5,   # Gulf trade normalised
        utility_no=6.0,    # Non-hostile access + Russian crude = adequate
        blocking_mechanisms=[
            BlockingMechanism.VETO_POWER,
            BlockingMechanism.FINANCIAL_FLOW,
        ],
        presence_in_signal_chain=0.5,
        description=(
            "Vetoed UN resolution. Has non-hostile access already. "
            "Less urgent than appears — workaround exists."
        ),
    )

    # US — credibility-compromised actor
    us = Actor(
        name="United_States",
        actor_type="state",
        utility_yes=8.0,   # Claims victory, energy prices fall
        utility_no=3.0,    # Crisis continues, allies angry, oil prices high
        blocking_mechanisms=[
            BlockingMechanism.MILITARY,
            BlockingMechanism.RESOURCE_CONTROL,
        ],
        presence_in_signal_chain=0.90,
        description=(
            "Created the crisis by striking Iran. "
            "Now begging India/China to help reopen what US military closed. "
            "Lost credibility with Iran — Tier 1 source, contextual discount 0.40."
        ),
    )

    # Iran — control actor
    iran = Actor(
        name="Iran",
        actor_type="state",
        utility_yes=4.0,   # Loses Hormuz leverage, sanctions continue
        utility_no=7.0,    # Hormuz as bargaining chip, survival pressure
        blocking_mechanisms=[
            BlockingMechanism.MILITARY,
            BlockingMechanism.RESOURCE_CONTROL,  # Controls the strait physically
        ],
        presence_in_signal_chain=0.85,
        description="Controls the physical strait. New Khamenei = unpredictable.",
    )

    # Venezuela — supply type analysis
    venezuela_supply = {
        "actor": "Venezuela",
        "supply_type": SupplyType.ADDITIVE,
        "volumes_bpd": 300000,
        "india_context": (
            "India buys Venezuelan AND Russian. "
            "Venezuelan is ADDITIVE not substitutional — "
            "confirmed by India petroleum ministry May 18 2026. "
            "Does NOT reduce India's Hormuz urgency."
        ),
        "oracle_direction": 0.0,   # Neutral — additive, not substitutional
    }

    analysis.actors = [russia, india, china, us, iran]
    analysis.supply_signals = [venezuela_supply]

    return analysis.run()


def build_fed_rate_car() -> CARAnalysis:
    """
    Fed rate decision — LOW CAR complexity.
    Shows oracle behaves correctly when no structural obstructors exist.
    """
    analysis = CARAnalysis(
        event_id="fed_rate_jun2023",
        event_description="Federal Reserve raises rates at June 14, 2023 FOMC",
        resolution_criteria="YES: Fed funds rate increases 25bps+. NO: Hold or cut.",
    )

    fed = Actor(
        name="Federal_Reserve",
        actor_type="institutional",
        utility_yes=6.0,
        utility_no=5.0,
        blocking_mechanisms=[],
        presence_in_signal_chain=0.95,
        description="Primary source. No obstruction incentive.",
    )
    markets = Actor(
        name="Financial_Markets",
        actor_type="collective",
        utility_yes=5.5,
        utility_no=5.5,
        blocking_mechanisms=[],
        presence_in_signal_chain=0.80,
        description="Neutral — both outcomes have winners and losers.",
    )

    analysis.actors = [fed, markets]
    return analysis.run()


def build_adani_car() -> CARAnalysis:
    """
    Adani US action — MEDIUM CAR complexity.
    Short sellers benefit from sustained narrative — partial obstruction.
    """
    analysis = CARAnalysis(
        event_id="adani_us_severe_action",
        event_description="US takes severe action against Adani Group",
        resolution_criteria="YES: DOJ conviction or SEC operational restriction. NO: Dropped/settled.",
    )

    doj = Actor(
        name="US_DOJ",
        actor_type="institutional",
        utility_yes=7.0,
        utility_no=4.0,
        blocking_mechanisms=[BlockingMechanism.VETO_POWER],
        presence_in_signal_chain=0.90,
        description="Primary actor. No obstruction — wants resolution.",
    )
    hindenburg = Actor(
        name="Hindenburg_Research",
        actor_type="corporate",
        utility_yes=8.0,
        utility_no=7.0,   # Sustained narrative = media attention = fund raising
        blocking_mechanisms=[BlockingMechanism.INFO_CONTROL],
        presence_in_signal_chain=0.70,
        description=(
            "Short position. Benefits from sustained uncertainty. "
            "But benefit from YES (short pays off) > NO (narrative sustains). "
            "Mild obstruction incentive — prefers prolonged uncertainty over quick NO."
        ),
    )
    gqg = Actor(
        name="GQG_Partners",
        actor_type="corporate",
        utility_yes=2.0,   # Long position destroyed if YES
        utility_no=9.0,    # Long position intact, vindicated
        blocking_mechanisms=[BlockingMechanism.FINANCIAL_FLOW],
        presence_in_signal_chain=0.40,
        description=(
            "$1.87B long position. Strong obstruction incentive. "
            "CAPITAL FLOW SIGNAL — real money against YES resolution."
        ),
    )

    analysis.actors = [doj, hindenburg, gqg]
    return analysis.run()


# ── Training Data Generator ──────────────────────────────────────────────────

CAR_SYSTEM_PROMPT = """You are a Counterfactual Adversarial Reasoning (CAR) oracle.

Before estimating ANY probability for ANY event, you MUST execute
the following 7-step protocol. This is non-negotiable. Skipping steps
produces the "cringe AI" response that confidently states naive probabilities
without asking who benefits from the opposite outcome.

STEP 1 — ACTOR ENUMERATION
List EVERY actor with material interest in this event.
Include actors who are NOT generating news signals.
Silence is data. Absence is evidence.

STEP 2 — PAYOFF MAPPING
For each actor, estimate:
  U(YES): their utility if event resolves YES [0-10]
  U(NO): their utility if event resolves NO [0-10]
Use financial flows, voting records, trade data.
NOT stated positions — those are noise.
NEVER trust what an actor says. Watch what they do.

STEP 3 — OBSTRUCTION DETECTION
Flag every actor where U(NO) > U(YES).
Obstruction Incentive: OI = max(0, U(NO) - U(YES))
These actors have structural reasons to prevent resolution.

STEP 4 — STRUCTURAL SILENCE AUDIT
For each flagged actor:
Ask: Are they present in the information signal chain?
Structural Silence Score: SS = OI × Blocking_Power × (1 - Presence)
HIGH SS = most dangerous actor for the oracle.
The actor most absent from signals while having strongest obstruction
incentive is the one standard AI completely misses.

STEP 5 — ACTOR FRACTURE ANALYSIS
Ask: Is this actor internally unified?
Reliance Industries ≠ Indian Government ≠ Indian State Refiners
Each has different payoffs. Model them separately.
The dominant sub-actor's incentive determines the composite signal.

STEP 6 — ADVERSARIAL HAIRCUT
Apply: AH = f(total structural silence scores)
P_adjusted = P_naive × (1 - AH)
This is not pessimism. It is calibration.

STEP 7 — SUPPLY SIGNAL CLASSIFICATION
For any supply-side signal, classify:
  SUBSTITUTIONAL: reduces urgency for resolution
  ADDITIVE: no urgency change
  COMPLEMENTARY TO DISRUPTION: actor profits from disruption itself
Only substitutional supply reduces resolution probability.

Only after completing all 7 steps: output your probability estimate.
"""

NAIVE_SYSTEM_PROMPT = """You are a probability estimation assistant.
Given an event, estimate the probability it will resolve YES.
Consider available evidence and provide your best estimate."""


def generate_comparison(event_description: str, resolution_criteria: str,
                         naive_estimate: float, car_estimate: float,
                         car_report: dict) -> dict:
    """
    Generate a training example comparing naive vs CAR response.
    This is the preference pair for RLHF training.
    """
    return {
        "prompt": f"Event: {event_description}\nResolution: {resolution_criteria}",
        "naive_response": {
            "probability": naive_estimate,
            "reasoning": "Based on available signals and recent developments.",
            "adversarial_check": "NOT PERFORMED",
            "quality": "REJECTED — cringe AI response",
        },
        "car_response": {
            "probability": car_estimate,
            "car_report": car_report,
            "quality": "PREFERRED — adversarial reasoning applied",
        },
        "preference": "car_response",
        "preference_reason": (
            f"CAR identified {len([a for a in car_report['step1_actors'] if a['obstruction_incentive'] > 0])} "
            f"obstructing actors. Naive estimate ignored structural silence. "
            f"Adversarial haircut: {car_report['step6_adversarial_haircut']:.1%}"
        ),
    }


def generate_synthetic_scenario(topic: str) -> dict:
    """
    Use Claude to generate a novel geopolitical scenario
    and its CAR analysis — for training data diversity.
    """
    prompt = f"""Generate a geopolitical probability estimation scenario about: {topic}

Output ONLY valid JSON with this structure:
{{
  "event_description": "...",
  "resolution_criteria": "...",
  "actors": [
    {{
      "name": "...",
      "actor_type": "state|corporate|institutional|multilateral",
      "utility_yes": 7.0,
      "utility_no": 4.0,
      "blocking_mechanisms": ["veto_power", "resource_control"],
      "presence_in_signal_chain": 0.8,
      "description": "Why this actor has these payoffs"
    }}
  ],
  "naive_probability": 0.55,
  "car_adjusted_probability": 0.38,
  "key_insight": "The one thing standard AI would miss about this scenario"
}}

Make it realistic. Include at least one structurally silent obstructor.
blocking_mechanisms can be: veto_power, resource_control, proxy_influence,
info_control, financial_flow, military"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            system=(
                "You generate training data for Counterfactual Adversarial Reasoning. "
                "Output ONLY valid JSON. No preamble, no markdown fences."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        print(f"Generation error: {e}")
        return None


# ── Dataset Builder ──────────────────────────────────────────────────────────

SYNTHETIC_TOPICS = [
    "OPEC production cut negotiations",
    "Taiwan Strait military tensions",
    "Central bank interest rate decision during election year",
    "Sovereign debt restructuring",
    "Nuclear arms treaty renewal",
    "Technology export controls between US and China",
    "Gaza ceasefire negotiations",
    "Currency peg defence by emerging market central bank",
    "Climate treaty ratification",
    "EU energy embargo on authoritarian state",
    "Corporate merger requiring antitrust approval",
    "Cryptocurrency exchange regulatory shutdown",
    "Trade union strike at critical infrastructure",
    "Pharmaceutical drug approval for major disease",
    "Water rights treaty between neighbouring countries",
    "Arctic shipping route sovereignty dispute",
    "Election integrity dispute in emerging democracy",
    "State-owned enterprise privatisation",
    "Cross-border pipeline construction approval",
    "Sanctions relief negotiation with pariah state",
]


def build_training_dataset(n_synthetic: int = 20) -> list:
    """
    Build the full CAR training dataset.
    Returns list of (prompt, rejected_response, preferred_response) triples.
    """
    dataset = []

    # 1. Empirical cases — ground truth
    print("Building empirical cases...")

    # Hormuz — full CAR complexity
    hormuz_car = build_hormuz_car()
    hormuz_data = generate_comparison(
        event_description=hormuz_car.event_description,
        resolution_criteria=hormuz_car.resolution_criteria,
        naive_estimate=0.52,
        car_estimate=hormuz_car.apply_to_probability(0.52),
        car_report=hormuz_car.report(),
    )
    dataset.append(hormuz_data)
    print(f"  ✓ Hormuz — AH: {hormuz_car.adversarial_haircut:.1%}, "
          f"P: 52% → {hormuz_car.apply_to_probability(0.52):.1%}")

    # Adani — medium complexity
    adani_car = build_adani_car()
    adani_data = generate_comparison(
        event_description=adani_car.event_description,
        resolution_criteria=adani_car.resolution_criteria,
        naive_estimate=0.72,
        car_estimate=adani_car.apply_to_probability(0.72),
        car_report=adani_car.report(),
    )
    dataset.append(adani_data)
    print(f"  ✓ Adani — AH: {adani_car.adversarial_haircut:.1%}, "
          f"P: 72% → {adani_car.apply_to_probability(0.72):.1%}")

    # Fed rate — near-zero CAR complexity
    fed_car = build_fed_rate_car()
    fed_data = generate_comparison(
        event_description=fed_car.event_description,
        resolution_criteria=fed_car.resolution_criteria,
        naive_estimate=0.52,
        car_estimate=fed_car.apply_to_probability(0.52),
        car_report=fed_car.report(),
    )
    dataset.append(fed_data)
    print(f"  ✓ Fed Rate — AH: {fed_car.adversarial_haircut:.1%}, "
          f"P: 52% → {fed_car.apply_to_probability(0.52):.1%} (near-unchanged, correct)")

    # 2. Synthetic cases — diversity
    print(f"\nGenerating {n_synthetic} synthetic scenarios...")

    import random
    topics = random.sample(SYNTHETIC_TOPICS, min(n_synthetic, len(SYNTHETIC_TOPICS)))

    for i, topic in enumerate(topics):
        scenario = generate_synthetic_scenario(topic)
        if scenario:
            dataset.append({
                "prompt": f"Event: {scenario['event_description']}\nResolution: {scenario['resolution_criteria']}",
                "naive_response": {
                    "probability": scenario.get("naive_probability", 0.5),
                    "quality": "REJECTED",
                    "adversarial_check": "NOT PERFORMED",
                },
                "car_response": {
                    "probability": scenario.get("car_adjusted_probability", 0.4),
                    "key_insight": scenario.get("key_insight", ""),
                    "actors": scenario.get("actors", []),
                    "quality": "PREFERRED",
                },
                "preference": "car_response",
                "source": "synthetic",
                "topic": topic,
            })
            print(f"  ✓ {i+1}/{n_synthetic}: {topic}")
            time.sleep(0.5)  # Rate limiting

    print(f"\nDataset complete: {len(dataset)} training examples")
    return dataset


if __name__ == "__main__":
    import os, json

    print("=" * 60)
    print("CAR Training Data Generator")
    print("Guruprasad Venkatakrishnan (2026)")
    print("=" * 60)

    dataset = build_training_dataset(n_synthetic=5)

    os.makedirs("training_data", exist_ok=True)
    with open("training_data/car_training_v1.jsonl", "w") as f:
        for example in dataset:
            f.write(json.dumps(example) + "\n")

    print(f"\nSaved: training_data/car_training_v1.jsonl")
    print(f"Total examples: {len(dataset)}")

    # Show the Hormuz example in detail
    print("\n" + "=" * 60)
    print("SAMPLE: Hormuz CAR Analysis")
    print("=" * 60)
    hormuz = build_hormuz_car()
    report = hormuz.report()
    print(f"\nDominant obstructor: {report['step3_dominant_obstructor']}")
    print(f"Structural silence warning: {report['step4_silence_warning']}")
    print(f"Adversarial haircut: {report['step6_adversarial_haircut']:.1%}")
    print(f"P_naive: 52% → P_CAR: {hormuz.apply_to_probability(0.52):.1%}")
    print(f"\nFractured actors: {[a['name'] for a in report['step5_fractured_actors']]}")
    for fa in report['step7_regime_trees']:
        print(f"\n  {fa['actor']} regime divergence: {fa['regime_divergence']:.2f}")
        for sa in fa['sub_actor_payoffs']:
            print(f"    {sa['name']}: U(YES)={sa['utility_yes']}, U(NO)={sa['utility_no']}")
