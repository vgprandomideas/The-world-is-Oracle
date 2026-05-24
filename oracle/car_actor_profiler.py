"""
CAR Actor Profiler — Stage 2

Builds a complete behavioural fingerprint for any geopolitical actor
from their ACTIONS (not statements) across history.

Core principle: Actions reveal utility functions.
Statements reveal desired perception.
CAR uses actions. Standard AI uses statements.

Guruprasad Venkatakrishnan (2026)
Verslan / predictmarkets.finance / verslan.xyz
"""

import json
import os
import anthropic

client = anthropic.Anthropic()

# ── The 9 CAR Questions ──────────────────────────────────────────────────────

CAR_QUESTIONS = [
    {
        "id": "Q1_economic_reality",
        "question": "What is this actor's economic reality? Are they strong or broken? Who do they depend on financially?",
        "looks_for": ["GDP", "debt", "IMF", "sanctions", "trade deficit", "bailout", "reserves"],
        "reveals": "survival_desperation"
    },
    {
        "id": "Q2_psychological_driver",
        "question": "What is this actor's core psychological driver? Envy, revenge, pride, survival, glory, fear?",
        "looks_for": ["historical humiliation", "rivalry", "comparison", "identity", "religion", "nationalism"],
        "reveals": "inferiority_complex or superiority_drive"
    },
    {
        "id": "Q3_who_benefits_from_no",
        "question": "Who benefits if this event does NOT resolve? Are they present in the signal chain?",
        "looks_for": ["revenue from crisis", "leverage from conflict", "absent from discourse", "silent beneficiary"],
        "reveals": "structural_silence_score"
    },
    {
        "id": "Q4_multi_party_selling",
        "question": "How many parties is this actor selling to simultaneously? What structural contradiction exists?",
        "looks_for": ["deals with opposing parties", "defence pact", "backchannel", "multiple alliances"],
        "reveals": "structural_contradiction"
    },
    {
        "id": "Q5_actor_fracture",
        "question": "Is this actor internally unified? What are the sub-actors and their divergent payoffs?",
        "looks_for": ["private sector vs government", "military vs civilian", "factions", "internal opposition"],
        "reveals": "actor_fracture"
    },
    {
        "id": "Q6_sovereignty_check",
        "question": "Does this actor physically control their own strategic choices? Or are they a captured state?",
        "looks_for": ["foreign military base", "debt trap", "imposed conditions", "occupation", "dependency"],
        "reveals": "sovereignty_quotient"
    },
    {
        "id": "Q7_shakuni_check",
        "question": "Does this actor have a historical grievance so deep it overrides current material incentives? Are they engineering destruction from inside?",
        "looks_for": ["historical atrocity", "generational grievance", "revenge oath", "martyrdom acceptance", "long game"],
        "reveals": "vendetta_score"
    },
    {
        "id": "Q8_collapse_trigger",
        "question": "What single event would collapse this actor's strategy? How close is that trigger?",
        "looks_for": ["structural contradiction exposure", "patron withdrawal", "economic collapse", "military defeat"],
        "reveals": "fragility_score"
    },
    {
        "id": "Q9_india_equivalent",
        "question": "Who is this actor's India — the comparator they cannot bear to see succeed? How does this drive their blocking behaviour?",
        "looks_for": ["rivalry", "zero-sum framing", "blocking successful competitor", "envy-driven decisions"],
        "reveals": "envy_blocking_factor"
    }
]


ACTOR_PROFILER_SYSTEM = """You are a CAR (Counterfactual Adversarial Reasoning) Actor Profiler.

Your job: Build a complete behavioural fingerprint for a geopolitical actor
from their HISTORICAL ACTIONS — not their statements.

CRITICAL RULE: Statements are noise. Actions are data.
- "We want peace" is noise.
- Signing a defence pact while mediating the enemy is data.
- "We are neutral" is noise.
- Hosting 10,000 enemy troops is data.

For each actor, you must answer the 9 CAR Questions using ONLY
evidence from their past actions, financial flows, voting records,
military deployments, and treaty signings.

Output format: Valid JSON only. No preamble."""


def profile_actor(
    actor_name: str,
    event_context: str,
    historical_actions: list,
) -> dict:
    """
    Build a complete CAR actor profile from historical actions.

    historical_actions: list of dicts with:
      {"date": "2025-09-17",
       "action": "Signed Strategic Mutual Defence Agreement with Saudi Arabia",
       "source": "Saudi Press Agency",
       "type": "treaty|financial|military|vote|statement"}
    """

    actions_text = "\n".join([
        f"[{a['date']}] [{a['type'].upper()}] {a['action']} (Source: {a.get('source', 'verified')})"
        for a in sorted(historical_actions, key=lambda x: x['date'])
    ])

    prompt = f"""Actor: {actor_name}
Event Context: {event_context}

VERIFIED HISTORICAL ACTIONS (actions only — not statements):
{actions_text}

Answer all 9 CAR Questions based ONLY on the actions above.
Do not use general knowledge — only infer from the actions provided.

Return this exact JSON structure:
{{
  "actor": "{actor_name}",
  "car_questions": {{
    "Q1_economic_reality": {{
      "finding": "...",
      "evidence_actions": ["date of action that reveals this"],
      "score": 0.0
    }},
    "Q2_psychological_driver": {{
      "finding": "...",
      "evidence_actions": [],
      "driver_type": "envy|revenge|survival|pride|fear|glory"
    }},
    "Q3_who_benefits_from_no": {{
      "finding": "...",
      "obstruction_incentive": 0.0,
      "presence_in_signal_chain": 0.0,
      "structural_silence_score": 0.0
    }},
    "Q4_multi_party_selling": {{
      "parties": [],
      "products_sold": [],
      "structural_contradiction": "...",
      "contradiction_score": 0.0
    }},
    "Q5_actor_fracture": {{
      "is_fractured": true,
      "sub_actors": [
        {{"name": "...", "utility_yes": 0.0, "utility_no": 0.0, "dominant": true}}
      ]
    }},
    "Q6_sovereignty_check": {{
      "sovereignty_quotient": 0.0,
      "capture_mechanisms": [],
      "is_captured_state": false
    }},
    "Q7_shakuni_check": {{
      "vendetta_score": 0.0,
      "historical_grievance": "...",
      "martyrdom_acceptance": 0.0,
      "is_vendetta_architect": false
    }},
    "Q8_collapse_trigger": {{
      "trigger": "...",
      "proximity": "imminent|near|distant",
      "fragility_score": 0.0
    }},
    "Q9_india_equivalent": {{
      "comparator": "...",
      "envy_blocking_factor": 0.0,
      "evidence": "..."
    }}
  }},
  "actor_class": "obstructor|fence_sitter|nuclear_rentier|achieved_goal_extractor|captured_state|credibility_depleted_resolver|wounded_opportunist|vendetta_architect|genuine_resolver",
  "utility_yes": 0.0,
  "utility_no": 0.0,
  "obstruction_incentive": 0.0,
  "blocking_power": 0.0,
  "structural_silence_score": 0.0,
  "adversarial_haircut_contribution": 0.0,
  "one_line_summary": "..."
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=ACTOR_PROFILER_SYSTEM,
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
        return {"actor": actor_name, "error": str(e), "raw": raw}


# ── Pre-built Pakistan profile from verified actions ─────────────────────────

PAKISTAN_VERIFIED_ACTIONS = [
    {"date": "1947-08-14", "type": "political", "action": "Partition — same economic starting point as India", "source": "Historical record"},
    {"date": "1958-01-01", "type": "financial", "action": "First IMF bailout — begins pattern of 24 bailouts over 68 years", "source": "IMF records"},
    {"date": "1998-05-28", "type": "military", "action": "Conducts nuclear tests — becomes only Islamic nuclear power", "source": "IAEA"},
    {"date": "2001-09-01", "type": "military", "action": "Provides US military bases for Afghanistan war in exchange for aid", "source": "DoD"},
    {"date": "2018-01-01", "type": "diplomatic", "action": "Trump calls Pakistan out for 'lies and deceit' — labelled pariah", "source": "White House"},
    {"date": "2023-06-01", "type": "financial", "action": "IMF bailout #24 — $3B Stand-By Arrangement", "source": "IMF"},
    {"date": "2024-09-01", "type": "financial", "action": "IMF bailout extended — $7B arrangement running to 2027", "source": "IMF"},
    {"date": "2025-09-17", "type": "treaty", "action": "Signs Strategic Mutual Defence Agreement with Saudi Arabia — nuclear umbrella implied", "source": "Saudi Press Agency"},
    {"date": "2026-01-01", "type": "military", "action": "Pakistan Air Force chief visits Saudi Arabia — operational planning", "source": "Saudi MoD"},
    {"date": "2026-03-03", "type": "diplomatic", "action": "Tells Iranian FM about Saudi defence pact while positioning as neutral mediator", "source": "Al Jazeera"},
    {"date": "2026-04-11", "type": "diplomatic", "action": "Hosts Islamabad Talks — 21 hours, zero deal, process continues", "source": "Wikipedia"},
    {"date": "2026-04-11", "type": "military", "action": "Deploys fighter jets to King Abdulaziz Air Base, Saudi Arabia", "source": "Fortune/Saudi MoD"},
    {"date": "2026-05-07", "type": "diplomatic", "action": "Military spokesperson: 'Chosen by God to protect Two Holy Mosques'", "source": "Arab News"},
    {"date": "2026-05-01", "type": "economic", "action": "GDP per capita $1,710 — 162nd globally vs India's 11x larger economy", "source": "IMF/World Bank"},
]
