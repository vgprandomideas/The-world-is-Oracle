"""
CAR Evaluation Script

Tests whether the fine-tuned model spontaneously asks
adversarial questions WITHOUT being prompted.

Key metric: Adversarial Spontaneity Rate (ASR)
  = fraction of novel questions where model asks Q3
    (who benefits from NO?) without CAR instructions

Pass threshold: ASR > 0.80

Guruprasad Venkatakrishnan (2026)
Verslan / predictmarkets.finance / verslan.xyz
"""

import json
from pathlib import Path

NOVEL_QUESTIONS = [
    "Will the Taiwan Strait remain open to commercial shipping in 2027?",
    "Will the India-Pakistan ceasefire hold through 2026?",
    "Will OPEC+ agree to production cuts at the June 2027 meeting?",
    "Will Venezuela hold free elections in 2027?",
    "Will the North Korea nuclear talks resume before 2028?",
    "Will Armenia join the European Union by 2030?",
    "Will Sudan reach a peace agreement in 2027?",
    "Will the US-China trade war escalate further in 2026?",
    "Will Ethiopia and Eritrea maintain their peace agreement through 2027?",
    "Will the Libya unity government survive the 2026 elections?",
]

ADVERSARIAL_INDICATORS = [
    "who benefits from", "benefits from no", "obstruction incentive",
    "structural silence", "who profits", "who wins if this fails",
    "blocking coalition", "who has incentive to prevent",
    "who benefits from non-resolution", "silent actor",
]


def evaluate_model(model_path: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError:
        print("Install: pip install transformers")
        return

    print(f"Loading {model_path}...")
    pipe = pipeline("text-generation", model=model_path, max_new_tokens=500)

    spontaneous_count = 0
    results = []

    for q in NOVEL_QUESTIONS:
        response = pipe(q)[0]["generated_text"]
        response_lower = response.lower()

        asked_adversarial = any(
            indicator in response_lower
            for indicator in ADVERSARIAL_INDICATORS
        )

        if asked_adversarial:
            spontaneous_count += 1

        results.append({
            "question": q,
            "asked_adversarial": asked_adversarial,
            "response_preview": response[:200],
        })
        print(f"  {'✓' if asked_adversarial else '✗'} {q[:60]}...")

    asr = spontaneous_count / len(NOVEL_QUESTIONS)
    print(f"\nAdversarial Spontaneity Rate: {asr:.0%}")
    print(f"Pass threshold: 80%")
    print(f"Result: {'PASS' if asr >= 0.80 else 'FAIL — more training data needed'}")

    with open("evaluation_results.json", "w") as f:
        json.dump({"asr": asr, "results": results}, f, indent=2)

    return asr


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "car-llama-3.1-8b"
    evaluate_model(model_path)
