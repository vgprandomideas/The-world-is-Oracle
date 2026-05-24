# ⬡ World-as-Oracle

**Adversarial-Resistant AI Probability Estimation for Event-Driven Financial Instruments**

> *"The world is the oracle. Events speak for themselves — if you know how to listen across the right time horizon with the right source verification. And if you know which actors benefit from you listening to the wrong things."*

**Paper 1:** Venkatakrishnan (2026a). *The World as Oracle: Adversarial-Resistant AI Probability Estimation for Event-Driven Financial Instruments*

**Paper 2:** Venkatakrishnan (2026b). *Counterfactual Adversarial Reasoning (CAR): A Training Framework for Incentive-Aware Language Models*

**Affiliation:** Verslan | [predictmarkets.finance](https://predictmarkets.finance) | [verslan.xyz](https://verslan.xyz)

**Live App:** [the-world-is-oracle.streamlit.app](https://theworldisoracle.streamlit.app)

---

## What This Is

Two things in one repository:

**1. A 7-layer probability oracle** that resists information manipulation — the temporal persistence filter prevents manufactured narrative bursts from dominating probability estimates.

**2. CAR — Counterfactual Adversarial Reasoning** — the first formal framework for detecting actors who benefit from being invisible to the oracle. The oracle tells you the probability. CAR tells you who is gaming it.

---

## The CAR Framework — Eight Actor Classes

| Class | Example | Detection | Oracle Effect |
|---|---|---|---|
| Obstructor | Russia on Hormuz | Structural Silence Score | P discount (AH) |
| Dual-Class | Russia (also Extractor) | OI + Duration Payoff | AH + Duration Bias |
| Fence Sitter | China on Hormuz | FSS = BP × indifference × ambiguity | CI inflation |
| Achieved-Goal Extractor | Israel on Hormuz | Revealed objectives vs stated threat | Near-zero weight |
| Captured State | Qatar on Hormuz | Sovereignty Quotient < 0.20 | Mediator signals = noise |
| Credibility-Depleted Resolver | USA on Hormuz | Caused the crisis they propose to solve | Tier 1 × 0.40 |
| Nuclear Rentier | Pakistan on Hormuz | Multi-party incompatible sales | Duration bias + collapse risk |
| Vendetta Architect | Houthis on Hormuz | VS = grievance × infiltration × martyrdom | Ceasefire-proof discount |

---

## The Nine Adversarial Questions

Before any probability estimate, CAR fires nine questions:

1. What is this actor's economic reality?
2. What is their psychological driver — envy, revenge, survival, pride?
3. **Who benefits from NO, and are they in the signal chain?**
4. How many parties are they selling to simultaneously?
5. Is this actor internally fractured?
6. Do they physically control their own strategic choices?
7. Do they have a generational grievance overriding material incentives?
8. What collapses their strategy?
9. Who is their India — the comparator they block from succeeding?

Standard AI asks none of these. A CAR-trained model asks all nine before outputting any number.

---

## Validated Results

| Event | Naive P | Adversarial Haircut | P_CAR | Outcome |
|---|---|---|---|---|
| Hormuz normalisation 2027 | 52% | 45% (Russia SS=4.52) | 15% | Unresolved — Stable Disruption |
| Adani US conviction 2026 | 72% | 45% (GQG $1.87B long) | 40% | NO — charges dismissed May 2026 |
| Fed rate Jun 2023 | 52% | **0%** (no obstructors) | 52% | HOLD — zero haircut correct |

The Fed case is as important as the Hormuz case. CAR discriminates — it does not uniformly discount.

---

## Quick Start

```bash
git clone https://github.com/vgprandomideas/The-world-is-Oracle
cd The-world-is-Oracle
pip install -r requirements.txt
streamlit run app.py
```

**Run the Hormuz demo:**
```bash
python examples/maersk_hormuz.py
```

**Run the full CAR pipeline:**
```python
from oracle.car_pipeline import run_car_pipeline
import oracle.car_game_theory as gt
gt.construct_payoff_matrix = gt.construct_payoff_matrix_deterministic

result = run_car_pipeline(
    question="Will Hormuz reopen by December 2027?",
    known_actors=["Russia", "China", "Pakistan", "Houthis"],
    p_naive=0.52,
    use_prebuilt_profiles=True,
    verbose=True,
)
```

---

## Training Dataset — 10,003 Examples

The CAR training dataset is in `training_data/`:

| File | Examples | Purpose |
|---|---|---|
| `car_dataset_train.jsonl` | 8,002 | DPO fine-tuning |
| `car_dataset_val.jsonl` | 1,000 | Validation / early stopping |
| `car_dataset_test.jsonl` | 1,001 | Final evaluation |

Each example: `(prompt, naive_response[REJECTED], car_response[PREFERRED])`

**Fine-tune your own CAR model:**
```bash
pip install transformers trl peft bitsandbytes accelerate datasets
python scripts/finetune_dpo.py
python scripts/evaluate_car.py car-llama-3.1-8b
```

Target: **Adversarial Spontaneity Rate > 80%** — model asks Q3 without being prompted.

---

## Repository Structure

```
oracle/
  models.py            Article, OracleOutput, SwapContract
  config.py            Event category configs (λ, W, F_max)
  independence.py      ι(a_i) = 1 - max sim — citation amplification detection
  temporal_filter.py   F(t) fast shock + P(t) persistent signal
  probability.py       Platt scaling, Brier score, confidence interval
  state_machine.py     BASELINE/SHOCK/BUILDING/SUSTAINED/CONTESTED
  mtm.py               Daily VM, DMM initial margin, swap settlement
  oracle.py            WorldOracle — full 7-layer pipeline
  car.py               CAR framework — 8 classes, formal definitions
  car_actor_profiler.py  9 CAR questions, behavioural fingerprints
  car_game_theory.py   Payoff matrix, Nash equilibrium detection
  car_pipeline.py      Full 5-stage automated pipeline
  car_dataset_generator.py  10,003 example generator

actors/
  russia.json          Verified action history, SS=4.52
  china.json           FSS=0.72, fence sitter profile
  pakistan.json        Nuclear rentier, structural contradiction
  houthis.json         VS=5.99, vendetta architect
  india.json           Fractured — 3 sub-actors

training_data/
  car_dataset_train.jsonl   8,002 training examples
  car_dataset_val.jsonl     1,000 validation examples
  car_dataset_test.jsonl    1,001 test examples
  car_dataset_stats.json    Full statistics

scripts/
  finetune_dpo.py      DPO fine-tuning on Llama 3.1 8B
  evaluate_car.py      Adversarial Spontaneity Rate evaluation

examples/
  adani_retrospective.py   Paper Section 5.3
  fed_rate_decision.py     Paper Section 5.4
  maersk_hormuz.py         Paper Section 5.6
```

---

## Citation

```bibtex
@article{guruprasad2026oracle,
  title={The World as Oracle: Adversarial-Resistant AI Probability Estimation for Event-Driven Financial Instruments},
  author={Guruprasad Venkatakrishnan},
  institution={Verslan},
  year={2026},
  note={Working Paper. github.com/vgprandomideas/The-world-is-Oracle}
}

@article{guruprasad2026car,
  title={Counterfactual Adversarial Reasoning (CAR): A Training Framework for Incentive-Aware Language Models},
  author={Guruprasad Venkatakrishnan},
  institution={Verslan},
  year={2026},
  note={Working Paper. github.com/vgprandomideas/The-world-is-Oracle}
}
```

---

## License

MIT License — Verslan Private Limited, 2026
