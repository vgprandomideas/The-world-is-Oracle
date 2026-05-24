"""
CAR Fine-Tuning Script — DPO on Llama 3.1 8B

Usage:
  pip install transformers trl peft bitsandbytes accelerate datasets
  python scripts/finetune_dpo.py

Trains a CAR-aware language model that spontaneously asks
adversarial questions before estimating any probability.

Guruprasad Venkatakrishnan (2026)
Verslan / predictmarkets.finance / verslan.xyz
"""

import json
from pathlib import Path

def load_dataset(split: str):
    path = Path(f"training_data/car_dataset_{split}.jsonl")
    examples = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            examples.append({
                "prompt": ex["prompt"],
                "chosen": (
                    f"[CAR Protocol Engaged]\n\n"
                    f"Actors: {json.dumps([a['name']+':'+a['class'] for a in ex['actors']])}\n\n"
                    f"Key insight: {ex['car_response']['key_insight']}\n\n"
                    f"Equilibrium: {ex['car_response']['equilibrium']}\n"
                    f"Adversarial Haircut: {ex['car_response']['adversarial_haircut']:.1%}\n\n"
                    f"P_CAR = {ex['car_response']['probability']:.0%}"
                ),
                "rejected": (
                    f"Based on available signals, the probability is approximately "
                    f"{ex['naive_response']['probability']:.0%}. "
                    f"{ex['naive_response']['reasoning']}"
                ),
            })
    return examples


def train():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOTrainer, DPOConfig
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset
        import torch
    except ImportError:
        print("Install: pip install transformers trl peft bitsandbytes accelerate datasets")
        return

    MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    OUTPUT = "car-llama-3.1-8b"

    print(f"Loading {MODEL}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj","v_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_data = Dataset.from_list(load_dataset("train"))
    val_data   = Dataset.from_list(load_dataset("val"))

    training_args = DPOConfig(
        output_dir=OUTPUT,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        beta=0.1,
        max_length=2048,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=500,
        fp16=False,
        bf16=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    print("Training...")
    trainer.train()
    trainer.save_model(OUTPUT)
    print(f"Model saved to {OUTPUT}")
    print("Run evaluation: python scripts/evaluate_car.py")


if __name__ == "__main__":
    train()
