#!/usr/bin/env python3
"""
Phase 3b: Qwen3.5-9B Training for Ablation Study
Fine-tunes Qwen3.5-9B via Unsloth on each threshold dataset.
All hyperparameters held constant across runs.
Based on code/finetune/individual_script/qwen3_5_9b_finetune_local.py
"""

import os
import json
import time
import argparse
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

DATASETS_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies/datasets")
MODELS_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies/models/qwen3.5-9b")
RESULTS_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies/results")

BASE_MODEL = "Qwen/Qwen3.5-9B"
THRESHOLDS = [1.0, 0.95, 0.90, 0.85, 0.80]

INSTRUCTION = """You are a clinical NLP specialist. Analyze the following clinical note and classify the primary reason for GLP-1 receptor agonist discontinuation.

Categories:
- ADVERSE_EFFECTS_TOLERABILITY: GI side effects, systemic symptoms, serious adverse events, injection site reactions
- ACCESS_COST_SYSTEM_BARRIERS: Insurance issues, high costs, prior authorization, supply shortages
- EFFICACY_EXPECTATION_MISMATCH: Weight plateau, insufficient results, medication not working as expected
- PATIENT_PREFERENCE_BELIEFS: Safety fears, misinformation, stigma, preference for natural approaches
- PLANNED_GOAL_BASED_DISCONTINUATION: Reached goal weight, planned tapering, underlying condition improved
- CLINICAL_DECISION_MEDICAL_APPROPRIATENESS: Contraindications, drug interactions, switching therapy, pregnancy
- OTHER_UNKNOWN: Reason unclear, lost to follow-up, ambiguous documentation

Respond with a JSON object containing:
1. "primary_reason": The main discontinuation category
2. "confidence": Your confidence level (high/medium/low)
3. "supporting_evidence": Key phrases from the note supporting your classification
4. "secondary_factors": Any additional contributing factors (if applicable)"""

TRAIN_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 5,
    "max_steps": 100,
    "weight_decay": 0.001,
    "lora_r": 8,
    "lora_alpha": 16,
}


def prepare_sharegpt_dataset(train_data, output_path):
    records = []
    for item in train_data:
        note = item.get("clinical_note", "")
        label = item.get("primary_reason", "OTHER_UNKNOWN")
        messages = [
            {"role": "user", "content": INSTRUCTION + "\n\nClinical Note:\n" + note},
            {"role": "assistant", "content": json.dumps({"primary_reason": label, "confidence": "high", "supporting_evidence": "", "secondary_factors": ""})}
        ]
        records.append({"conversations": messages})
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return output_path


def train_at_threshold(threshold, train_data):
    label = str(threshold).replace(".", "_")
    output_path = MODELS_DIR / f"threshold_{label}"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Training Qwen3.5-9B @ threshold={threshold}")
    print(f"Train: {len(train_data)} samples | Output: {output_path}")
    print(f"{'='*70}")

    tmp_sharegpt = f"/tmp/qwen_train_{label}.jsonl"
    prepare_sharegpt_dataset(train_data, tmp_sharegpt)

    start_time = time.time()

    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=16384,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=TRAIN_CONFIG["lora_r"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj", "out_proj"],
        lora_alpha=TRAIN_CONFIG["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

    dataset = load_dataset("json", data_files=tmp_sharegpt, split="train")
    dataset = standardize_data_formats(dataset)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=SFTConfig(
            dataset_text_field="text",
            dataset_num_proc=1,
            per_device_train_batch_size=TRAIN_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=TRAIN_CONFIG["gradient_accumulation_steps"],
            warmup_steps=TRAIN_CONFIG["warmup_steps"],
            max_steps=TRAIN_CONFIG["max_steps"],
            learning_rate=TRAIN_CONFIG["learning_rate"],
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=TRAIN_CONFIG["weight_decay"],
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    gpu_props = torch.cuda.get_device_properties(0)
    start_mem = round(torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024, 3)
    max_mem = round(gpu_props.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_props.name} | Max memory: {max_mem} GB | Reserved: {start_mem} GB")

    print("Starting training...")
    trainer.train()

    used_mem = round(torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024, 3)
    print(f"Peak memory: {used_mem} GB")

    print(f"Saving model to {output_path}...")
    model.save_pretrained_merged(output_path, tokenizer, save_method="merged_16bit")

    elapsed = time.time() - start_time
    os.remove(tmp_sharegpt)

    result = {
        "threshold": threshold,
        "train_samples": len(train_data),
        "training_time_seconds": round(elapsed, 1),
        "output_path": str(output_path),
    }

    results_path = RESULTS_DIR / f"qwen3.5-9b_threshold_{label}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Time: {elapsed:.1f}s | Results: {results_path}")

    del model, trainer
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, choices=THRESHOLDS, default=None,
                        help="Run a specific threshold. If not set, runs all thresholds.")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 3b: Qwen3.5-9B Training Sweep (Unsloth)")
    print("=" * 70)
    print(f"Base model: {BASE_MODEL}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Config: {TRAIN_CONFIG}")

    thresholds_to_run = [args.threshold] if args.threshold else THRESHOLDS
    all_results = {}

    for threshold in thresholds_to_run:
        label = str(threshold).replace(".", "_")
        train_path = DATASETS_DIR / f"train_{label}.jsonl"
        if not train_path.exists():
            print(f"WARNING: {train_path} not found, skipping threshold {threshold}")
            continue

        train_data = []
        with open(train_path) as f:
            for line in f:
                train_data.append(json.loads(line.strip()))

        results = train_at_threshold(threshold, train_data)
        all_results[threshold] = results

    summary_path = RESULTS_DIR / "qwen_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {summary_path}")

    print("\n" + "=" * 70)
    print("Summary: Qwen3.5-9B Training per Threshold")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Train N':<10} {'Time (s)':<10}")
    for threshold, res in all_results.items():
        print(f"{threshold:<12} {res['train_samples']:<10} {res['training_time_seconds']:<10}")
    print("\nPhase 3b complete. NOTE: Macro-F1 will be computed during evaluation phase.")


if __name__ == "__main__":
    main()