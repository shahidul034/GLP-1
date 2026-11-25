#!/usr/bin/env python3
"""
finetune_gender_swap.py

LoRA fine-tuning of Llama3.2-3B-Instruct on ShareGPT-format data.

Usage:
    CUDA_VISIBLE_DEVICES=7 python finetune_gender_swap.py \
        --train_file data/train_gender_swapped_sharegpt.jsonl \
        --output_dir models/llama3.2-3b_gender_swapped
"""

import os, sys, json, argparse
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"


def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    return model, tokenizer


def add_lora(model, r=16):
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model


def extract_conversations(examples):
    conversations = []
    for item in examples["conversations"]:
        conv = []
        for msg in item:
            if msg.get("from") == "human":
                conv.append({"role": "user", "content": msg["value"]})
            elif msg.get("from") == "gpt":
                conv.append({"role": "assistant", "content": msg["value"]})
        conversations.append(conv)
    return {"conversations": conversations}


def prepare_dataset(file_path):
    data = [json.loads(line) for line in open(file_path) if line.strip()]
    dataset = Dataset.from_list(data)
    dataset = dataset.map(extract_conversations, batched=True)
    return dataset


def train(model, tokenizer, dataset, output_dir, max_steps=100):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=max_steps,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    print(f"Starting training ({len(dataset)} samples, {max_steps} steps)...", flush=True)
    trainer.train()
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    print(f"Saved merged model to {output_dir}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {MODEL_NAME}...", flush=True)
    model, tokenizer = load_model()
    print("Adding LoRA...", flush=True)
    model = add_lora(model, r=16)
    print(f"Loading dataset from {args.train_file}...", flush=True)
    dataset = prepare_dataset(args.train_file)
    print(f"Training ({len(dataset)} samples)...", flush=True)
    train(model, tokenizer, dataset, args.output_dir, max_steps=args.max_steps)
    print(f"Done. Model saved to {args.output_dir}", flush=True)
