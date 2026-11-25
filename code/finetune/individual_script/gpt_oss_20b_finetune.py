#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import json

model_name = "unsloth/gpt-oss-20b"
output_dir = "/home/user/GLP_project/GLP_project_ad/code/finetune/models/gpt-oss-20b-finetune"

print(f"Loading model: {model_name}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    dtype = None,
    max_seq_length = 16384,
    load_in_4bit = False,
    full_finetuning = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

print("Loading local training data...")
dataset = load_dataset("json", data_files="/home/user/GLP_project/GLP_project_ad/code/finetune/train_sharegpt.jsonl", split = "train")
dataset = standardize_sharegpt(dataset)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

print(f"Dataset size: {len(dataset)}")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start|>user<|message|>",
    response_part = "<|start|>assistant<|channel|>final<|message|>",
)

print("Starting training...")
trainer.train()

print(f"Saving model to {output_dir}...")
model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit")

print("Done!")