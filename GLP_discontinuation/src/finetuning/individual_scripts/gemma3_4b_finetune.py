#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import json

model_name = "unsloth/gemma-3-4b-it"
output_dir = "/home/user/GLP_project/GLP_project_ad/code/finetune/models/gemma3-4b-finetune"

print(f"Loading model: {model_name}")

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 16384,
    load_in_4bit = False,
    load_in_8bit = False,
    full_finetuning = False,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers = False,
    finetune_language_layers = True,
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 8,
    lora_alpha = 8,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

tokenizer = get_chat_template(tokenizer, chat_template = "gemma-3")

print("Loading local training data...")
dataset = load_dataset("json", data_files="/home/user/GLP_project/GLP_project_ad/code/finetune/train_sharegpt.jsonl", split = "train")
dataset = standardize_data_formats(dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

print(f"Dataset size: {len(dataset)}")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 100,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

print("Starting training...")
trainer.train()

print(f"Saving model to {output_dir}...")
model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit")

print("Done!")