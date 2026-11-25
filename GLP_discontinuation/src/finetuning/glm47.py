# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

# 1. Configuration & Model Loading
max_seq_length = 2048 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/GLM-4.7-Flash",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # Use 4bit for efficient training; we will export to FP16 later
    trust_remote_code = True,
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Increased rank for better instruction following
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj", "out_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 3. Data Preparation (Using Alpaca-Cleaned for Instruction Following)
alpaca_prompt = """[gMASK]<sop><|user|>
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}<|assistant|>
<think></think>
{}"""

dataset = load_dataset("yahma/alpaca-cleaned", split = "train")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output)
        texts.append(text)
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 4. Training Setup
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = SFTConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100, # Adjust based on your time constraints
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


# Masking to train only on the assistant's response
trainer = train_on_responses_only(
    trainer,
    instruction_part = "[gMASK]<sop><|user|>",
    response_part = "</think>\n",
)

# import ipdb; ipdb.set_trace()

# 5. Execute Training
trainer.train()

# 6. Save in FP16 Format
# This merges the LoRA weights into the base model and exports to 16-bit precision
model.save_pretrained_merged(
    "/home/user/GLP_project/models/glm_4_7_finetuned_fp16", 
    tokenizer, 
    save_method = "merged_16bit"
)

print("Training complete. Model saved in 'glm_4_7_finetuned_fp16' folder.")