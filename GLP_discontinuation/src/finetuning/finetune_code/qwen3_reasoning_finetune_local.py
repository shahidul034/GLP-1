from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import Dataset
from unsloth.chat_templates import standardize_sharegpt, train_on_responses_only
from trl import SFTTrainer, SFTConfig

model_name = "unsloth/Qwen3-14B"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 16384,
    load_in_4bit = False,
    load_in_8bit = False,
    full_finetuning = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

import json

def load_sharegpt_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

train_file = "/home/user/GLP_project/GLP_project_ad/code/finetune/training_data/train_test/train_sharegpt.jsonl"
train_data = load_sharegpt_data(train_file)

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

dataset = Dataset.from_list(train_data)
dataset = dataset.map(extract_conversations, batched=True)
dataset = standardize_sharegpt(dataset)

conversations = tokenizer.apply_chat_template(
    list(dataset["conversations"]),
    tokenize = False,
)

combined_dataset = Dataset.from_pandas(pd.DataFrame({"text": conversations}))
combined_dataset = combined_dataset.shuffle(seed = 3407)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
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
        padding_free  = False,
    ),
)

trainer.train()

output_dir = "/home/user/GLP_project/GLP_project_ad/code/finetune/models/qwen3-14b-finetune-fp16"
model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit")