from unsloth import FastModel
import torch
from unsloth.chat_templates import get_chat_template, standardize_data_formats, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import json

model_name = "unsloth/gemma-4-E2B-it"

model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    dtype = None,
    max_seq_length = 16384,
    load_in_4bit = False,
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

tokenizer = get_chat_template(tokenizer, chat_template = "gemma-4")

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
dataset = standardize_data_formats(dataset)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
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
        report_to = "none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|turn>user\n",
    response_part = "<|turn>model\n",
)

trainer.train()

output_dir = "/home/user/GLP_project/GLP_project_ad/code/finetune/models/gemma4-e2b-finetune-fp16"
model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit")