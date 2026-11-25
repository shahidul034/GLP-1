from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import json
import os
os.environ["HF_HUB_DISABLE_MXF4"] = "1"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    dtype = torch.bfloat16,
    max_seq_length = 16384,
    load_in_4bit = False,
    load_in_8bit = False,
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
    use_gradient_checkpointing = False,
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

tokenizer = get_chat_template(tokenizer, chat_template = "gpt-oss")

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

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

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

trainer.train()

output_dir = "/home/user/GLP_project/GLP_project_ad/code/finetune/models/gpt-oss-20b-finetune-fp16"
model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit")