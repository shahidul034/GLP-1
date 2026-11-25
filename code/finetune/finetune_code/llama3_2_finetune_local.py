from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
import json

model_name = "unsloth/Llama-3.2-3B-Instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 16384,
    dtype = None,
    load_in_4bit = False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
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

tokenizer = get_chat_template(tokenizer, chat_template = "llama-3.1")

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

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 16384,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    packing = False,
    args = SFTConfig(
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
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

trainer.train()

output_dir = "/home/user/GLP_project/GLP_project_ad/code/finetune/models/llama3-2-3b-finetune-fp16"
model.save_pretrained_merged(output_dir, tokenizer, save_method = "merged_16bit")