from unsloth import FastLanguageModel
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only
import json

MODELS = {
    "27B": "unsloth/Qwen3.5-27B",
    "9B": "unsloth/Qwen3.5-9B",
    "2B": "unsloth/Qwen3.5-2B",
    "0.8B": "unsloth/Qwen3.5-0.8B",
}

def load_model(model_name: str, max_seq_length: int = 16384):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
    )
    return model, tokenizer

def add_lora(model, r: int = 8):
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", "out_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return model

def load_sharegpt_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

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

def prepare_dataset(file_path: str):
    train_data = load_sharegpt_data(file_path)
    dataset = Dataset.from_list(train_data)
    dataset = dataset.map(extract_conversations, batched=True)
    return dataset

def train(model, tokenizer, dataset, output_dir: str,
          per_device_train_batch_size: int = 4,
          gradient_accumulation_steps: int = 2,
          warmup_steps: int = 5,
          max_steps: int = 100,
          learning_rate: float = 2e-4,
          logging_steps: int = 1):
    
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False) for convo in convos]
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
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n<think>",
    )

    trainer.train()
    return trainer

def save_fp16(model, tokenizer, output_dir: str):
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")

train_file = "/home/user/GLP_project/GLP_project_ad/code/finetune/training_data/train_test/train_sharegpt.jsonl"

for model_size in ["27B", "9B", "2B", "0.8B"]:
    print(f"Training Qwen3.5-{model_size}...")
    
    model_name = MODELS[model_size]
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, 16384)
    
    print("Adding LoRA adapters...")
    model = add_lora(model, r=8)
    
    print("Preparing dataset...")
    dataset = prepare_dataset(train_file)
    
    output_dir = f"/home/user/GLP_project/GLP_project_ad/code/finetune/models/qwen3.5-{model_size.lower().replace('.', '-')}-finetune-fp16"
    
    print("Training...")
    if model_size == "27B":
        batch_size = 1
        grad_accum = 4
    elif model_size == "9B":
        batch_size = 2
        grad_accum = 4
    else:
        batch_size = 2
        grad_accum = 4
    
    train(model, tokenizer, dataset, output_dir,
          per_device_train_batch_size=batch_size,
          gradient_accumulation_steps=grad_accum,
          max_steps=100,
          learning_rate=2e-4)
    
    print(f"Saving fp16 to {output_dir}...")
    save_fp16(model, tokenizer, output_dir)
    
    print(f"Completed Qwen3.5-{model_size}!")
    print("-" * 50)