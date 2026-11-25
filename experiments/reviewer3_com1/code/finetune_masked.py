import os, json, sys
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq

MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
TRAIN_FILE = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
os.makedirs(OUTPUT_DIR, exist_ok=True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=4096, dtype=None, load_in_4bit=False,
)
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=3407,
)

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

data = [json.loads(line) for line in open(TRAIN_FILE)]
dataset = Dataset.from_list(data).map(extract_conversations, batched=True)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="text", max_seq_length=4096,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer), packing=False,
    args=SFTConfig(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=5, max_steps=100, learning_rate=2e-4,
        logging_steps=1, optim="adamw_8bit", weight_decay=0.001,
        lr_scheduler_type="linear", seed=3407, output_dir=OUTPUT_DIR,
        report_to="none",
    ),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)
trainer.train()
model.save_pretrained_merged(OUTPUT_DIR, tokenizer, save_method="merged_16bit")
print(f"Saved to {OUTPUT_DIR}")
