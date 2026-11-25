import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

# 1. Configuration
max_seq_length = 8192
model_name = "unsloth/Qwen3.6-27B"

# 2. Load Model and Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = False, # Set to False if you have enough VRAM for BF16/FP16 loading
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",    
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# 4. Data Preparation (Instruction Following)
# Using Alpaca-Cleaned for high-quality general instructions
dataset = load_dataset("yahma/alpaca-cleaned", split = "train")

# Qwen 3.5 uses the ChatML template
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        # Combine instruction and input if input exists
        user_content = f"{instruction}\n{input_text}" if input_text else instruction
        
        parts = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        texts.append(tokenizer.apply_chat_template(parts, tokenize = False, add_generation_prompt = False))
    return { "text" : texts, }

dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. Training Setup
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100, # Adjust based on dataset size and needs
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

# Mask instructions to train only on assistant responses
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

# 6. Execute Training
trainer.train()

# 7. Save in FP16 Format
# This merges the LoRA weights back into the base model and saves as a single FP16 model
model.save_pretrained_merged("/home/user/GLP_project/models/qwen3_6_27b_instruct_fp16", tokenizer, save_method = "merged_16bit")
print("Model saved in FP16 format successfully.")