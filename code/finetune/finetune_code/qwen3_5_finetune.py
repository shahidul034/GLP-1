# -*- coding: utf-8 -*-
import argparse
import os

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only


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


def prepare_dataset(dataset_name: str = "unsloth/OpenMathReasoning-mini"):
    dataset = load_dataset(dataset_name, split="cot")

    def generate_conversation(examples):
        problems = examples["problem"]
        solutions = examples["generated_solution"]
        conversations = []
        for problem, solution in zip(problems, solutions):
            conversations.append([
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ])
        return {"conversations": conversations}

    dataset = dataset.map(generate_conversation, batched=True)

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset


def train(model, tokenizer, dataset, output_dir: str = "qwen_lora",
          per_device_train_batch_size: int = 4,
          gradient_accumulation_steps: int = 2,
          warmup_steps: int = 5,
          max_steps: int = 60,
          learning_rate: float = 2e-4,
          logging_steps: int = 1):
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


def save_lora(model, tokenizer, output_dir: str):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_fp16(model, tokenizer, output_dir: str = "qwen_finetune_16bit"):
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5 models")
    parser.add_argument("--model", type=str, default="27B",
                        choices=["27B", "9B", "2B", "0.8B"],
                        help="Model size to use")
    parser.add_argument("--output_lora", type=str, default="qwen_lora",
                        help="Output directory for LoRA adapters")
    parser.add_argument("--output_fp16", type=str, default="qwen_finetune_16bit",
                        help="Output directory for fp16 merged model")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--per_device_batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=60,
                        help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--dataset", type=str, default="unsloth/OpenMathReasoning-mini",
                        help="Dataset name")
    parser.add_argument("--save_fp16", action="store_true",
                        help="Save fp16 merged model")

    args = parser.parse_args()

    model_name = MODELS[args.model]
    print(f"Loading model: {model_name}")
    model, tokenizer = load_model(model_name, args.max_seq_length)

    print("Adding LoRA adapters...")
    model = add_lora(model, r=args.lora_r)

    print("Preparing dataset...")
    dataset = prepare_dataset(args.dataset)

    print("Training...")
    train(model, tokenizer, dataset,
          per_device_train_batch_size=args.per_device_batch_size,
          gradient_accumulation_steps=args.gradient_accumulation_steps,
          max_steps=args.max_steps,
          learning_rate=args.learning_rate)

    print(f"Saving LoRA to {args.output_lora}...")
    save_lora(model, tokenizer, args.output_lora)

    if args.save_fp16:
        print(f"Saving fp16 to {args.output_fp16}...")
        save_fp16(model, tokenizer, args.output_fp16)

    print("Done!")


if __name__ == "__main__":
    main()