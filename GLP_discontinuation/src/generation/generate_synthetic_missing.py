"""
Multi-model synthetic clinical note generation for MISSING categories using vLLM.

Supports Qwen3.5-27B, Qwen3.6-27B, and GLM-4.7-Flash with automatic
chat template handling via the tokenizer. Generates 100 notes per
discontinuation category with batched inference.

This script covers the categories NOT included in generate_synthetic_multi.py:
- OTHER_UNKNOWN
- EFFICACY_EXPECTATION_MISMATCH
- CLINICAL_DECISION_MEDICAL_APPROPRIATENESS

# Generate all 3 missing categories
python generate_synthetic_missing.py --model Qwen/Qwen3.5-27B

# Generate specific categories only
python generate_synthetic_missing.py --model Qwen/Qwen3.5-27B --categories OTHER_UNKNOWN EFFICACY_EXPECTATION_MISMATCH

# With custom settings
python generate_synthetic_missing.py \
    --model Qwen/Qwen3.6-27B \
    --num-examples 50 \
    --batch-size 10 \
    --tensor-parallel-size 2

"""

import json
import argparse
import time
import re
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


CATEGORIES = {
    "OTHER_UNKNOWN": "other_unknown.txt",
    "EFFICACY_EXPECTATION_MISMATCH": "efficacy_expectation_mismatch.txt",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS": "clinical_decision_medical_appropriateness.txt",
}

MODEL_CONFIGS = {
    "google/gemma-4-31B-it": {
        "short_name": "gemma-4-31B-it",
        "system_message": "You are a helpful assistant.",
    },
    "Qwen/Qwen3.5-27B": {
        "short_name": "qwen3.5-27B",
        "system_message": "You are a helpful assistant. /no_think",
    },
    "Qwen/Qwen3.6-27B": {
        "short_name": "qwen3.6-27B",
        "system_message": "You are a helpful assistant. /no_think",
    },
    "zai-org/GLM-4.7-Flash": {
        "short_name": "GLM-4.7-Flash",
        "system_message": "You are a helpful assistant.",
    },
    "/home/user/GLP_project/models/qwen3_5_27b_instruct_fp16": {
        "short_name": "qwen3.5-27B",
        "system_message": "You are a helpful assistant. /no_think",
    },
}

NUM_EXAMPLES_PER_CATEGORY = 100
BATCH_SIZE = 25


def load_prompt_template(prompt_path: str) -> str:
    return Path(prompt_path).read_text()


def load_few_shot_examples(seed_dir: str, filename: str) -> str:
    return Path(seed_dir, filename).read_text()


def build_prompt(template: str, category: str, few_shot_text: str) -> str:
    prompt = template.replace("{PRIMARY_REASON}", category)
    prompt = prompt.replace("{CATEGORY_FEW_SHOT_EXAMPLES}", few_shot_text)
    return prompt


def format_chat_prompt(
    tokenizer: AutoTokenizer,
    user_content: str,
    system_message: str,
) -> str:
    """Use the tokenizer's built-in chat template for correct formatting."""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    brace_start = text.find("{")
    if brace_start == -1:
        return None

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def generate_for_category(
    llm: LLM,
    sampling_params: SamplingParams,
    tokenizer: AutoTokenizer,
    system_message: str,
    template: str,
    category: str,
    few_shot_text: str,
    num_examples: int,
    batch_size: int,
) -> list[dict]:
    """Generate synthetic notes for a single category using batched inference."""
    raw_prompt = build_prompt(template, category, few_shot_text)
    chat_prompt = format_chat_prompt(tokenizer, raw_prompt, system_message)

    results = []
    total_generated = 0
    attempt = 0
    max_attempts = (num_examples * 2) // batch_size + 1

    while total_generated < num_examples and attempt < max_attempts:
        remaining = num_examples - total_generated
        current_batch = min(batch_size, remaining)
        prompts = [chat_prompt] * current_batch

        print(
            f"  [{category}] Batch {attempt + 1}: generating {current_batch} "
            f"(have {total_generated}/{num_examples})"
        )

        outputs = llm.generate(prompts, sampling_params)

        for output in outputs:
            generated_text = output.outputs[0].text
            parsed = extract_json(generated_text)
            if parsed and "clinical_note" in parsed:
                parsed["primary_reason"] = category
                results.append(parsed)
                total_generated += 1
            else:
                print(f"  [{category}] Warning: failed to parse one output, skipping")

        attempt += 1

    if total_generated < num_examples:
        print(
            f"  [{category}] Warning: only generated {total_generated}/{num_examples} "
            f"valid examples after {attempt} attempts"
        )

    return results[:num_examples]


def load_existing_data(output_path: str) -> list[dict]:
    """Load existing JSONL data if file exists."""
    path = Path(output_path)
    if not path.exists():
        return []
    existing = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    existing.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return existing


def count_existing_data(output_dir: Path, categories: dict, categories_to_generate: list) -> dict:
    """Count existing records per category and combined file."""
    counts = {}
    for category in categories_to_generate:
        seed_file = categories[category]
        cat_file = output_dir / f"{seed_file.replace('.txt', '')}.jsonl"
        counts[category] = len(load_existing_data(cat_file))
    
    combined_file = output_dir / "missing_categories_synthetic_notes.jsonl"
    counts["_combined"] = len(load_existing_data(combined_file))
    return counts


def save_results(results: list[dict], output_path: str, merge: bool = True) -> int:
    """Save results, optionally merging with existing data. Returns total count."""
    if merge:
        existing = load_existing_data(output_path)
        results = existing + results
    with open(output_path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    return len(results)


def get_model_short_name(model_id: str) -> str:
    if model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_id]["short_name"]
    return model_id.split("/")[-1]


def get_system_message(model_id: str) -> str:
    if model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_id]["system_message"]
    return "You are a helpful assistant."


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinical notes for MISSING categories with vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="prompts/base_prompt.md",
        help="Path to prompt template",
    )
    parser.add_argument(
        "--seed-dir",
        type=str,
        default="data/seed_examples",
        help="Directory containing seed example files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/user/GLP_project/GLP_project_ad/data/synthetic_data/new_data_New_cat",
        help="Base output directory (model subfolder created automatically)",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=NUM_EXAMPLES_PER_CATEGORY,
        help="Number of examples per category",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for vLLM generation",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.92,
        help="Fraction of GPU memory for vLLM",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        choices=list(CATEGORIES.keys()),
        default=None,
        help="Specific categories to generate (default: all missing categories)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    prompt_path = project_root / args.prompt_file
    seed_dir = project_root / args.seed_dir

    short_name = get_model_short_name(args.model)
    output_dir = project_root / args.output_dir / short_name
    output_dir.mkdir(parents=True, exist_ok=True)

    system_message = get_system_message(args.model)

    categories_to_generate = args.categories if args.categories else list(CATEGORIES.keys())

    print(f"Model:          {args.model}")
    print(f"Short name:     {short_name}")
    print(f"Output dir:     {output_dir}")
    print(f"Examples/cat:   {args.num_examples}")
    print(f"Batch size:     {args.batch_size}")
    print(f"TP size:        {args.tensor_parallel_size}")
    print(f"System msg:     {system_message[:60]}...")
    print(f"Categories:     {', '.join(categories_to_generate)}")
    print()

    existing_counts = count_existing_data(output_dir, CATEGORIES, categories_to_generate)
    print("=== Existing Data in Output Folder ===")
    total_existing = 0
    for category in categories_to_generate:
        count = existing_counts.get(category, 0)
        total_existing += count
        print(f"  {category}: {count} records")
    print(f"  Combined file: {existing_counts.get('_combined', 0)} records")
    print(f"  Total existing (per-category): {total_existing} records")
    if total_existing > 0:
        print("  [New data will be MERGED with existing data]")
    print()

    template = load_prompt_template(prompt_path)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("Loading model...")
    llm = LLM(
        model=args.model,
        max_model_len=8192,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    print("Model loaded.\n")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=4192,
        seed=args.seed,
    )

    all_results = []
    metadata = {
        "model": args.model,
        "short_name": short_name,
        "system_message": system_message,
        "num_examples_per_category": args.num_examples,
        "batch_size": args.batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": 4192,
        "max_model_len": 8192,
        "seed": args.seed,
        "categories": {},
    }

    total_start = time.time()

    for category in categories_to_generate:
        seed_file = CATEGORIES[category]
        print(f"=== Generating: {category} ===")
        cat_start = time.time()

        few_shot_text = load_few_shot_examples(seed_dir, seed_file)
        results = generate_for_category(
            llm=llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            system_message=system_message,
            template=template,
            category=category,
            few_shot_text=few_shot_text,
            num_examples=args.num_examples,
            batch_size=args.batch_size,
        )

        cat_elapsed = time.time() - cat_start

        cat_output = output_dir / f"{seed_file.replace('.txt', '')}.jsonl"
        existing_cat_count = existing_counts.get(category, 0)
        total_cat_count = save_results(results, cat_output, merge=True)
        print(
            f"  Saved {len(results)} new + {existing_cat_count} existing = {total_cat_count} total "
            f"to {cat_output} ({cat_elapsed:.1f}s)\n"
        )

        all_results.extend(results)
        metadata["categories"][category] = {
            "new_count": len(results),
            "existing_count": existing_cat_count,
            "total_count": total_cat_count,
            "output_file": str(cat_output.name),
            "elapsed_seconds": round(cat_elapsed, 1),
        }

    combined_output = output_dir / "missing_categories_synthetic_notes.jsonl"
    existing_combined_count = existing_counts.get("_combined", 0)
    total_combined_count = save_results(all_results, combined_output, merge=True)

    total_elapsed = time.time() - total_start
    metadata["new_examples"] = len(all_results)
    metadata["existing_examples"] = existing_combined_count
    metadata["total_examples"] = total_combined_count
    metadata["total_elapsed_seconds"] = round(total_elapsed, 1)

    meta_path = output_dir / "missing_categories_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 60)
    print(f"Done. New: {len(all_results)}, Existing: {existing_combined_count}, Total: {total_combined_count} in {total_elapsed:.1f}s")
    print(f"Combined output: {combined_output}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
