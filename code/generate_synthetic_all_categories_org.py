"""
Unified multi-model synthetic clinical note generation for ALL GLP-1 discontinuation categories.

Supports Qwen3.5-27B, Qwen3.6-27B, GLM-4.7-Flash, and Gemma-4-31B with automatic
chat template handling via the tokenizer. Generates notes per discontinuation
category with batched inference.

Each category has subcategories that are cycled through to create diverse data.
The subcategory is injected into the prompt as a specific focus area.

Usage:
# Generate all 7 categories
CUDA_VISIBLE_DEVICES=0 python code/generate_synthetic_all_categories.py --model google/gemma-4-31B-it --num-examples 1000 --resume
CUDA_VISIBLE_DEVICES=0 python code/generate_synthetic_all_categories.py --model google/gemma-3-27b-it --num-examples 1000 --resume
CUDA_VISIBLE_DEVICES=1 python code/generate_synthetic_all_categories.py --model google/gemma-4-31B-it --num-examples 1000
CUDA_VISIBLE_DEVICES=1 python code/generate_synthetic_all_categories.py --model google/gemma-3-27B-it --num-examples 1000
python /home/user/GLP_project/rc.py --g 3 --p 0.6

CUDA_VISIBLE_DEVICES=0 python code/generate_synthetic_all_categories.py --model /home/user/GLP_project/models/glm_4_7_finetuned_fp16 --num-examples 1000

# Generate specific categories only
python generate_synthetic_all_categories.py --model Qwen/Qwen3.5-27B \
    --categories ADVERSE_EFFECTS_TOLERABILITY ACCESS_COST_SYSTEM_BARRIERS

# With custom settings
python generate_synthetic_all_categories.py \
    --model Qwen/Qwen3.6-27B \
    --num-examples 50 \
    --batch-size 10 \
    --tensor-parallel-size 2
"""

import json
import argparse
import time
import re
import random
from pathlib import Path
from typing import Iterator

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# Categories with their subcategories for diverse data generation
CATEGORY_DEFINITIONS = {
    "ADVERSE_EFFECTS_TOLERABILITY": {
        "seed_file": "adverse_effects.txt",
        "subcategories": [
            "GI side effects (nausea, vomiting, diarrhea)",
            "systemic symptoms (fatigue, dizziness, headache)",
            "serious adverse events (pancreatitis, gallbladder issues)",
            "injection site reactions",
            "gastrointestinal discomfort and bloating",
            "loss of appetite beyond therapeutic intent",
        ],
    },
    "ACCESS_COST_SYSTEM_BARRIERS": {
        "seed_file": "access_cost_system_barriers.txt",
        "subcategories": [
            "insurance loss or denial of coverage",
            "coverage differences between anti-obesity medication (AOM) and anti-diabetic medication (ADM) indications",
            "high out-of-pocket costs",
            "formulary restrictions or prior authorization requirements",
            "medication shortage or supply issues",
            "pharmacy access or availability problems",
            "Medication unavailability or sold out"
        ],
    },
    "EFFICACY_EXPECTATION_MISMATCH": {
        "seed_file": "efficacy_expectation_mismatch.txt",
        "subcategories": [
            "weight plateau after initial success",
            "insufficient weight loss compared to expectations",
            "inadequate glycemic control despite medication",
            "'not working anymore' perception",
            "slower than expected results",
            "weight regain while still on medication",
        ],
    },
    "PATIENT_PREFERENCE_BELIEFS": {
        "seed_file": "patient_preference_beliefs.txt",
        "subcategories": [
            "fear of long-term safety concerns (thyroid cancer, cardiovascular)",
            "misinformation from social media or online sources",
            "social stigma around weight loss medications",
            "desire to stop after reaching goal weight",
            "psychological discomfort with medication dependency",
            "preference for 'natural' or lifestyle-only approaches",
            "family or peer influence against medication use",
        ],
    },
    "PLANNED_GOAL_BASED_DISCONTINUATION": {
        "seed_file": "planned_goal_based_discontinuation.txt",
        "subcategories": [
            "used as a short-term tool to jumpstart weight loss",
            "reached target weight goal",
            "intentional tapering as planned with provider",
            "underlying condition improved (diabetes remission, A1c normalized)",
            "transition to maintenance phase without medication",
            "planned pregnancy or fertility considerations",
        ],
    },
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS": {
        "seed_file": "clinical_decision_medical_appropriateness.txt",
        "subcategories": [
            "contraindications discovered (CKD progression, pancreatitis history)",
            "drug-drug interactions requiring discontinuation",
            "new comorbidity making GLP-1 inappropriate",
            "switching to alternative therapy (bariatric surgery, different medication class)",
            "guideline-driven medication changes",
            "symptom or disease resolution making treatment unnecessary",
            "provider recommendation due to lack of benefit",
            "pregnancy",
        ],
    },
    "OTHER_UNKNOWN": {
        "seed_file": "other_unknown.txt",
        "subcategories": [
            "reason not documented or unclear from note",
            "patient lost to follow-up",
            "medication simply not continued without explanation",
            "ambiguous or vague documentation",
            "multiple possible reasons without clear primary cause",
            "patient non-adherence without stated reason",
            "discontinuation noted in medication list only",
        ],
    },
}

# Simple mapping for backward compatibility
CATEGORIES = {cat: info["seed_file"] for cat, info in CATEGORY_DEFINITIONS.items()}

MODEL_CONFIGS = {
    "google/gemma-4-31B-it": {
        "short_name": "gemma-4-31b-finetuned-fp16",
        "system_message": "You are a helpful assistant.",
    },
    "/home/user/GLP_project/models/qwen3_6_27b_instruct_fp16": {
        "short_name": "qwen3.6-27B",
        "system_message": "You are a helpful assistant. /no_think",
    },
    "/home/user/GLP_project/models/glm_4_7_finetuned_fp16": {
        "short_name": "GLM-4.7-Flash",
        "system_message": "You are a helpful assistant.",
    },
    "/home/user/GLP_project/models/qwen3_5_27b_instruct_fp16": {
        "short_name": "qwen3.5-27B",
        "system_message": "You are a helpful assistant. /no_think",
    },
    "google/gemma-3-27b-it": {
        "short_name": "gemma_3_27b_alpaca_fp16",
        "system_message": "You are a helpful assistant.",
    },
}

NUM_EXAMPLES_PER_CATEGORY = 100
BATCH_SIZE = 25


def load_prompt_template(prompt_path: str) -> str:
    return Path(prompt_path).read_text()


def load_few_shot_examples(seed_dir: str, filename: str) -> str:
    filepath = Path(seed_dir, filename)
    if filepath.exists():
        return filepath.read_text()
    return ""


def subcategory_cycler(subcategories: list[str]) -> Iterator[str]:
    """Infinite iterator that cycles through subcategories."""
    while True:
        random.shuffle(subcategories)
        for subcat in subcategories:
            yield subcat


def build_prompt(
    template: str,
    category: str,
    subcategory: str,
    few_shot_text: str,
) -> str:
    """Build prompt with category and subcategory placeholders."""
    prompt = template.replace("{PRIMARY_REASON}", category)
    prompt = prompt.replace("{SUBCATEGORY_FOCUS}", subcategory)
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
    subcategories: list[str],
    few_shot_text: str,
    num_examples: int,
    batch_size: int,
) -> list[dict]:
    """Generate synthetic notes for a single category using batched inference.
    
    Cycles through subcategories to create diverse data within each category.
    """
    results = []
    total_generated = 0
    attempt = 0
    max_attempts = (num_examples * 2) // batch_size + 1
    
    subcat_iter = subcategory_cycler(subcategories.copy())
    subcategory_counts = {sc: 0 for sc in subcategories}

    while total_generated < num_examples and attempt < max_attempts:
        remaining = num_examples - total_generated
        current_batch = min(batch_size, remaining)
        
        batch_subcats = [next(subcat_iter) for _ in range(current_batch)]
        
        prompts = []
        for subcat in batch_subcats:
            raw_prompt = build_prompt(template, category, subcat, few_shot_text)
            chat_prompt = format_chat_prompt(tokenizer, raw_prompt, system_message)
            prompts.append(chat_prompt)

        print(
            f"  [{category}] Batch {attempt + 1}: generating {current_batch} "
            f"(have {total_generated}/{num_examples})"
        )

        outputs = llm.generate(prompts, sampling_params)

        for output, subcat in zip(outputs, batch_subcats):
            generated_text = output.outputs[0].text
            parsed = extract_json(generated_text)
            if parsed and "clinical_note" in parsed:
                parsed["primary_reason"] = category
                parsed["subcategory_focus"] = subcat
                results.append(parsed)
                subcategory_counts[subcat] += 1
                total_generated += 1
            else:
                print(f"  [{category}] Warning: failed to parse one output, skipping")

        attempt += 1

    if total_generated < num_examples:
        print(
            f"  [{category}] Warning: only generated {total_generated}/{num_examples} "
            f"valid examples after {attempt} attempts"
        )

    print(f"  [{category}] Subcategory distribution:")
    for sc, count in sorted(subcategory_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    - {sc}: {count}")

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


def count_existing_data(output_dir: Path, categories_to_generate: list[str]) -> dict:
    """Count existing records per category and combined file."""
    counts = {}
    for category in categories_to_generate:
        seed_file = CATEGORY_DEFINITIONS[category]["seed_file"]
        cat_file = output_dir / f"{seed_file.replace('.txt', '')}.jsonl"
        counts[category] = len(load_existing_data(cat_file))
    
    combined_file = output_dir / "all_categories_synthetic_notes.jsonl"
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
        description="Generate synthetic clinical notes for ALL GLP-1 discontinuation categories with vLLM"
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
        default="prompts/base_prompt_subcategory.md",
        help="Path to prompt template (should include {SUBCATEGORY_FOCUS} placeholder)",
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
        default="/home/user/GLP_project/GLP_project_ad/data/synthetic_data",
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
        choices=list(CATEGORY_DEFINITIONS.keys()),
        default=None,
        help="Specific categories to generate (default: all categories)",
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Do not merge with existing data (overwrite instead)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    project_root = Path(__file__).parent.parent
    prompt_path = project_root / args.prompt_file
    seed_dir = project_root / args.seed_dir

    short_name = get_model_short_name(args.model)
    output_dir = Path(args.output_dir) / short_name
    output_dir.mkdir(parents=True, exist_ok=True)

    system_message = get_system_message(args.model)

    categories_to_generate = args.categories if args.categories else list(CATEGORY_DEFINITIONS.keys())

    print("=" * 70)
    print("GLP-1 Discontinuation Synthetic Note Generator (All Categories)")
    print("=" * 70)
    print(f"Model:          {args.model}")
    print(f"Short name:     {short_name}")
    print(f"Output dir:     {output_dir}")
    print(f"Examples/cat:   {args.num_examples}")
    print(f"Batch size:     {args.batch_size}")
    print(f"TP size:        {args.tensor_parallel_size}")
    print(f"System msg:     {system_message[:60]}...")
    print(f"Merge mode:     {'Enabled' if not args.no_merge else 'Disabled (overwrite)'}")
    print()
    
    print("=== Categories to Generate ===")
    for cat in categories_to_generate:
        subcats = CATEGORY_DEFINITIONS[cat]["subcategories"]
        print(f"  {cat}")
        print(f"    Subcategories ({len(subcats)}):")
        for sc in subcats:
            print(f"      - {sc}")
    print()

    existing_counts = count_existing_data(output_dir, categories_to_generate)
    print("=== Existing Data in Output Folder ===")
    total_existing = 0
    for category in categories_to_generate:
        count = existing_counts.get(category, 0)
        total_existing += count
        print(f"  {category}: {count} records")
    print(f"  Combined file: {existing_counts.get('_combined', 0)} records")
    print(f"  Total existing (per-category): {total_existing} records")
    if total_existing > 0 and not args.no_merge:
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
        "merge_mode": not args.no_merge,
        "category_definitions": {
            cat: CATEGORY_DEFINITIONS[cat]["subcategories"] 
            for cat in categories_to_generate
        },
        "categories": {},
    }

    total_start = time.time()

    for category in categories_to_generate:
        cat_info = CATEGORY_DEFINITIONS[category]
        seed_file = cat_info["seed_file"]
        subcategories = cat_info["subcategories"]
        
        print(f"=== Generating: {category} ===")
        print(f"    ({len(subcategories)} subcategories)")
        cat_start = time.time()

        few_shot_text = load_few_shot_examples(seed_dir, seed_file)
        if not few_shot_text:
            print(f"  Note: No seed file found at {seed_dir}/{seed_file}, using empty few-shot")
        
        results = generate_for_category(
            llm=llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            system_message=system_message,
            template=template,
            category=category,
            subcategories=subcategories,
            few_shot_text=few_shot_text,
            num_examples=args.num_examples,
            batch_size=args.batch_size,
        )

        cat_elapsed = time.time() - cat_start

        cat_output = output_dir / f"{seed_file.replace('.txt', '')}.jsonl"
        existing_cat_count = existing_counts.get(category, 0) if not args.no_merge else 0
        total_cat_count = save_results(results, cat_output, merge=not args.no_merge)
        print(
            f"  Saved {len(results)} new + {existing_cat_count} existing = {total_cat_count} total "
            f"to {cat_output} ({cat_elapsed:.1f}s)\n"
        )

        all_results.extend(results)
        
        subcat_dist = {}
        for r in results:
            sc = r.get("subcategory_focus", "unknown")
            subcat_dist[sc] = subcat_dist.get(sc, 0) + 1
        
        metadata["categories"][category] = {
            "new_count": len(results),
            "existing_count": existing_cat_count,
            "total_count": total_cat_count,
            "output_file": str(cat_output.name),
            "elapsed_seconds": round(cat_elapsed, 1),
            "subcategory_distribution": subcat_dist,
        }

    combined_output = output_dir / "all_categories_synthetic_notes.jsonl"
    existing_combined_count = existing_counts.get("_combined", 0) if not args.no_merge else 0
    total_combined_count = save_results(all_results, combined_output, merge=not args.no_merge)

    total_elapsed = time.time() - total_start
    metadata["new_examples"] = len(all_results)
    metadata["existing_examples"] = existing_combined_count
    metadata["total_examples"] = total_combined_count
    metadata["total_elapsed_seconds"] = round(total_elapsed, 1)

    meta_path = output_dir / "generation_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"New examples:      {len(all_results)}")
    print(f"Existing examples: {existing_combined_count}")
    print(f"Total examples:    {total_combined_count}")
    print(f"Time elapsed:      {total_elapsed:.1f}s")
    print(f"Combined output:   {combined_output}")
    print(f"Metadata:          {meta_path}")
    print()
    print("Category Summary:")
    for cat in categories_to_generate:
        cat_meta = metadata["categories"][cat]
        print(f"  {cat}: {cat_meta['total_count']} total ({cat_meta['new_count']} new)")


if __name__ == "__main__":
    main()
