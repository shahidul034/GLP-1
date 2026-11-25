"""
LLM-as-a-Judge evaluation of synthetic clinical notes using vLLM.

Scores each synthetic note on 5 dimensions using a structured rubric,
outputs per-note judgments and aggregate statistics.
"""

import json
import argparse
import time
import re
from pathlib import Path
from collections import defaultdict

from vllm import LLM, SamplingParams


VALID_CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "OTHER_UNKNOWN",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
]

JUDGE_PROMPT_TEMPLATE = """\
You are an expert clinical informaticist reviewing synthetic VA-style clinical notes for quality. \
Your task is to evaluate the following synthetic clinical note on 5 dimensions using a 1-5 Likert scale.

The note was generated with the intended primary discontinuation reason category: **{PRIMARY_REASON}**

--- BEGIN CLINICAL NOTE ---
{CLINICAL_NOTE}
--- END CLINICAL NOTE ---

Evaluate the note on each dimension below. For each, provide a score (1-5) and a brief justification (1-2 sentences).

**Scoring rubric (1-5):**
- 1 = Very poor / completely fails
- 2 = Poor / major issues
- 3 = Acceptable / some issues
- 4 = Good / minor issues
- 5 = Excellent / no issues

**Dimensions:**

1. **clinical_realism**: Does the note read like authentic VA-style clinical documentation? Consider abbreviations, tone, section structure, plausible clinical details, and realistic imperfections.

2. **category_alignment**: Does the note's content clearly reflect the labeled primary reason category ({PRIMARY_REASON})? The reason may be implicit or require inference, but the dominant factor must match. Penalize if the note reflects a different primary category or mixes categories without a clear dominant one.

3. **clinical_coherence**: Is the note internally consistent? Are vitals, labs, medications, history, and plan logically compatible? Penalize contradictions, implausible values, or disjointed reasoning.

4. **note_variability**: Does the note avoid being templatic or formulaic? Consider variation in length, section ordering, tone, level of detail, and whether the GLP-1 issue is foregrounded vs. embedded. A note that feels like a unique patient encounter scores higher.

5. **glp1_relevance**: Is a GLP-1 medication (semaglutide, liraglutide, tirzepatide, dulaglutide, exenatide, etc.) appropriately included? Is the medication scenario (continued, held, tapered, discontinued) clinically reasonable?

Output EXACTLY one valid JSON object with this structure (no markdown fences, no extra text):
{{
  "clinical_realism": {{"score": <int 1-5>, "justification": "<string>"}},
  "category_alignment": {{"score": <int 1-5>, "justification": "<string>"}},
  "clinical_coherence": {{"score": <int 1-5>, "justification": "<string>"}},
  "note_variability": {{"score": <int 1-5>, "justification": "<string>"}},
  "glp1_relevance": {{"score": <int 1-5>, "justification": "<string>"}},
  "overall_accept": <boolean true if all scores >= 3 else false>
}}"""

DIMENSIONS = [
    "clinical_realism",
    "category_alignment",
    "clinical_coherence",
    "note_variability",
    "glp1_relevance",
]

BATCH_SIZE = 16


def load_synthetic_data(input_path: str) -> list[dict]:
    records = []
    with open(input_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                record["_index"] = i
                records.append(record)
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed line {i} in {input_path}")
    return records


def build_judge_prompt(record: dict) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        PRIMARY_REASON=record["primary_reason"],
        CLINICAL_NOTE=record["clinical_note"],
    )


def format_qwen_chat(user_content: str) -> str:
    """Wrap in Qwen3 chat format with thinking disabled for structured output."""
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant. /no_think<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def format_gemma_chat(user_content: str) -> str:
    """Wrap in Gemma chat format."""
    return (
        "<start_of_turn>user\n"
        f"{user_content}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def get_chat_formatter(model_id: str):
    """Return the appropriate chat formatter based on model ID."""
    model_lower = model_id.lower()
    if "gemma" in model_lower:
        return format_gemma_chat
    return format_qwen_chat


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


def validate_judgment(parsed: dict) -> bool:
    """Check that the parsed JSON has the expected structure."""
    for dim in DIMENSIONS:
        if dim not in parsed:
            return False
        entry = parsed[dim]
        if not isinstance(entry, dict):
            return False
        if "score" not in entry or "justification" not in entry:
            return False
        if not isinstance(entry["score"], int) or entry["score"] not in range(1, 6):
            return False
    return True


def judge_batch(
    llm: LLM,
    sampling_params: SamplingParams,
    records: list[dict],
    chat_formatter=format_qwen_chat,
) -> list[dict]:
    """Judge a batch of records, returning parsed judgments."""
    prompts = [chat_formatter(build_judge_prompt(r)) for r in records]
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for record, output in zip(records, outputs):
        generated_text = output.outputs[0].text
        parsed = extract_json(generated_text)

        if parsed and validate_judgment(parsed):
            judgment = {
                "_index": record["_index"],
                "primary_reason": record["primary_reason"],
                "scores": {dim: parsed[dim]["score"] for dim in DIMENSIONS},
                "justifications": {
                    dim: parsed[dim]["justification"] for dim in DIMENSIONS
                },
                "overall_accept": parsed.get("overall_accept", all(
                    parsed[dim]["score"] >= 3 for dim in DIMENSIONS
                )),
            }
        else:
            judgment = {
                "_index": record["_index"],
                "primary_reason": record["primary_reason"],
                "scores": {dim: -1 for dim in DIMENSIONS},
                "justifications": {dim: "PARSE_FAILURE" for dim in DIMENSIONS},
                "overall_accept": False,
                "_parse_error": True,
                "_raw_output": generated_text[:500],
            }
            print(f"  Warning: failed to parse judgment for index {record['_index']}")

        results.append(judgment)
    return results


def compute_statistics(judgments: list[dict]) -> dict:
    """Compute aggregate statistics from all judgments."""
    stats = {
        "total": len(judgments),
        "parse_failures": sum(1 for j in judgments if j.get("_parse_error")),
        "overall_accept_count": sum(1 for j in judgments if j["overall_accept"]),
    }

    valid = [j for j in judgments if not j.get("_parse_error")]
    stats["valid_count"] = len(valid)

    if valid:
        stats["overall_accept_rate"] = round(
            stats["overall_accept_count"] / stats["total"], 4
        )

        global_scores = defaultdict(list)
        per_category = defaultdict(lambda: defaultdict(list))

        for j in valid:
            cat = j["primary_reason"]
            for dim in DIMENSIONS:
                score = j["scores"][dim]
                global_scores[dim].append(score)
                per_category[cat][dim].append(score)

        stats["global_mean_scores"] = {
            dim: round(sum(scores) / len(scores), 3)
            for dim, scores in global_scores.items()
        }
        stats["global_mean_overall"] = round(
            sum(
                sum(scores) / len(scores) for scores in global_scores.values()
            ) / len(DIMENSIONS),
            3,
        )

        stats["per_category"] = {}
        for cat, dim_scores in per_category.items():
            cat_stats = {
                "count": len(next(iter(dim_scores.values()))),
                "mean_scores": {
                    dim: round(sum(s) / len(s), 3)
                    for dim, s in dim_scores.items()
                },
                "accept_rate": round(
                    sum(
                        1
                        for j in valid
                        if j["primary_reason"] == cat and j["overall_accept"]
                    )
                    / max(
                        1,
                        sum(1 for j in valid if j["primary_reason"] == cat),
                    ),
                    4,
                ),
            }
            cat_stats["mean_overall"] = round(
                sum(cat_stats["mean_scores"].values()) / len(DIMENSIONS), 3
            )
            stats["per_category"][cat] = cat_stats

        score_dist = defaultdict(lambda: defaultdict(int))
        for j in valid:
            for dim in DIMENSIONS:
                score_dist[dim][j["scores"][dim]] += 1
        stats["score_distribution"] = {
            dim: dict(sorted(dist.items())) for dim, dist in score_dist.items()
        }

    return stats


def print_summary(stats: dict) -> None:
    print("\n" + "=" * 70)
    print("JUDGE SUMMARY")
    print("=" * 70)
    print(f"Total notes judged:   {stats['total']}")
    print(f"Valid judgments:       {stats['valid_count']}")
    print(f"Parse failures:       {stats['parse_failures']}")
    print(f"Overall accept rate:  {stats.get('overall_accept_rate', 'N/A')}")
    print()

    if "global_mean_scores" in stats:
        print("Global mean scores:")
        for dim, score in stats["global_mean_scores"].items():
            print(f"  {dim:<25s} {score:.3f}")
        print(f"  {'OVERALL':<25s} {stats['global_mean_overall']:.3f}")
        print()

    if "per_category" in stats:
        print("Per-category breakdown:")
        for cat, cat_stats in stats["per_category"].items():
            print(f"\n  {cat} (n={cat_stats['count']}):")
            for dim, score in cat_stats["mean_scores"].items():
                print(f"    {dim:<25s} {score:.3f}")
            print(f"    {'OVERALL':<25s} {cat_stats['mean_overall']:.3f}")
            print(f"    {'accept_rate':<25s} {cat_stats['accept_rate']:.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Judge synthetic clinical notes with LLM-as-a-Judge via vLLM"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-27B",
        help="HuggingFace model ID for the judge",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="synthetic_data/all_synthetic_notes.jsonl",
        help="Input JSONL file with synthetic notes",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="judge_results",
        help="Output directory for judgments and statistics",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for vLLM inference",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input
    model_name = input_path.parent.name
    output_dir = project_root / args.output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    chat_formatter = get_chat_formatter(args.model)
    fmt_name = "gemma" if "gemma" in args.model.lower() else "qwen"

    print(f"Judge model:  {args.model}")
    print(f"Chat format:  {fmt_name}")
    print(f"Input:        {input_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Batch size:   {args.batch_size}")
    print()

    records = load_synthetic_data(input_path)
    print(f"Loaded {len(records)} synthetic notes to judge")

    category_counts = defaultdict(int)
    for r in records:
        category_counts[r["primary_reason"]] += 1
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
    print()

    print("Loading judge model...")
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
    print("Judge model loaded.\n")

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.95,
        max_tokens=1024,
        seed=args.seed,
    )

    all_judgments = []
    total_start = time.time()

    for batch_start in range(0, len(records), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(records))
        batch = records[batch_start:batch_end]

        print(
            f"Judging batch {batch_start // args.batch_size + 1} "
            f"(notes {batch_start + 1}-{batch_end} of {len(records)})"
        )

        judgments = judge_batch(llm, sampling_params, batch, chat_formatter)
        all_judgments.extend(judgments)

    total_elapsed = time.time() - total_start

    judgments_path = output_dir / "judgments.jsonl"
    with open(judgments_path, "w") as f:
        for j in all_judgments:
            f.write(json.dumps(j) + "\n")

    stats = compute_statistics(all_judgments)
    stats["judge_model"] = args.model
    stats["input_file"] = str(input_path.name)
    stats["elapsed_seconds"] = round(total_elapsed, 1)

    stats_path = output_dir / "judge_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    per_cat_dir = output_dir / "per_category"
    per_cat_dir.mkdir(exist_ok=True)
    cat_judgments = defaultdict(list)
    for j in all_judgments:
        cat_judgments[j["primary_reason"]].append(j)
    for cat, jlist in cat_judgments.items():
        cat_path = per_cat_dir / f"{cat.lower()}.jsonl"
        with open(cat_path, "w") as f:
            for j in jlist:
                f.write(json.dumps(j) + "\n")

    print_summary(stats)
    print(f"\nElapsed: {total_elapsed:.1f}s")
    print(f"Judgments: {judgments_path}")
    print(f"Statistics: {stats_path}")
    print(f"Per-category: {per_cat_dir}/")


if __name__ == "__main__":
    main()
