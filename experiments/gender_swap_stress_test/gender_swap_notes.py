#!/usr/bin/env python3
"""
gender_swap_notes.py

Detect gender-coded language in clinical notes and use Gemma-4-31B-it
(via local vLLM on port 8091) to rewrite gender-coded language while
preserving all clinical content and labels.

Usage:
    CUDA_VISIBLE_DEVICES=7 python gender_swap_notes.py
"""

import os, sys, json, re, asyncio, aiohttp, time, random
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
from collections import Counter, defaultdict

BASE_DIR = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/gender_swap_stress_test"
DATA_DIR = f"{BASE_DIR}/data"
os.makedirs(DATA_DIR, exist_ok=True)

TRAIN_SRC = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/dataset/train.jsonl"
TEST_SRC = "/home/mshahidul/GLP_project/GLP_project_ad/experiments/dataset/test.jsonl"

GEMMA_URL = "http://localhost:8091/v1/chat/completions"
GEMMA_MODEL = "google/gemma-4-31B-it"
MAX_CONCURRENT = 20
MAX_RETRIES = 3

MALE_PRONOUNS = re.compile(r'\b(he|him|his|himself)\b', re.IGNORECASE)
FEMALE_PRONOUNS = re.compile(r'\b(she|her|hers|herself)\b', re.IGNORECASE)
MALE_NOUNS = re.compile(r'\b(man|men|male|gentleman|father|husband|boy|son|brother)\b', re.IGNORECASE)
FEMALE_NOUNS = re.compile(r'\b(woman|women|female|lady|mother|wife|girl|daughter|sister)\b', re.IGNORECASE)

MEDICATION_KEYWORDS = [
    'semaglutide', 'Ozempic', 'Wegovy', 'liraglutide', 'Saxenda', 'Victoza',
    'dulaglutide', 'Trulicity', 'tirzepatide', 'Mounjaro', 'Zepbound',
    'exenatide', 'Byetta', 'Bydureon', 'orforglipron', 'retatrutide',
    'metformin', 'GLP-1', 'GLP-1RA', 'insulin', 'glargine', 'Lantus',
]

PRIMARY_REASON_KEYWORDS = {
    'ADVERSE_EFFECTS_TOLERABILITY': ['nausea', 'vomiting', 'diarrhea', 'gi', 'abdominal pain',
                                       'injection site', 'side effect', 'intolerance'],
    'ACCESS_COST_SYSTEM_BARRIERS': ['insurance', 'cost', 'prior authorization', 'copay',
                                      'deductible', 'shortage', 'backorder', 'formulary'],
    'EFFICACY_EXPECTATION_MISMATCH': ['plateau', 'not working', 'ineffective', 'no weight loss',
                                        'insufficient', 'a1c not improved'],
    'PATIENT_PREFERENCE_BELIEFS': ['preference', 'fear', 'concerned', 'does not want',
                                     'decided to stop', 'natural', 'declined'],
    'PLANNED_GOAL_BASED_DISCONTINUATION': ['goal', 'target weight', 'course complete', 'taper'],
    'CLINICAL_DECISION_MEDICAL_APPROPRIATENESS': ['contraindication', 'pregnancy', 'switch',
                                                     'provider recommended', 'clinical decision'],
    'OTHER_UNKNOWN': [],
}

LABELS = list(PRIMARY_REASON_KEYWORDS.keys())


def detect_gender(note):
    male_pron = len(MALE_PRONOUNS.findall(note))
    female_pron = len(FEMALE_PRONOUNS.findall(note))
    male_nouns = len(MALE_NOUNS.findall(note))
    female_nouns = len(FEMALE_NOUNS.findall(note))
    male_score = male_pron + male_nouns * 2
    female_score = female_pron + female_nouns * 2
    if male_score > female_score:
        return "male"
    elif female_score > male_score:
        return "female"
    else:
        if male_score > 0 and female_score == 0:
            return "male"
        elif female_score > 0 and male_score == 0:
            return "female"
        else:
            return "neutral"


def build_gemma_prompt(item_id, label, source_gender, target_gender, clinical_note):
    prompt = f"""You are editing synthetic outpatient EHR-style clinical notes for a controlled NLP robustness experiment.

Task:
Rewrite the note by changing only gender-coded language from {source_gender} to {target_gender}.

Strict rules:
1. Preserve the original GLP-1 discontinuation reason and all clinical facts.
2. Do not add new symptoms, diagnoses, medications, labs, dates, or reasoning.
3. Do not remove the discontinuation evidence.
4. Preserve note structure, section headers, and clinical style.
5. Change only pronouns, gendered nouns, and directly related grammar.
6. If a sex-specific clinical detail makes the swap medically incoherent, return "FLAG_FOR_REVIEW" and explain briefly.
7. Return valid JSON only.

Input JSON:
{{
  "id": "{item_id}",
  "label": "{label}",
  "source_gender": "{source_gender}",
  "target_gender": "{target_gender}",
  "clinical_note": {json.dumps(clinical_note)}
}}

Output JSON:
{{
  "id": "{item_id}",
  "label": "{label}",
  "swap_status": "swapped" | "unchanged_neutral" | "flag_for_review",
  "source_gender": "{source_gender}",
  "target_gender": "{target_gender}",
  "clinical_note_swapped": "...",
  "change_summary": "brief summary of gender-coded edits only"
}}"""
    return prompt


async def call_gemma(session, semaphore, prompt, retries=MAX_RETRIES):
    async with semaphore:
        for attempt in range(retries):
            try:
                payload = {
                    "model": GEMMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 8192,
                }
                async with session.post(GEMMA_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return content
                    else:
                        text = await resp.text()
                        if attempt < retries - 1:
                            await asyncio.sleep(2 ** attempt)
                        else:
                            return None
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
    return None


def parse_gemma_output(raw_output, item_id, label):
    if raw_output is None:
        return {"id": item_id, "label": label, "swap_status": "parse_failure",
                "error": "None response from API", "clinical_note_swapped": None}

    cleaned = re.sub(r'```json\s*', '', raw_output)
    cleaned = re.sub(r'```\s*', '', cleaned).strip()
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if not json_match:
        # Try the entire output as JSON
        try:
            result = json.loads(cleaned)
        except:
            return {"id": item_id, "label": label, "swap_status": "parse_failure",
                    "error": f"No JSON found in: {raw_output[:200]}", "clinical_note_swapped": None, "_raw": raw_output[:500]}
    else:
        try:
            result = json.loads(json_match.group())
        except:
            return {"id": item_id, "label": label, "swap_status": "parse_failure",
                    "error": f"JSON parse error", "_raw": raw_output[:500]}

    swap_status = result.get("swap_status", "unknown")
    if swap_status not in ("swapped", "unchanged_neutral", "flag_for_review"):
        swap_status = "flag_for_review"

    return {
        "id": result.get("id", item_id),
        "label": result.get("label", label),
        "swap_status": swap_status,
        "source_gender": result.get("source_gender", ""),
        "target_gender": result.get("target_gender", ""),
        "clinical_note_swapped": result.get("clinical_note_swapped", ""),
        "change_summary": result.get("change_summary", ""),
    }


def check_label_preserved(original_label, parsed):
    return original_label == parsed.get("label")


def check_non_empty(note):
    return bool(note and note.strip())


def check_length_ratio(original, swapped):
    if not swapped:
        return False, 0
    ratio = len(swapped) / len(original)
    return 0.85 <= ratio <= 1.15, ratio


def check_medications_preserved(original, swapped):
    if not swapped:
        return False, []
    missing = []
    for med in MEDICATION_KEYWORDS:
        if re.search(re.escape(med), original, re.IGNORECASE) and not re.search(re.escape(med), swapped, re.IGNORECASE):
            missing.append(med)
    return len(missing) == 0, missing


def check_pronoun_direction(swapped, target_gender):
    if not swapped:
        return False
    female_pron = len(FEMALE_PRONOUNS.findall(swapped))
    male_pron = len(MALE_PRONOUNS.findall(swapped))
    if target_gender == "female":
        return female_pron > male_pron if (female_pron + male_pron) > 0 else True
    elif target_gender == "male":
        return male_pron > female_pron if (female_pron + male_pron) > 0 else True
    return True


def check_reason_keywords(original_note, swapped_note, label):
    if not swapped_note:
        return False, []
    keywords = PRIMARY_REASON_KEYWORDS.get(label, [])
    if not keywords:
        return True, []
    missing = []
    for kw in keywords:
        if re.search(re.escape(kw), original_note, re.IGNORECASE) and not re.search(re.escape(kw), swapped_note, re.IGNORECASE):
            missing.append(kw)
    return len(missing) == 0, missing


def run_quality_checks(original, parsed, label, target_gender):
    checks = {}
    checks["label_preserved"] = check_label_preserved(label, parsed)
    checks["non_empty"] = check_non_empty(parsed.get("clinical_note_swapped"))
    length_ok, length_ratio = check_length_ratio(original["clinical_note"], parsed.get("clinical_note_swapped", ""))
    checks["length_ratio_ok"] = length_ok
    checks["length_ratio"] = length_ratio
    med_ok, missing_meds = check_medications_preserved(original["clinical_note"], parsed.get("clinical_note_swapped", ""))
    checks["medications_preserved"] = med_ok
    checks["missing_medications"] = missing_meds
    reason_ok, missing_reason = check_reason_keywords(original["clinical_note"], parsed.get("clinical_note_swapped", ""), label)
    checks["reason_keywords_preserved"] = reason_ok
    checks["missing_reason_keywords"] = missing_reason
    checks["pronoun_direction_ok"] = check_pronoun_direction(parsed.get("clinical_note_swapped", ""), target_gender)
    checks["passed_all"] = all([
        checks["label_preserved"],
        checks["non_empty"],
        checks["length_ratio_ok"],
        checks["medications_preserved"],
        checks["reason_keywords_preserved"],
        checks["pronoun_direction_ok"],
    ])
    return checks


async def process_item(session, semaphore, item, source_gender, target_gender, idx):
    item_id = item.get("note_id", f"note_{idx}")
    label = item["primary_reason"]
    clinical_note = item["clinical_note"]

    if source_gender == target_gender:
        return {
            "id": item_id, "label": label, "swap_status": "unchanged_neutral",
            "source_gender": source_gender, "target_gender": target_gender,
            "clinical_note_swapped": clinical_note,
            "change_summary": "No change needed (neutral gender)",
            "quality_checks": {"passed_all": True},
        }

    prompt = build_gemma_prompt(item_id, label, source_gender, target_gender, clinical_note)
    raw_output = await call_gemma(session, semaphore, prompt)
    parsed = parse_gemma_output(raw_output, item_id, label)

    if parsed["swap_status"] == "parse_failure":
        checks = {"passed_all": False, "error": parsed.get("error", "")}
    else:
        checks = run_quality_checks(item, parsed, label, target_gender)
        if not checks["passed_all"]:
            parsed["swap_status"] = "flag_for_review"

    parsed["quality_checks"] = checks
    return parsed


def compile_stats(results, items, split_name):
    stats = {
        "split": split_name,
        "total": len(results),
        "swapped": sum(1 for r in results if r["swap_status"] == "swapped"),
        "unchanged_neutral": sum(1 for r in results if r["swap_status"] == "unchanged_neutral"),
        "flag_for_review": sum(1 for r in results if r["swap_status"] == "flag_for_review"),
        "parse_failure": sum(1 for r in results if r["swap_status"] == "parse_failure"),
        "quality_passed": sum(1 for r in results if r.get("quality_checks", {}).get("passed_all")),
        "gender_distribution": {},
    }

    gender_dist = Counter()
    for item in items:
        gender_dist[item.get("_detected_gender", "unknown")] += 1
    stats["gender_distribution"] = dict(gender_dist)

    label_gender_dist = defaultdict(lambda: Counter())
    for item in items:
        label_gender_dist[item["primary_reason"]][item.get("_detected_gender", "unknown")] += 1
    stats["label_gender_distribution"] = {k: dict(v) for k, v in label_gender_dist.items()}

    return stats


def save_swapped_data(results, output_path):
    """Save the swapped notes (simple schema) keeping only successful swaps/unchanged."""
    count = 0
    with open(output_path, 'w') as f:
        for r in results:
            if r["swap_status"] == "parse_failure":
                continue
            note = r.get("clinical_note_swapped", "")
            if not note:
                continue
            out = {
                "clinical_note": note,
                "primary_reason": r["label"],
            }
            if "id" in r and r["id"]:
                out["note_id"] = r["id"]
            f.write(json.dumps(out) + '\n')
            count += 1
    print(f"  Saved {count} notes to {output_path}")
    return count


def save_augmented_data(original_items, results, output_path):
    """Save original + gender-swapped notes (augmented dataset)."""
    count = 0
    seen_ids = set()
    with open(output_path, 'w') as f:
        for item in original_items:
            out = {"clinical_note": item["clinical_note"], "primary_reason": item["primary_reason"]}
            note_id = item.get("note_id", "")
            if note_id:
                out["note_id"] = note_id
            f.write(json.dumps(out) + '\n')
            count += 1
            if note_id:
                seen_ids.add(note_id)

        for r in results:
            if r["swap_status"] in ("parse_failure",):
                continue
            note = r.get("clinical_note_swapped", "")
            if not note:
                continue
            note_id = r.get("id", "")
            if note_id and note_id in seen_ids:
                key = note_id + "_swapped"
            else:
                key = note_id + "_swapped" if note_id else f"aug_{count}"
            out = {"clinical_note": note, "primary_reason": r["label"]}
            out["note_id"] = key
            if note_id:
                seen_ids.add(key)
            f.write(json.dumps(out) + '\n')
            count += 1
    print(f"  Saved {count} notes (original + swapped) to {output_path}")
    return count


def save_log(results, output_path):
    log_entries = []
    for r in results:
        entry = {
            "id": r.get("id", ""),
            "label": r.get("label", ""),
            "swap_status": r.get("swap_status", ""),
            "source_gender": r.get("source_gender", ""),
            "target_gender": r.get("target_gender", ""),
            "change_summary": r.get("change_summary", ""),
            "quality_passed": r.get("quality_checks", {}).get("passed_all", False),
        }
        if "error" in r.get("quality_checks", {}):
            entry["error"] = r["quality_checks"]["error"]
        log_entries.append(entry)
    with open(output_path, 'w') as f:
        json.dump(log_entries, f, indent=2)
    print(f"  Saved log to {output_path}")


def print_stats(stats, split_name):
    print(f"\n--- {split_name} Statistics ---", flush=True)
    print(f"  Total: {stats['total']}", flush=True)
    print(f"  Swapped: {stats['swapped']}", flush=True)
    print(f"  Unchanged (neutral): {stats['unchanged_neutral']}", flush=True)
    print(f"  Flagged for review: {stats['flag_for_review']}", flush=True)
    print(f"  Parse failures: {stats['parse_failure']}", flush=True)
    print(f"  Quality checks passed: {stats['quality_passed']}", flush=True)
    print(f"  Gender distribution: {stats['gender_distribution']}", flush=True)
    print(f"  Label x Gender:", flush=True)
    for label, dist in stats.get("label_gender_distribution", {}).items():
        print(f"    {label}: {dist}", flush=True)


def save_intermediate(items, results, split_name):
    """Save swapped data immediately after processing a split."""
    save_swapped_data(results, f"{DATA_DIR}/{split_name}_gender_swapped.jsonl")
    stats = compile_stats(results, items, split_name)
    print_stats(stats, split_name)
    return stats


CHECKPOINT_INTERVAL = 200


def save_checkpoint(results, items, split_name, suffix=""):
    path = f"{DATA_DIR}/{split_name}_gender_swapped{suffix}.jsonl"
    count = save_swapped_data(results, path)
    stats = compile_stats(results, items, split_name)
    with open(f"{DATA_DIR}/{split_name}_checkpoint_stats{suffix}.json", 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Checkpoint saved: {count} notes to {path}", flush=True)
    return stats


async def process_split(items, split_name):
    """Process a single split (train or test) and save results."""
    print(f"\n{'='*60}", flush=True)
    print(f"Processing {split_name}: {len(items)} notes", flush=True)
    print(f"{'='*60}", flush=True)

    for item in items:
        item["_detected_gender"] = detect_gender(item["clinical_note"])

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for idx, item in enumerate(items):
            src_gender = item["_detected_gender"]
            if src_gender == "male":
                tgt_gender = "female"
            elif src_gender == "female":
                tgt_gender = "male"
            else:
                tgt_gender = src_gender
            tasks.append(process_item(session, semaphore, item, src_gender, tgt_gender, idx))

        results = []
        total = len(tasks)
        start_time = time.time()
        last_checkpoint = 0
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{total}] {rate:.1f} notes/s, ETA: {eta:.0f}s", flush=True)
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(results[:i+1], items[:i+1], split_name, f"_partial_{i+1}")

    return results


async def main():
    print("Loading datasets...", flush=True)
    train_items = [json.loads(line) for line in open(TRAIN_SRC) if line.strip()]
    test_items = [json.loads(line) for line in open(TEST_SRC) if line.strip()]
    print(f"  Train: {len(train_items)} notes", flush=True)
    print(f"  Test: {len(test_items)} notes", flush=True)

    # Process train
    train_results = await process_split(train_items, "train")
    train_stats = save_intermediate(train_items, train_results, "train")

    # Save augmented dataset early (needs train_results)
    save_augmented_data(train_items, train_results, f"{DATA_DIR}/train_gender_balanced_augmented.jsonl")

    # Process test
    test_results = await process_split(test_items, "test")
    test_stats = save_intermediate(test_items, test_results, "test")

    # Save combined analysis and log
    analysis = {"train": train_stats, "test": test_stats}
    with open(f"{DATA_DIR}/gender_distribution_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\nSaved gender distribution analysis to {DATA_DIR}/gender_distribution_analysis.json", flush=True)

    save_log(train_results + test_results, f"{DATA_DIR}/gender_swap_log.json")

    print("\nDone! Gender-swapped datasets created.", flush=True)


async def main_split(split_name):
    print("Loading datasets...", flush=True)
    train_items = [json.loads(line) for line in open(TRAIN_SRC) if line.strip()]
    test_items = [json.loads(line) for line in open(TEST_SRC) if line.strip()]
    print(f"  Train: {len(train_items)} notes", flush=True)
    print(f"  Test: {len(test_items)} notes", flush=True)

    if split_name == "train":
        items = train_items
        results = await process_split(items, "train")
        save_intermediate(items, results, "train")
        # Also save augmented data (needs original + swapped)
        save_augmented_data(items, results, f"{DATA_DIR}/train_gender_balanced_augmented.jsonl")
    elif split_name == "test":
        items = test_items
        results = await process_split(items, "test")
        save_intermediate(items, results, "test")

    print(f"\nDone processing {split_name}.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    if args.split == "both":
        asyncio.run(main())
    else:
        asyncio.run(main_split(args.split))
