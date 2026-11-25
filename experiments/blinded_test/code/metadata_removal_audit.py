#!/usr/bin/env python3
"""Metadata removal audit for test set label leakage.

Checks test.jsonl for:
  1. Labels in filenames/note IDs
  2. Category names appearing in note text (exact string matches)
  3. Prompt remnants (e.g., "reason category:", "primary reason:")
  4. Boilerplate phrases unique to one category (high TF-IDF)
  5. Direct statements about category membership
"""
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from math import log

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_SET = Path("/home/mshahidul/GLP_project/GLP_project_ad/experiments/dataset/test.jsonl")

CATEGORIES = [
    "ADVERSE_EFFECTS_TOLERABILITY",
    "ACCESS_COST_SYSTEM_BARRIERS",
    "EFFICACY_EXPECTATION_MISMATCH",
    "PATIENT_PREFERENCE_BELIEFS",
    "PLANNED_GOAL_BASED_DISCONTINUATION",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "OTHER_UNKNOWN",
]

# Short label forms that might appear in text
LABEL_VARIANTS = {
    "adverse effects": "ADVERSE_EFFECTS_TOLERABILITY",
    "adverse effect": "ADVERSE_EFFECTS_TOLERABILITY",
    "side effects": "ADVERSE_EFFECTS_TOLERABILITY",
    "tolerability": "ADVERSE_EFFECTS_TOLERABILITY",
    "access cost": "ACCESS_COST_SYSTEM_BARRIERS",
    "cost barrier": "ACCESS_COST_SYSTEM_BARRIERS",
    "insurance barrier": "ACCESS_COST_SYSTEM_BARRIERS",
    "efficacy mismatch": "EFFICACY_EXPECTATION_MISMATCH",
    "expectation mismatch": "EFFICACY_EXPECTATION_MISMATCH",
    "not working": "EFFICACY_EXPECTATION_MISMATCH",
    "patient preference": "PATIENT_PREFERENCE_BELIEFS",
    "patient belief": "PATIENT_PREFERENCE_BELIEFS",
    "planned goal": "PLANNED_GOAL_BASED_DISCONTINUATION",
    "goal based": "PLANNED_GOAL_BASED_DISCONTINUATION",
    "clinical decision": "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "medical appropriateness": "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS",
    "other unknown": "OTHER_UNKNOWN",
}

# Prompt remnant patterns
PROMPT_REMNANT_PATTERNS = [
    r"reason category\s*:\s*",
    r"primary reason\s*:\s*",
    r"discontinuation reason\s*:\s*",
    r"category\s*:\s*(adverse|access|efficacy|patient|planned|clinical|other)",
    r"\"primary_reason\"\s*:\s*\"",
    r"assigned_label\s*:\s*",
]


def load_test_set():
    records = []
    with open(TEST_SET) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def check_note_ids(records):
    """Check if note IDs contain label hints."""
    findings = []
    pattern = re.compile(
        r'(adverse|access|efficacy|patient|planned|clinical|other|cost|'
        r'tolerability|barrier|preference|goal|unknown|mismatch)',
        re.IGNORECASE,
    )
    for rec in records:
        nid = rec.get("note_id", "")
        match = pattern.search(nid)
        if match:
            findings.append({
                "note_id": nid,
                "matched_text": match.group(),
                "intended_label": rec["primary_reason"],
            })
    return findings


def check_category_names_in_text(records):
    """Check if exact category label strings appear in note text."""
    findings = []
    for rec in records:
        text = rec["clinical_note"]
        text_lower = text.lower()
        for variant, category in LABEL_VARIANTS.items():
            if variant in text_lower:
                findings.append({
                    "note_id": rec.get("note_id", "unknown"),
                    "matched_variant": variant,
                    "maps_to_category": category,
                    "intended_label": rec["primary_reason"],
                })
    return findings


def check_prompt_remnants(records):
    """Check for prompt remnants that might leak labels."""
    findings = []
    for pattern_str in PROMPT_REMNANT_PATTERNS:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        for rec in records:
            text = rec["clinical_note"]
            matches = pattern.findall(text)
            if matches:
                findings.append({
                    "note_id": rec.get("note_id", "unknown"),
                    "pattern": pattern_str,
                    "matches_found": len(matches),
                    "intended_label": rec["primary_reason"],
                })
    return findings


def check_boilerplate_phrases(records):
    """Find high-TF-IDF phrases unique to one category (potential boilerplate)."""
    from collections import defaultdict
    from collections import Counter as Ctr
    import re

    # Build phrase frequency per category and overall doc frequency
    cat_phrase_counts = {}  # cat -> Counter(phrase -> count)
    cat_total = {}  # cat -> total phrases
    cat_phrase_sets = {}  # cat -> set of unique phrases

    for rec in records:
        cat = rec["primary_reason"]
        text = rec["clinical_note"].lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words) - 1)]
        if cat not in cat_phrase_counts:
            cat_phrase_counts[cat] = Ctr()
            cat_total[cat] = 0
        cat_phrase_counts[cat].update(bigrams)
        cat_total[cat] += len(bigrams)

    # Build set of unique phrases per category and doc frequency
    cat_phrase_sets = {cat: set(counter.keys()) for cat, counter in cat_phrase_counts.items()}
    N = len(cat_phrase_counts)

    # Precompute IDF: doc_count across categories for each phrase
    all_phrases = set()
    for phrases in cat_phrase_sets.values():
        all_phrases.update(phrases)
    print(f"  Unique bigrams across all categories: {len(all_phrases)}")

    idf = {}
    for phrase in all_phrases:
        doc_count = sum(1 for phrases in cat_phrase_sets.values() if phrase in phrases)
        idf[phrase] = log(N / max(doc_count, 1))

    # Efficient TF-IDF per category, only for phrases with doc_count=1
    distinctive_phrases = {}
    for cat in CATEGORIES:
        if cat not in cat_phrase_counts:
            continue
        scores = []
        total = cat_total[cat]
        cat_set = cat_phrase_sets[cat]
        # Only check phrases that appear in exactly this category
        for phrase in cat_set:
            idf_val = idf.get(phrase, 0)
            if idf_val < 0.5:  # Skip low-IDF (appears in multiple categories)
                continue
            tf_val = cat_phrase_counts[cat][phrase] / max(total, 1)
            tfidf = tf_val * idf_val
            if tfidf > 0.005:
                scores.append((phrase, tfidf, cat_phrase_counts[cat][phrase]))
        scores.sort(key=lambda x: -x[1])
        distinctive_phrases[cat] = [
            {"phrase": p, "tfidf": round(s, 4), "frequency": f}
            for p, s, f in scores[:10]
        ]
        if distinctive_phrases[cat]:
            print(f"  {cat}: {len(distinctive_phrases[cat])} distinctive phrases")

    return distinctive_phrases


def check_direct_label_statements(records):
    """Check for direct statements like 'reason category: adverse effects'."""
    direct_patterns = [
        r"(?:reason|category|class|type|label)(?:\s+is|\s*:|\s+of)\s+(?:the\s+)?"
        r"(?:primary\s+)?(?:discontinuation\s+)?(?:reason\s+)?"
        r"(?:is\s+)?(?:adverse\s+effects?|access\s+cost|efficacy|patient\s+preference|"
        r"planned\s+goal|clinical\s+decision|other|unknown)",
        r"(?:this\s+(?:is\s+)?(?:a|an)\s+case\s+of\s+)(?:adverse|access|efficacy|patient|planned|clinical|other)",
    ]
    findings = []
    for pattern_str in direct_patterns:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        for rec in records:
            text = rec["clinical_note"]
            match = pattern.search(text)
            if match:
                findings.append({
                    "note_id": rec.get("note_id", "unknown"),
                    "pattern": pattern_str,
                    "matched_text": match.group(),
                    "intended_label": rec["primary_reason"],
                })
    return findings


def main():
    records = load_test_set()
    print(f"Loaded {len(records)} test records for audit")

    report = []
    report.append("# Metadata Removal Audit Report\n")
    report.append(f"*Dataset:* {TEST_SET}\n")
    report.append(f"*Total records:* {len(records)}\n")
    report.append(f"*Date:* {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}\n")
    report.append("\n---\n")

    # 1. Note ID check
    report.append("## 1. Label Hints in Note IDs\n")
    nid_findings = check_note_ids(records)
    if nid_findings:
        report.append(f"Found {len(nid_findings)} note IDs with potential label-related text:\n")
        report.append("| Note ID | Matched Text | Intended Label |\n")
        report.append("| --- | --- | --- |\n")
        for f in nid_findings:
            report.append(f"| {f['note_id']} | `{f['matched_text']}` | {f['intended_label']} |\n")
    else:
        report.append("No label hints found in note IDs.\n")
    report.append("\n")

    # 2. Category names in text
    report.append("## 2. Category Label Strings in Note Text\n")
    cat_findings = check_category_names_in_text(records)
    if cat_findings:
        report.append(f"Found {len(cat_findings)} instances of label-related phrases in note text:\n")
        report.append("| Note ID | Matched Variant | Maps To | Intended Label |\n")
        report.append("| --- | --- | --- | --- |\n")
        for f in cat_findings:
            report.append(f"| {f['note_id']} | `{f['matched_variant']}` | {f['maps_to_category']} | {f['intended_label']} |\n")
    else:
        report.append("No category label phrases found in note text.\n")
    report.append("\n")

    # 3. Prompt remnants
    report.append("## 3. Prompt Remnants\n")
    prompt_findings = check_prompt_remnants(records)
    if prompt_findings:
        report.append(f"Found {len(prompt_findings)} potential prompt remnants:\n")
        report.append("| Note ID | Pattern | Matches | Intended Label |\n")
        report.append("| --- | --- | --- | --- |\n")
        for f in prompt_findings:
            report.append(f"| {f['note_id']} | `{f['pattern']}` | {f['matches_found']} | {f['intended_label']} |\n")
    else:
        report.append("No prompt remnants found.\n")
    report.append("\n")

    # 4. Distinctive boilerplate
    report.append("## 4. Category-Distinctive Phrases (Potential Boilerplate)\n")
    boilerplate = check_boilerplate_phrases(records)
    has_distinctive = any(v for v in boilerplate.values())
    if has_distinctive:
        for cat, phrases in boilerplate.items():
            if phrases:
                report.append(f"### {cat}\n")
                report.append("| Phrase | TF-IDF | Frequency |\n")
                report.append("| --- | --- | --- |\n")
                for p in phrases[:5]:
                    report.append(f"| `{p['phrase']}` | {p['tfidf']} | {p['frequency']} |\n")
                report.append("\n")
    else:
        report.append("No highly distinctive boilerplate phrases found.\n")
    report.append("\n")

    # 5. Direct label statements
    report.append("## 5. Direct Label Statements\n")
    direct_findings = check_direct_label_statements(records)
    if direct_findings:
        report.append(f"Found {len(direct_findings)} direct label statements:\n")
        for f in direct_findings:
            report.append(f"- **{f['note_id']}**: `{f['matched_text']}` (intended: {f['intended_label']})\n")
    else:
        report.append("No direct label statements found.\n")
    report.append("\n")

    # Summary
    report.append("## Summary\n")
    report.append(f"- Note IDs with label hints: {len(nid_findings)}/{len(records)}\n")
    report.append(f"- Category phrases in text: {len(cat_findings)}/{len(records)}\n")
    report.append(f"- Prompt remnants: {len(prompt_findings)}/{len(records)}\n")
    report.append(f"- Direct label statements: {len(direct_findings)}/{len(records)}\n")
    report.append(f"- Categories with distinctive boilerplate: {sum(1 for v in boilerplate.values() if v)}/{len(CATEGORIES)}\n")
    report.append("\n")
    report.append("*Note: Finding a category-related phrase in note text does not necessarily mean "
                  "the label is leaked — it may reflect clinically appropriate terminology. "
                  "Each finding should be reviewed manually to assess whether it reveals the intended label.*\n")

    out_path = RESULTS_DIR / "metadata_audit_report.md"
    with open(out_path, "w") as f:
        f.write("".join(report))
    print(f"Audit report written to {out_path}")


if __name__ == "__main__":
    main()
