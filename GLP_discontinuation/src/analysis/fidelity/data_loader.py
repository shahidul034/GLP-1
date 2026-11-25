"""
data_loader.py — Shared data loading utilities for fidelity experiments.

Loads synthetic GLP-1 notes from train.jsonl / test.jsonl and (optionally)
MIMIC reference notes.  Provides reproducible random sampling and basic
corpus-level statistics.
"""

import json
import random
import os
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_synthetic_data(filepath: str, limit: int | None = None) -> list[dict]:
    """Load synthetic notes from a JSONL file.

    Returns a list of dicts with at least 'text' and 'primary_reason' keys.
    """
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("clinical_note") or data.get("text", "")
            if not text:
                continue
            records.append({
                "text": text,
                "primary_reason": data.get("primary_reason", "UNKNOWN"),
                "note_id": data.get("note_id", ""),
            })
            if limit and len(records) >= limit:
                break
    return records


def load_mimic_data(filepath: str, limit: int | None = None) -> list[str]:
    """Load MIMIC reference notes from a JSON or JSONL file.

    Returns a flat list of note texts.
    """
    notes: list[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        if filepath.endswith(".jsonl"):
            for line in f:
                data = json.loads(line)
                text = data.get("cleaned_note") or data.get("text", "")
                if text:
                    notes.append(text)
                if limit and len(notes) >= limit:
                    break
        elif filepath.endswith(".json"):
            data = json.load(f)
            for item in data:
                text = item.get("cleaned_note") or item.get("text", "")
                if text:
                    notes.append(text)
                if limit and len(notes) >= limit:
                    break
        else:
            # Fallback: try reading as plain text, one doc per line
            for line in f:
                stripped = line.strip()
                if stripped:
                    notes.append(stripped)
                if limit and len(notes) >= limit:
                    break
    return notes


# ---------------------------------------------------------------------------
# Sampling / preparation
# ---------------------------------------------------------------------------

def prepare_data(
    synthetic_path: str,
    mimic_path: str | None = None,
    sample_size: int = 1000,
    seed: int = 42,
) -> tuple:
    """Load and optionally subsample synthetic + MIMIC data.

    Parameters
    ----------
    synthetic_path : str
        Path to the synthetic JSONL file.
    mimic_path : str or None
        Path to the MIMIC reference file.  If None, only synthetic data is
        returned.
    sample_size : int
        Maximum number of notes to use from each corpus.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    synthetic_notes : list[str]
        Sampled synthetic note texts.
    mimic_notes : list[str] or None
        Sampled MIMIC note texts, or None if mimic_path is not given.
    synthetic_records : list[dict]
        Full records (text + metadata) for per-category analysis.
    """
    random.seed(seed)

    print(f"Loading synthetic notes from {synthetic_path} ...")
    records = load_synthetic_data(synthetic_path)
    if not records:
        raise ValueError(
            f"No synthetic notes found in {synthetic_path}.  "
            "Check that each line has a 'clinical_note' or 'text' field."
        )

    mimic_notes = None
    if mimic_path and os.path.isfile(mimic_path):
        print(f"Loading MIMIC notes from {mimic_path} ...")
        mimic_notes = load_mimic_data(mimic_path)
        if not mimic_notes:
            raise ValueError(
                f"No MIMIC notes found in {mimic_path}.  "
                "Check that each item has a 'cleaned_note' or 'text' field."
            )
        print(f"  → {len(mimic_notes):,} MIMIC notes loaded.")
        if len(mimic_notes) > sample_size:
            mimic_notes = random.sample(mimic_notes, sample_size)
            print(f"  → Subsampled to {sample_size:,} MIMIC notes.")
    else:
        if mimic_path:
            print(f"[WARN] MIMIC file not found at {mimic_path}; skipping MIMIC comparison.")

    print(f"  → {len(records):,} synthetic notes loaded.")
    if len(records) > sample_size:
        records = random.sample(records, sample_size)
        print(f"  → Subsampled to {sample_size:,} synthetic notes.")

    synthetic_notes = [r["text"] for r in records]

    # Quick corpus summary
    categories = Counter(r["primary_reason"] for r in records)
    print(f"\nCategory distribution (sampled):")
    for cat, cnt in categories.most_common():
        print(f"  {cat}: {cnt}")

    return synthetic_notes, mimic_notes, records


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def ensure_output_dir(output_dir: str = "results") -> Path:
    """Create the output directory (relative to CWD) and return its Path."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict, filepath: str | Path) -> None:
    """Write *data* as pretty-printed JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {filepath}")
