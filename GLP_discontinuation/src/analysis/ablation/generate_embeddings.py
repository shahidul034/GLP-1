#!/usr/bin/env python3
"""
Phase 1: Embedding Generation for Ablation Study
Uses Qwen/Qwen3-Embedding-8B to compute 1024-dim embeddings for all 50,885
synthetic clinical notes. Embeddings are saved to disk to prevent re-computation.
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda"
BATCH_SIZE = 32
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
INPUT_PATH = "/home/user/GLP_project/GLP_project_ad/data/combined_synthetic/combined.jsonl"
OUTPUT_DIR = Path("/home/user/GLP_project/GLP_project_ad/code/ablation_studies/embeddings")


def load_synthetic_data(input_path: str):
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
                print(f"Warning: skipping malformed line {i}")
    return records


def compute_embeddings_batch(model, tokenizer, texts, batch_size=32):
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt"
            )
            encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
            outputs = model(**encoded)
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            all_embeddings.append(embedding.cpu())
    return torch.cat(all_embeddings, dim=0)


def main():
    print("=" * 70)
    print("Phase 1: Embedding Generation")
    print("=" * 70)
    print(f"Model:       {MODEL_NAME}")
    print(f"Input:       {INPUT_PATH}")
    print(f"Output:      {OUTPUT_DIR}")
    print(f"Batch size:  {BATCH_SIZE}")
    print(f"Device:      {DEVICE}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    records = load_synthetic_data(INPUT_PATH)
    print(f"Loaded {len(records)} notes\n")

    texts = [r["clinical_note"] for r in records]
    note_ids = [r["_index"] for r in records]
    categories = [r["primary_reason"] for r in records]

    emb_path = OUTPUT_DIR / "notes_emb.pt"
    ids_path = OUTPUT_DIR / "notes_ids.json"
    cats_path = OUTPUT_DIR / "notes_categories.json"

    if emb_path.exists() and ids_path.exists():
        print("Embeddings already exist, loading from disk...")
        embeddings = torch.load(emb_path)
        note_ids = json.loads(ids_path.read_text())
        categories = json.loads(cats_path.read_text())
        print(f"Loaded embeddings shape: {embeddings.shape}")
    else:
        print("Loading embedding model...")
        start_load = time.time()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = model.to(DEVICE)
        model.eval()
        print(f"Model loaded in {time.time() - start_load:.1f}s\n")

        print("Computing embeddings...")
        start_emb = time.time()
        embeddings = compute_embeddings_batch(model, tokenizer, texts, BATCH_SIZE)
        print(f"Embeddings computed in {time.time() - start_emb:.1f}s")
        print(f"Embedding shape: {embeddings.shape}")

        print("Saving embeddings to disk...")
        torch.save(embeddings.float(), emb_path)
        import json
        Path(ids_path).write_text(json.dumps(note_ids))
        Path(cats_path).write_text(json.dumps(categories))
        print("Embeddings saved.\n")

        del model
        torch.cuda.empty_cache()

    cat_counts = {}
    for c in categories:
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print("Per-category counts:")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")
    print(f"\nTotal: {len(note_ids)} notes, {embeddings.shape[1]}-dim embeddings")
    print("Phase 1 complete.")


if __name__ == "__main__":
    main()