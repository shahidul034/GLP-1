"""
Step 2 – Semantic Diversity: Pairwise Semantic Similarity
==========================================================
Computes intra-category pairwise cosine similarity using ClinicalBERT 
to detect mode collapse. High scores (>0.90) indicate mode collapse (bad).
(Note: Replaced bert_score with direct cosine similarity due to Python 3.13 tokenizer incompatibilities).
"""
import json
import random
import itertools
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from config import (
    RESULTS_DIR, CATEGORY_ORDER, CLINICAL_MODEL,
    BERTSCORE_SAMPLE_PER_CATEGORY, BERTSCORE_MAX_PAIRS,
)
from data_loader import load_train


def get_embeddings(texts: list[str], model_name: str = CLINICAL_MODEL) -> np.ndarray:
    """Generate mean-pooled embeddings for a list of texts."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Truncate aggressively to avoid OOM or slow processing on CPU/MPS
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)

    attention_mask = encoded["attention_mask"].unsqueeze(-1)
    token_embeddings = outputs.last_hidden_state
    summed = (token_embeddings * attention_mask).sum(dim=1)
    counts = attention_mask.sum(dim=1)
    mean_pooled = summed / counts

    return mean_pooled.cpu().numpy()


def compute_pairwise_similarity(notes: list[str], max_pairs: int = BERTSCORE_MAX_PAIRS) -> dict:
    """Compute pairwise cosine similarity between all note pairs."""
    # Generate all unique pairs
    all_pairs = list(itertools.combinations(range(len(notes)), 2))

    # Subsample if too many
    if len(all_pairs) > max_pairs:
        all_pairs = random.sample(all_pairs, max_pairs)

    embeddings = get_embeddings(notes)
    
    # Compute cosine similarities for the sampled pairs
    sims = []
    for i, j in all_pairs:
        emb1 = embeddings[i]
        emb2 = embeddings[j]
        # Cosine similarity
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sims.append(float(sim))

    sims = np.array(sims)

    return {
        "n_notes":      len(notes),
        "n_pairs":      len(all_pairs),
        "mean_sim":     round(float(np.mean(sims)), 6),
        "std_sim":      round(float(np.std(sims)), 6),
        "median_sim":   round(float(np.median(sims)), 6),
        "min_sim":      round(float(np.min(sims)), 6),
        "max_sim":      round(float(np.max(sims)), 6),
    }


def main():
    print("=" * 60)
    print("  Step 2: Semantic Diversity (Pairwise Similarity)")
    print("=" * 60)
    print(f"  Model: {CLINICAL_MODEL}")
    print(f"  Sample per category: {BERTSCORE_SAMPLE_PER_CATEGORY}")
    print(f"  Max pairs per category: {BERTSCORE_MAX_PAIRS}")

    df = load_train()
    results = {}

    print(f"\n  {'Category':<22} {'Notes':>6} {'Pairs':>6} {'Mean Sim':>9} {'Std':>8} {'Med':>8} {'Range':>16}")
    print(f"  {'-'*76}")

    for cat in CATEGORY_ORDER:
        cat_df = df[df["category"] == cat]
        if len(cat_df) == 0:
            continue

        notes = cat_df["clinical_note"].tolist()
        if len(notes) > BERTSCORE_SAMPLE_PER_CATEGORY:
            random.seed(42)
            notes = random.sample(notes, BERTSCORE_SAMPLE_PER_CATEGORY)

        cat_result = compute_pairwise_similarity(notes)
        results[cat] = cat_result

        print(f"  {cat:<22} {cat_result['n_notes']:>6} {cat_result['n_pairs']:>6} "
              f"{cat_result['mean_sim']:>9.4f} {cat_result['std_sim']:>8.4f} "
              f"{cat_result['median_sim']:>8.4f} [{cat_result['min_sim']:.3f}, {cat_result['max_sim']:.3f}]")

    # Save
    out_path = f"{RESULTS_DIR}/step2_semantic_diversity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
