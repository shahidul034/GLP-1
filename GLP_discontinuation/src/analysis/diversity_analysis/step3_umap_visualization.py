"""
Step 3 – Visual Semantic Coverage: UMAP of ClinicalBERT Embeddings
===================================================================
Generates document embeddings with ClinicalBERT (mean-pooling),
projects them with UMAP to 2D, and creates publication-quality scatter plots.
"""
import json
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import umap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import seaborn as sns

from config import (
    RESULTS_DIR, FIGURES_DIR, CATEGORY_ORDER, CATEGORY_COLORS,
    CLINICAL_MODEL, UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
    UMAP_METRIC, UMAP_RANDOM_STATE,
)
from data_loader import load_train, load_test, load_all


def get_embeddings(
    texts: list[str],
    model_name: str = CLINICAL_MODEL,
    batch_size: int = 16,
    max_length: int = 512,
) -> np.ndarray:
    """
    Generate document-level embeddings via mean-pooling of ClinicalBERT
    token embeddings.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"    Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="    Embedding"):
        batch_texts = texts[i : i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)

        # Mean-pooling: average non-padding token embeddings
        attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (B, seq_len, 1)
        token_embeddings = outputs.last_hidden_state               # (B, seq_len, 768)
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pooled = summed / counts

        all_embeddings.append(mean_pooled.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def fit_umap(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings down to 2D with UMAP."""
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
        n_components=2,
    )
    return reducer.fit_transform(embeddings)


def plot_umap(
    coords_2d: np.ndarray,
    categories: list[str],
    title: str,
    filename: str,
):
    """Create a publication-quality UMAP scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create DataFrame for seaborn
    plot_df = pd.DataFrame({
        "UMAP-1": coords_2d[:, 0],
        "UMAP-2": coords_2d[:, 1],
        "Category": categories,
    })

    sns.scatterplot(
        data=plot_df,
        x="UMAP-1", y="UMAP-2",
        hue="Category",
        hue_order=CATEGORY_ORDER,
        palette=CATEGORY_COLORS,
        alpha=0.6,
        s=30,
        edgecolor="none",
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)

    # Legend outside the plot
    ax.legend(
        title="Discontinuation Category",
        title_fontsize=10,
        fontsize=9,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved figure: {out_path}")
    return out_path


def main():
    print("=" * 60)
    print("  Step 3: Visual Semantic Coverage (UMAP)")
    print("=" * 60)
    print(f"  Model: {CLINICAL_MODEL}")
    print(f"  UMAP params: n_neighbors={UMAP_N_NEIGHBORS}, "
          f"min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}")

    # Load data
    df = load_all()

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    # ── Train set embeddings ──────────────────────────────────────────────
    print(f"\n  Generating embeddings for TRAIN set ({len(train_df)} notes)...")
    train_emb = get_embeddings(train_df["clinical_note"].tolist())

    # Save embeddings
    emb_path = os.path.join(RESULTS_DIR, "train_embeddings.npy")
    np.save(emb_path, train_emb)
    print(f"    ✓ Embeddings saved: {emb_path}")

    # UMAP on train
    print("  Fitting UMAP on train embeddings...")
    train_2d = fit_umap(train_emb)
    np.save(os.path.join(RESULTS_DIR, "train_umap_2d.npy"), train_2d)

    plot_umap(
        train_2d,
        train_df["category"].tolist(),
        "UMAP Projection of Synthetic Clinical Notes (Train Set)",
        "umap_train.png",
    )

    # ── Test set embeddings ───────────────────────────────────────────────
    print(f"\n  Generating embeddings for TEST set ({len(test_df)} notes)...")
    test_emb = get_embeddings(test_df["clinical_note"].tolist())

    emb_path_test = os.path.join(RESULTS_DIR, "test_embeddings.npy")
    np.save(emb_path_test, test_emb)
    print(f"    ✓ Embeddings saved: {emb_path_test}")

    # UMAP on test
    print("  Fitting UMAP on test embeddings...")
    test_2d = fit_umap(test_emb)
    np.save(os.path.join(RESULTS_DIR, "test_umap_2d.npy"), test_2d)

    plot_umap(
        test_2d,
        test_df["category"].tolist(),
        "UMAP Projection of Synthetic Clinical Notes (Test Set)",
        "umap_test.png",
    )

    # ── Combined UMAP (Train + Test) ─────────────────────────────────────
    print(f"\n  Fitting UMAP on combined embeddings...")
    combined_emb = np.concatenate([train_emb, test_emb], axis=0)
    combined_2d = fit_umap(combined_emb)
    np.save(os.path.join(RESULTS_DIR, "combined_umap_2d.npy"), combined_2d)

    # Color by category, but shape/marker by split
    combined_cats = train_df["category"].tolist() + test_df["category"].tolist()
    combined_splits = ["Train"] * len(train_df) + ["Test"] * len(test_df)

    plot_df = pd.DataFrame({
        "UMAP-1": combined_2d[:, 0],
        "UMAP-2": combined_2d[:, 1],
        "Category": combined_cats,
        "Split": combined_splits,
    })

    fig, ax = plt.subplots(figsize=(11, 8))
    for cat in CATEGORY_ORDER:
        for split, marker in [("Train", "o"), ("Test", "^")]:
            subset = plot_df[(plot_df["Category"] == cat) & (plot_df["Split"] == split)]
            if len(subset) == 0:
                continue
            ax.scatter(
                subset["UMAP-1"], subset["UMAP-2"],
                c=CATEGORY_COLORS[cat],
                marker=marker,
                alpha=0.5 if split == "Train" else 0.8,
                s=25 if split == "Train" else 50,
                edgecolor="black" if split == "Test" else "none",
                linewidths=0.5 if split == "Test" else 0,
                label=f"{cat} ({split})" if split == "Train" else None,
            )

    ax.set_title("UMAP Projection: Train (●) vs Test (▲)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)

    # Simplified legend: only category colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=CATEGORY_COLORS[c],
                          markersize=8, label=c) for c in CATEGORY_ORDER]
    handles.append(plt.Line2D([0], [0], marker='o', color='gray', markersize=6, linestyle='', label='Train'))
    handles.append(plt.Line2D([0], [0], marker='^', color='gray', markersize=8, linestyle='',
                              markeredgecolor='black', markeredgewidth=0.5, label='Test'))
    ax.legend(handles=handles, title="Legend", fontsize=8, title_fontsize=9,
              bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    plt.tight_layout()
    combined_path = os.path.join(FIGURES_DIR, "umap_combined.png")
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved figure: {combined_path}")

    # ── Save embedding metadata ──────────────────────────────────────────
    meta = {
        "model": CLINICAL_MODEL,
        "embedding_dim": int(train_emb.shape[1]),
        "n_train": int(train_emb.shape[0]),
        "n_test": int(test_emb.shape[0]),
        "umap_params": {
            "n_neighbors": UMAP_N_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "metric": UMAP_METRIC,
            "random_state": UMAP_RANDOM_STATE,
        },
    }
    with open(os.path.join(RESULTS_DIR, "step3_umap_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Step 3 complete.")


if __name__ == "__main__":
    main()
