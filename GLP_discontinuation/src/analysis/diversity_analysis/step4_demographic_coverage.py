"""
Step 4 – Clinical & Demographic Coverage
=========================================
Analyzes the distribution of discontinuation categories and extracts
demographic/clinical metadata if present in the notes (Age, Sex, Race, etc.)
using regex-based heuristics tailored to clinical note formatting.
"""
import json
import re
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from config import (
    RESULTS_DIR, FIGURES_DIR, CATEGORY_ORDER, CATEGORY_COLORS,
    CATEGORY_SHORT,
)
from data_loader import load_train, load_test, load_all


# ── Regex-based demographic extraction ────────────────────────────────────
def extract_age(text: str) -> int | None:
    """Extract patient age from clinical note text."""
    patterns = [
        r'(\d{1,3})\s*[-]?\s*(?:year|yr|y/?o|y\.o\.)\s*(?:[-]?\s*old)?',
        r'(?:age|aged)\s*[:=]?\s*(\d{1,3})',
        r'(\d{2,3})\s*(?:yo|y/o|yof|yom)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            if 18 <= age <= 110:
                return age
    return None


def extract_sex(text: str) -> str | None:
    """Extract patient sex/gender from clinical note text."""
    text_lower = text.lower()
    male_patterns = [
        r'\b(?:male|man|gentleman|he |his )\b',
        r'\d+\s*(?:y/?o|year)\s*(?:old)?\s*male',
        r'\bmale veteran\b',
        r'\byom\b',
    ]
    female_patterns = [
        r'\b(?:female|woman|lady|she |her )\b',
        r'\d+\s*(?:y/?o|year)\s*(?:old)?\s*female',
        r'\bfemale veteran\b',
        r'\byof\b',
    ]

    male_score = sum(1 for p in male_patterns if re.search(p, text_lower))
    female_score = sum(1 for p in female_patterns if re.search(p, text_lower))

    if male_score > female_score:
        return "Male"
    elif female_score > male_score:
        return "Female"
    return None


def extract_bmi(text: str) -> float | None:
    """Extract BMI value from clinical note text."""
    patterns = [
        r'BMI\s*[:=]?\s*(\d{2,3}(?:\.\d{1,2})?)',
        r'body mass index\s*[:=]?\s*(\d{2,3}(?:\.\d{1,2})?)',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            bmi = float(match.group(1))
            if 15 <= bmi <= 70:
                return bmi
    return None


def extract_race_ethnicity(text: str) -> str | None:
    """Extract race/ethnicity if mentioned in the note."""
    text_lower = text.lower()
    race_map = {
        "White/Caucasian": [r'\b(?:white|caucasian)\b'],
        "Black/African American": [r'\b(?:black|african[- ]american|aa)\b'],
        "Hispanic/Latino": [r'\b(?:hispanic|latino|latina|latinx)\b'],
        "Asian": [r'\b(?:asian|chinese|japanese|korean|vietnamese|filipino)\b'],
        "Native American": [r'\b(?:native american|american indian|indigenous)\b'],
        "Pacific Islander": [r'\b(?:pacific islander|hawaiian|samoan)\b'],
    }
    for race, patterns in race_map.items():
        for pat in patterns:
            if re.search(pat, text_lower):
                return race
    return None


def extract_a1c(text: str) -> float | None:
    """Extract HbA1c value from clinical note text."""
    patterns = [
        r'(?:A1[cC]|HbA1[cC]|hemoglobin A1[cC])\s*[:=]?\s*(\d{1,2}(?:\.\d{1,2})?)\s*%?',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if 4.0 <= val <= 20.0:
                return val
    return None


def extract_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all demographics from clinical notes."""
    records = []
    for _, row in df.iterrows():
        text = row["clinical_note"]
        records.append({
            "note_id": row["note_id"],
            "category": row["category"],
            "age": extract_age(text),
            "sex": extract_sex(text),
            "bmi": extract_bmi(text),
            "race": extract_race_ethnicity(text),
            "a1c": extract_a1c(text),
        })
    return pd.DataFrame(records)


def plot_category_distribution(df: pd.DataFrame, title: str, filename: str):
    """Plot category distribution bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    counts = df["category"].value_counts()
    cats = [c for c in CATEGORY_ORDER if c in counts.index]
    vals = [counts[c] for c in cats]
    colors = [CATEGORY_COLORS[c] for c in cats]

    bars = ax.barh(cats, vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Number of Notes", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add count labels
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{val}", va="center", fontsize=10)

    ax.invert_yaxis()
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved: {out_path}")


def plot_age_distribution(demo_df: pd.DataFrame, filename: str):
    """Plot age distribution histogram by category."""
    valid = demo_df.dropna(subset=["age"])
    if len(valid) == 0:
        print("    ⚠ No age data extracted, skipping age plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    active_cats = [c for c in CATEGORY_ORDER if c in valid["category"].unique()]
    sns.histplot(
        data=valid, x="age", hue="category",
        hue_order=active_cats,
        palette=CATEGORY_COLORS,
        bins=20, alpha=0.6, kde=True, ax=ax,
    )
    ax.set_xlabel("Patient Age (years)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Age Distribution by Discontinuation Category", fontsize=14, fontweight="bold")
    if ax.get_legend():
        ax.legend(title="Category", fontsize=8, title_fontsize=9,
                  bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved: {out_path}")


def plot_sex_distribution(demo_df: pd.DataFrame, filename: str):
    """Plot sex distribution overall and by category."""
    valid = demo_df.dropna(subset=["sex"])
    if len(valid) == 0:
        print("    ⚠ No sex data extracted, skipping sex plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall sex distribution
    sex_counts = valid["sex"].value_counts()
    axes[0].pie(sex_counts, labels=sex_counts.index, autopct="%1.1f%%",
                colors=["#56B4E9", "#E69F00"], startangle=90)
    axes[0].set_title("Overall Sex Distribution", fontsize=12, fontweight="bold")

    # Sex by category
    cross = pd.crosstab(valid["category"], valid["sex"], normalize="index") * 100
    cross = cross.reindex([c for c in CATEGORY_ORDER if c in cross.index])
    cross.plot(kind="barh", stacked=True, ax=axes[1],
               color=["#56B4E9", "#E69F00"], edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("Percentage (%)", fontsize=11)
    axes[1].set_title("Sex Distribution by Category", fontsize=12, fontweight="bold")
    axes[1].legend(title="Sex", fontsize=9)

    plt.suptitle("Patient Sex Distribution Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved: {out_path}")


def plot_bmi_distribution(demo_df: pd.DataFrame, filename: str):
    """Plot BMI distribution."""
    valid = demo_df.dropna(subset=["bmi"])
    if len(valid) == 0:
        print("    ⚠ No BMI data extracted, skipping BMI plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    active_cats = [c for c in CATEGORY_ORDER if c in valid["category"].unique()]
    sns.boxplot(
        data=valid, x="category", y="bmi",
        hue="category",
        order=active_cats,
        hue_order=active_cats,
        palette=CATEGORY_COLORS,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Discontinuation Category", fontsize=11)
    ax.set_ylabel("BMI", fontsize=11)
    ax.set_title("BMI Distribution by Discontinuation Category", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved: {out_path}")


def plot_a1c_distribution(demo_df: pd.DataFrame, filename: str):
    """Plot HbA1c distribution."""
    valid = demo_df.dropna(subset=["a1c"])
    if len(valid) == 0:
        print("    ⚠ No A1c data extracted, skipping A1c plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    active_cats = [c for c in CATEGORY_ORDER if c in valid["category"].unique()]
    sns.violinplot(
        data=valid, x="category", y="a1c",
        hue="category",
        order=active_cats,
        hue_order=active_cats,
        palette=CATEGORY_COLORS,
        inner="box", legend=False, ax=ax,
    )
    ax.set_xlabel("Discontinuation Category", fontsize=11)
    ax.set_ylabel("HbA1c (%)", fontsize=11)
    ax.set_title("HbA1c Distribution by Discontinuation Category", fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Saved: {out_path}")


def main():
    print("=" * 60)
    print("  Step 4: Clinical & Demographic Coverage")
    print("=" * 60)

    df = load_all()
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    results = {}

    # ── 4.1 Category distribution ────────────────────────────────────────
    print("\n  4.1 Category Distribution:")
    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        counts = split_df["category"].value_counts().to_dict()
        results[f"{split_name}_category_distribution"] = counts

        print(f"\n    {split_name.upper()} ({len(split_df)} notes):")
        for cat in CATEGORY_ORDER:
            n = counts.get(cat, 0)
            pct = n / len(split_df) * 100
            print(f"      {cat:<22} {n:>5} ({pct:>5.1f}%)")

        plot_category_distribution(
            split_df,
            f"Category Distribution ({split_name.title()} Set)",
            f"category_dist_{split_name}.png",
        )

    # ── 4.2 Demographic extraction ───────────────────────────────────────
    print("\n  4.2 Demographic Extraction:")
    demo_all = extract_demographics(df)

    # Summary stats
    for field in ["age", "sex", "bmi", "race", "a1c"]:
        n_found = demo_all[field].notna().sum()
        pct = n_found / len(demo_all) * 100
        print(f"    {field:<8}: extracted from {n_found:>5}/{len(demo_all)} notes ({pct:.1f}%)")

    # Detailed stats
    demo_stats = {}

    # Age
    valid_age = demo_all["age"].dropna()
    if len(valid_age) > 0:
        demo_stats["age"] = {
            "n_extracted": int(len(valid_age)),
            "mean": round(float(valid_age.mean()), 1),
            "std": round(float(valid_age.std()), 1),
            "median": round(float(valid_age.median()), 1),
            "min": int(valid_age.min()),
            "max": int(valid_age.max()),
        }

    # Sex
    valid_sex = demo_all["sex"].dropna()
    if len(valid_sex) > 0:
        sex_dist = valid_sex.value_counts().to_dict()
        sex_pct = {k: round(v / len(valid_sex) * 100, 1) for k, v in sex_dist.items()}
        demo_stats["sex"] = {"n_extracted": int(len(valid_sex)), "distribution": sex_dist, "percentage": sex_pct}

    # BMI
    valid_bmi = demo_all["bmi"].dropna()
    if len(valid_bmi) > 0:
        demo_stats["bmi"] = {
            "n_extracted": int(len(valid_bmi)),
            "mean": round(float(valid_bmi.mean()), 1),
            "std": round(float(valid_bmi.std()), 1),
            "median": round(float(valid_bmi.median()), 1),
            "min": round(float(valid_bmi.min()), 1),
            "max": round(float(valid_bmi.max()), 1),
        }

    # Race
    valid_race = demo_all["race"].dropna()
    if len(valid_race) > 0:
        race_dist = valid_race.value_counts().to_dict()
        race_pct = {k: round(v / len(valid_race) * 100, 1) for k, v in race_dist.items()}
        demo_stats["race"] = {"n_extracted": int(len(valid_race)), "distribution": race_dist, "percentage": race_pct}

    # A1c
    valid_a1c = demo_all["a1c"].dropna()
    if len(valid_a1c) > 0:
        demo_stats["a1c"] = {
            "n_extracted": int(len(valid_a1c)),
            "mean": round(float(valid_a1c.mean()), 1),
            "std": round(float(valid_a1c.std()), 1),
            "median": round(float(valid_a1c.median()), 1),
            "min": round(float(valid_a1c.min()), 1),
            "max": round(float(valid_a1c.max()), 1),
        }

    results["demographics"] = demo_stats

    # ── 4.3 Sex × Category cross-tabulation ──────────────────────────────
    valid_sex_cat = demo_all.dropna(subset=["sex"])
    if len(valid_sex_cat) > 0:
        cross = pd.crosstab(valid_sex_cat["category"], valid_sex_cat["sex"])
        results["sex_by_category"] = cross.to_dict()
        print("\n  4.3 Sex × Category cross-tab:")
        print(cross.to_string())

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n  Generating demographic plots...")
    plot_age_distribution(demo_all, "demographics_age.png")
    plot_sex_distribution(demo_all, "demographics_sex.png")
    plot_bmi_distribution(demo_all, "demographics_bmi.png")
    plot_a1c_distribution(demo_all, "demographics_a1c.png")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "step4_demographic_coverage.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {out_path}")

    # Save full demographics CSV
    demo_csv = os.path.join(RESULTS_DIR, "extracted_demographics.csv")
    demo_all.to_csv(demo_csv, index=False)
    print(f"✓ Demographics CSV saved to {demo_csv}")


if __name__ == "__main__":
    main()
