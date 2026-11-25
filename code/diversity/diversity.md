# Diversity Implementation Pipeline

**Date**: 2026-05-18  
**Scope**: Step-by-step guide for implementing the Diversity evaluation metrics (Section 2.4) for the synthetic GLP-1 Discontinuation EHR Notes dataset.

---

## Overview

Diversity analysis ensures your LLM did not suffer from "mode collapse" (generating highly repetitive, superficially varied notes that cluster tightly in semantic space) and that the generated patient populations align with expected real-world distributions. 

This pipeline covers four core areas:
1. **Lexical Diversity**: Vocabulary breadth (TTR, N-gram ratio).
2. **Semantic Diversity**: Deep linguistic variation (Pairwise BERTScore).
3. **Visual Semantic Coverage**: Embedding projections (UMAP).
4. **Clinical & Demographic Coverage**: Distribution checks.

---

## Step 1: Lexical Diversity (TTR and Unique N-gram Ratio)

Lexical metrics evaluate the breadth of vocabulary and phrasing at the surface level.

### 1.1 Preprocessing
*   **Action**: Clean the text by converting to lowercase, removing punctuation, and stripping non-alphanumeric characters.
*   **Tool**: Standard Python string operations or `nltk`/`spaCy`.

### 1.2 Type-Token Ratio (TTR)
*   **Action**: Calculate the ratio of unique words (types) to the total number of words (tokens).
*   **Formula**: `TTR = Number of Unique Words / Total Number of Words`
*   **Process**:
    1. Tokenize the corpus (all synthetic notes combined).
    2. Count the total tokens.
    3. Count the unique tokens (using a Python `set`).
    4. Compute the ratio.
*   **Note**: Because TTR is highly sensitive to text length, compare it against a sample of MIMIC notes of the **exact same total word count**.

### 1.3 Unique N-gram Ratio
*   **Action**: Calculate the diversity of short phrases (bi-grams and tri-grams).
*   **Process**:
    1. Extract all 2-grams and 3-grams using `nltk.ngrams`.
    2. Calculate: `Unique N-grams / Total N-grams`.
    3. Compare this ratio against your MIMIC baseline. High unique n-gram ratios indicate the LLM is not recycling the same templated phrasing.

---

## Step 2: Semantic Diversity (Pairwise BERTScore)

While lexical diversity checks surface words, semantic diversity ensures the underlying *meanings* and sentence structures are varied.

### 2.1 Setup
*   **Model**: Use a clinical encoder (e.g., `emilyalsentzer/Bio_ClinicalBERT` or `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`).
*   **Tool**: `bert-score` Python package.

### 2.2 Execution
*   **Action**: Compute the pairwise similarity between notes within the same category.
*   **Process**:
    1. Sample a subset of notes from the same discontinuation category (e.g., 100 notes from "Adverse Effects").
    2. Generate all unique pairs from this subset.
    3. Compute the BERTScore (specifically the F1 or Recall score) for each pair.
    4. Calculate the average pairwise BERTScore for the category.
*   **Interpretation**: 
    *   **Low Score**: High diversity (good). The notes mean different things or express them differently.
    *   **High Score (> 0.90)**: Mode collapse (bad). The LLM is generating the exact same semantic content repeatedly.

---

## Step 3: Visual Semantic Coverage (UMAP of Embeddings)

This step provides a compelling visual aid for your paper, demonstrating that notes from different discontinuation categories occupy distinct semantic spaces, yet show natural variation.

### 3.1 Generate Document Embeddings
*   **Action**: Convert each full note into a single dense vector.
*   **Process**:
    1. Pass each synthetic note through ClinicalBERT.
    2. Use mean-pooling (averaging the token embeddings) or the `[CLS]` token embedding to represent the entire document.

### 3.2 Dimensionality Reduction (UMAP)
*   **Action**: Project the high-dimensional embeddings (e.g., 768 dimensions) down to 2D for visualization.
*   **Tool**: `umap-learn` library.
*   **Parameters**: Tune `n_neighbors` (usually 15-50) and `min_dist` (0.1) to get clear clusters.

### 3.3 Visualization
*   **Action**: Plot the 2D points.
*   **Tool**: `matplotlib` and `seaborn`.
*   **Process**:
    1. Create a scatter plot.
    2. Color code the points by their **GLP-1 Discontinuation Category** (the 7 classes).
    3. **Expected Result**: You should see 7 relatively distinct clusters (showing the categories are semantically distinct) but with wide spreads (showing intra-category diversity), not tight single dots.

---

## Step 4: Clinical and Demographic Coverage

Reviewers want to ensure the generated patients reflect a realistic population, not just a skewed demographic (e.g., 90% white males).

### 4.1 Extract Demographics and Conditions
*   **Action**: If your generation prompt outputs structured metadata (Age, Sex, Race, BMI, Comorbidities) alongside the note, load this JSON/CSV data. If not, use an NER tool (like `scispaCy` or `stanza`) to extract these from the text.

### 4.2 Compare Against a Reference Baseline
*   **Action**: Plot the distributions of your synthetic data against a known epidemiological baseline (e.g., general MIMIC-IV outpatient demographics, or CDC data on GLP-1 users).
*   **Process**:
    1. Plot a histogram of Age distributions.
    2. Plot bar charts for Sex and Race.
    3. **Interpretation**: The distributions don't have to perfectly match reality, but they should be *plausible* and importantly, not severely skewed by LLM bias (e.g., ensuring female patients aren't exclusively associated with "Patient Preference" discontinuations while male patients get "Adverse Effects").

### 4.3 Category Balance Check
*   **Action**: Report the final distribution of the 7 discontinuation categories. Since you generated this data using a schema, this should be perfectly balanced (e.g., 1000 notes per category), which you should explicitly state as a methodological choice for classifier training.

---

## Recommended Code Structure

To implement this, you should create a script (e.g., `evaluate_diversity.py`) with the following logic:

```python
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.util import ngrams
from bert_score import score
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
import torch

# 1. Load Data
# df = pd.read_csv('synthetic_notes.csv')

# 2. Lexical Diversity
def calculate_ttr(texts):
    # Tokenize and calculate type-token ratio
    pass

def calculate_ngram_ratio(texts, n=2):
    # Calculate unique n-gram ratio
    pass

# 3. Semantic Diversity
def calculate_pairwise_bertscore(texts):
    # Use bert_score to compute pairwise similarities
    pass

# 4. UMAP Visualization
def plot_umap(texts, labels):
    # Generate embeddings using ClinicalBERT
    # Fit UMAP and plot with seaborn
    pass

# 5. Demographic distributions
def plot_demographics(df):
    # Plot histograms/bar charts
    pass
```
