"""
Shared configuration for diversity evaluation pipeline.
All paths, model names, and category mappings live here.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)   # synthetic_dataset/

TRAIN_JSONL = os.path.join(DATA_DIR, "train.jsonl")
TEST_JSONL  = os.path.join(DATA_DIR, "test.jsonl")

RESULTS_DIR  = os.path.join(BASE_DIR, "results")
FIGURES_DIR  = os.path.join(RESULTS_DIR, "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────────
CLINICAL_MODEL = "thomas-sounack/BioClinical-ModernBERT-large"

# ── Category short labels for plotting ─────────────────────────────────────
CATEGORY_SHORT = {
    "ACCESS_COST_SYSTEM_BARRIERS":          "Access/Cost",
    "ADVERSE_EFFECTS_TOLERABILITY":         "Adverse Effects",
    "CLINICAL_DECISION_MEDICAL_APPROPRIATENESS": "Clinical Decision",
    "EFFICACY_EXPECTATION_MISMATCH":        "Efficacy Mismatch",
    "OTHER_UNKNOWN":                        "Other/Unknown",
    "PATIENT_PREFERENCE_BELIEFS":           "Patient Preference",
    "PLANNED_GOAL_BASED_DISCONTINUATION":   "Planned/Goal-Based",
}

# ── Category colors (colorblind-friendly palette) ──────────────────────────
CATEGORY_COLORS = {
    "Access/Cost":        "#E69F00",
    "Adverse Effects":    "#56B4E9",
    "Clinical Decision":  "#009E73",
    "Efficacy Mismatch":  "#F0E442",
    "Other/Unknown":      "#0072B2",
    "Patient Preference": "#D55E00",
    "Planned/Goal-Based": "#CC79A7",
}

# Canonical order for bar charts
CATEGORY_ORDER = [
    "Adverse Effects",
    "Clinical Decision",
    "Access/Cost",
    "Efficacy Mismatch",
    "Patient Preference",
    "Planned/Goal-Based",
    "Other/Unknown",
]

# ── Pairwise BERTScore sampling ────────────────────────────────────────────
BERTSCORE_SAMPLE_PER_CATEGORY = 50   # notes sampled per category for pairs
BERTSCORE_MAX_PAIRS = 200            # max pairs per category to keep runtime manageable

# ── UMAP parameters ───────────────────────────────────────────────────────
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST    = 0.1
UMAP_METRIC      = "cosine"
UMAP_RANDOM_STATE = 42
