"""
data_loader.py – Loads JSONL data and provides helper functions for text preprocessing.
"""
import json
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (only once)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from config import TRAIN_JSONL, TEST_JSONL, CATEGORY_SHORT


def load_jsonl(path: str) -> pd.DataFrame:
    """Load a JSONL file into a pandas DataFrame."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    # Add short category label
    df["category"] = df["primary_reason"].map(CATEGORY_SHORT)
    return df


def load_train() -> pd.DataFrame:
    return load_jsonl(TRAIN_JSONL)


def load_test() -> pd.DataFrame:
    return load_jsonl(TEST_JSONL)


def load_all() -> pd.DataFrame:
    """Load both train and test with a 'split' column."""
    train = load_train()
    train["split"] = "train"
    test = load_test()
    test["split"] = "test"
    return pd.concat([train, test], ignore_index=True)


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation and non-alphanumeric chars."""
    text = text.lower()
    # Handle escaped newlines from JSONL
    text = text.replace("\\n", " ")
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Tokenize cleaned text into words."""
    return word_tokenize(clean_text(text))


if __name__ == "__main__":
    df = load_all()
    print(f"Loaded {len(df)} notes total")
    print(f"  Train: {(df['split']=='train').sum()}")
    print(f"  Test:  {(df['split']=='test').sum()}")
    print(f"\nCategories:\n{df['category'].value_counts().to_string()}")
