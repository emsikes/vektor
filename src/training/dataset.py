import json
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer


def load_split(split_name: str, splits_str: str = "data/splits") -> list[dict]:
    path = Path(splits_str) / f"{split_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)     # Return a list of dicts matching unified normalize_example schema defined in src/data/loaders.py
    
def build_tokenizer(model_name: str = "answerdotai/ModernBERT-large"):
    # Instantiate once and pass around so we don't re-download on every call
    return AutoTokenizer.from_pretrained(model_name)

def tokenize_split(examples: list[dict], tokenizer, max_length: int = 512) -> Dataset:
    """
    Extract texts and labels into separate lists for the tokenizer which expects a flat 
    list of strings, not a list dicts.
    """
    texts = [ex["text"] for ex in examples]
    labels = [ex["label"] for ex in examples]

    tokenized = tokenizer(
        texts,
        truncation=True,            # Truncate to max_length to avoid OOM for long inputs
        padding="max_length",       # Pad all sequences to same length for consistent batch shapes
        max_length=max_length,      # 512 for baseline, will move to 2048 in a later phase
        return_tensors=None         # Return a plain Pythin list so Dataset.from_dict can consume them
    )

    tokenized["labels"] = labels    # Add labels after tokenization as a plain list
    return Dataset.from_dict(tokenized)
