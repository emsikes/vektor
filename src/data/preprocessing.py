from sklearn.model_selection import train_test_split
import json
from pathlib import Path


def deduplicate(examples: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for ex in examples:
        key = ex["text"].lower().strip()    # normalize before hashing
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique

def class_balance_report(examples: list[dict]) -> None:
    from collections import Counter
    total = len(examples)
    labels = Counter(ex["label"] for ex in examples)

    print(f"\n Overall {total} examples:")
    print(f"  Clean (0):     {labels[0]} ({labels[0]/total*100:.1f}%)")
    print(f"  Injection (1): {labels[1]} ({labels[1]/total*100:.1f}%)")

    print("\nPer source:")
    sources = Counter(ex["source"] for ex in examples)
    for source in sources:
        subset = [ex for ex in examples if ex["source"] == source]
        src_labels = Counter(ex["label"] for ex in subset)
        print(f"  {source} ({len(subset)} examples): "
              f"clean={src_labels[0]} ({src_labels[0]/len(subset)*100:.1f}%) "
              f"injection={src_labels[1]} ({src_labels[1]/len(subset)*100:.1f}%)")
        
def split_dataset(examples: list[dict], val_size: float = 0.1, test_size: float = 0.1, seed: int = 42) -> tuple:
    labels = [ex["label"] for ex in examples]

    # Carve out test, then the remainder train/val
    train_val, test = train_test_split(examples, test_size=test_size, stratify=labels, random_state=seed)

    tv_labels = [ex["label"] for ex in train_val]

    # Adjust val_size relative to the remaining data once test is removed
    train, val = train_test_split(train_val, test_size=val_size / (1 - test_size), stratify=tv_labels, random_state=seed)

    return train, val, test

def save_splits(train: list[dict], val: list[dict], test: list[dict], output_dir: str = "data/splits") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = out / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2, ensure_ascii=False)   # Perserve non-ascii characters in prompts
        print(f"Saved {name}: {len(split)} examples in {path}")