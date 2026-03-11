from datasets import load_dataset


def normalize_example(text: str, label: int, source: str, attack_type: str = "unknown") -> dict:
    return {
        "text": text.strip(),        # remove leading/trailing whitespace from raw sources
        "label": int(label),         # enforce int — some sources use bool or string labels
        "source": source,            # track origin for per-source error analysis later
        "attack_type": attack_type   # "unknown" until Phase 3 multi-class labeling
    }

def load_deepset() -> list[dict]:
    raw = load_dataset("deepset/prompt-injections", split="train")
    examples = []
    for row in raw:
        examples.append(normalize_example(
            text=row["text"],
            label=row["label"],     # already int 0/1 — no mapping needed
            source="deepset"
        ))
    return examples

def load_jackhhao() -> list[dict]:
    LABEL_MAP = {"benign": 0, "jailbreak": 1}   # string labels -> int
    raw = load_dataset("jackhhao/jailbreak-classification", split="train")
    examples = []
    for row in raw:
        examples.append(normalize_example(
            text=row["prompt"],     # field is "prompt" not "text"
            label=LABEL_MAP[row["type"]],
            source="jackhhao",
            attack_type="jailbreak" if row["type"] == "jailbreak" else "unkown"
        ))
    return examples

def load_promptshield() -> list[dict]:
    raw = load_dataset("hendzh/PromptShield", split="train")    # Ignore val/test we will do our own splits
    examples = []
    for row in raw:
        examples.append(normalize_example(
            text=row["prompt"],     # field is "prompt" not "text"
            label=row["label"],     # already int 0/1
            source="promptshield"
        ))
    return examples

