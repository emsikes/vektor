from datasets import load_dataset


def normalize_example(text: str, label: int, source: str, attack_type: str = "unknown") -> dict:
    return {
        "text": text.strip(),
        "label": int(label),
        "source": source,
        "attack_type": attack_type
    }

def load_deepset() -> list[dict]:
    raw = load_dataset("deepset/prompt-injections", split="train")
    examples = []
    for row in raw:
        examples.append(normalize_example(
            text=row["text"],
            label=row["label"],
            source="deepset"
        ))
    return examples

def load_jackhhao() -> list[dict]:
    LABEL_MAP = {"benign": 0, "jailbreak": 1}
    raw = load_dataset("jackhhao/jailbreak-classification", split="train")
    examples = []
    for row in raw:
        examples.append(normalize_example(
            text=row["prompt"],
            label=LABEL_MAP[row["type"]],
            source="jackhhao",
            attack_type="jailbreak" if row["type"] == "jailbreak" else "unkown"
        ))
    return examples

def load_promptshield() -> list[dict]:
    raw = load_dataset("hendzh/PromptShield", split="train")
    examples = []
    for row in raw:
        examples.append(normalize_example(
            text=row["prompt"],
            label=row["label"],
            source="promptshield"
        ))
    return examples

