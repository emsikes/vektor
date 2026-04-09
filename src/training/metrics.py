import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


LABEL_NAMES = [
    "clean",
    "instruction_override",
    "indirect_injection",
    "jailbreak",
    "tool_call_hijacking"
]


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred

    # Take the argmax across class dimensions to get predicted class indices
    predictions = np.argmax(logits, axis=-1)

    # Macro average — computes F1 per class then averages equally across all classes
    # This prevents majority classes from dominating the score
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions,
        average="macro",
        zero_division=0
    )

    accuracy = accuracy_score(labels, predictions)

    # Per-class F1 — identifies which attack categories the model struggles with
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        labels, predictions,
        average=None,           # None returns per-class scores instead of averaged
        zero_division=0
    )

    """
    False negative rate for multi-class:
    A false negative means a real attack was classified as "clean" (label 0).

    'predictions == 0' - model predicted clean
    'labels != 0' - true label is any attack category
    '&' - True only where both are True: predicted clean but actually an attack
    'np.sum(labels != 0)' - total number of actual attack examples

    This remains the primary security metric — missing any attack type is worse than a false alarm.
    Target for Phase 3 is <=5% (relaxed from 2% — 5-class is harder than binary)
    """
    fn = np.sum((predictions == 0) & (labels != 0))
    total_attacks = np.sum(labels != 0)
    fnr = fn / total_attacks if total_attacks > 0 else 0.0

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "false_negative_rate": round(float(fnr), 4),
    }

    # Add per-class F1 scores for diagnostic visibility
    for i, label_name in enumerate(LABEL_NAMES):
        metrics[f"f1_{label_name}"] = round(float(per_class_f1[i]), 4)

    return metrics


def check_targets(
    metrics: dict,
    fnr_target: float = 0.05,
    macro_f1_target: float = 0.90,
    per_class_f1_minimum: float = 0.80
) -> None:
    # Report whether each key metric met the target for this phase
    print("\n--- Target Check ---")

    checks = [
        ("false_negative_rate", metrics["false_negative_rate"], fnr_target, "<="),
        ("macro_f1", metrics["macro_f1"], macro_f1_target, ">="),
    ]

    for name, value, target, direction in checks:
        passed = value <= target if direction == "<=" else value >= target
        status = "PASS" if passed else "FAIL"
        print(f" {status} {name}; {value:.4f} (target{direction} {target})")

    # Per-class F1 check — flag any category falling below minimum
    print("\n--- Per-Class F1 ---")
    for label_name in LABEL_NAMES:
        key = f"f1_{label_name}"
        value = metrics.get(key, 0.0)
        passed = value >= per_class_f1_minimum
        status = "PASS" if passed else "FAIL"
        print(f" {status} {label_name}; {value:.4f} (target>= {per_class_f1_minimum})")

    print("-----------------------------------------\n")