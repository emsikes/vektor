import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred

    # Take the argmax across class dimensions to get predicted class indices
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions,
        average="binary",       # Binary task for now, will convert to "macro" in later phase
        pos_label=1             # Injection is the positive class
    )

    accuracy = accuracy_score(labels, predictions)

    """ 
    What we attempt to measure here:

    A false negative means the model saw an injection attack as a "clean" request.

    'predictions == 1' - created a boolean array, True where the model predicted "clean"
    'labels == 1' - creates a boolean array, True where the true label is "injection"
    '&' - True only where BOTH are True, meaning predicted clean but actually an injection
    'np.sum()..' - counts up the True values - this acount becomes our false negative count

    This is the primary and crucial metric: missing an attack is worse than a false alarm
    Target for false predictions is <=2%
    """

    fn = np.sum((predictions == 0) & (labels == 1))
    fnr = fn / np.sum(labels == 1)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_negative_rate": round(fnr, 4)
    }

def check_targets(metrics: dict, fnr_target: float = 0.02, recall_target: float = 0.98, f1_target: float = 0.95) -> None:
    # Report whether each key metric met the target for this phase
    print("\n--- Target Check ---")
    checks = [
        """ 
        fnr and recall mirror one another but fnr is more intuitive in a security context: fnr = 1 - recall
        f1 prevents high recall via over-flagging and forces precision to stay high
        """
        ("false_negative_rate", metrics["false_negaitve_rate"], fnr_target, "<="),
        ("recall",              metrics["recall"],              recall_target, ">="),   
        ("f1",                  metrics["f1"],                  f1_target, ">="),       
    ]
    for name, value, target, direction in checks:
        if direction == "<=":
            passed = value <= target
        else:
            passed = value >= target
        status = "PASS" if passed else "FAIL"
        print(f" {status} {name}; {value:.4f} (target{direction} {target})")
    print("-----------------------------------------\n")