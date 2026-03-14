import yaml
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from src.training.dataset import load_split, build_tokenizer, tokenize_split
from src.training.metrics import compute_metrics


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    with open(Path(config_path), "r") as f:
        # Load all hyperparameters from yaml dynamically - no hardcoded values
        return yaml.safe_load(f)
    
def load_model(model_name: str):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,                       # Binary classification - clean (0) vs injection (1)
        ignore_mismatched_sizes=True       # Classification head randomly initialized - expected mismath
    )

def build_training_args(config: dict) -> TrainingArguments:
    # Map each key from configs/training_config.yaml

    cfg = config["training"]
    return TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_steps"],
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        eval_strategy=cfg["eval_strategy"],
        save_strategy=cfg["save_strategy"],
        load_best_model_at_end=cfg["load_best_model_at_end"],
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=cfg["greater_is_better"],
        logging_steps=cfg["logging_steps"],
        report_to=cfg["report_to"],
        # always save the best checkpoint by recall, not just the most recent
        save_total_limit=2,
        run_name="vektor-guard-v1-phase2"
    )

def build_trainer(config_path: str = "configs/training_config.yaml") -> Trainer:
    config = load_config(config_path)
    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]

    # build tokenizer and model from config
    tokenizer = build_tokenizer(model_name)
    model = load_model(model_name)

    # load and tokenize all three splits
    train_dataset = tokenize_split(load_split("train"), tokenizer, max_length)
    val_dataset = tokenize_split(load_split("val"), tokenizer, max_length)

    training_args = build_training_args(config)

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,        # val set used for mid-training evaluation
        processing_class=tokenizer,      # v5 replacement for deprecated tokenizer= arg — enables DataCollatorWithPadding
        compute_metrics=compute_metrics  # our custom metrics including FNR
    )
