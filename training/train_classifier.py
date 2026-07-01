import argparse
import os
from collections import Counter
from typing import Dict, Optional

import torch
from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


MODEL_NAME = "microsoft/codebert-base"
DATASET_NAME = "google/code_x_glue_cc_defect_detection"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
DEFAULT_MAX_LENGTH = 256


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss for imbalance handling."""

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT for defect detection")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_class_weights(labels):
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (len(label_counts) * label_counts.get(i, 1)) for i in range(2)],
        dtype=torch.float32,
    )
    return class_weights


def tokenize_function(tokenizer, max_length):
    def _tokenize(batch):
        return tokenizer(batch["func"], truncation=True, padding="max_length", max_length=max_length)

    return _tokenize


def inspect_class_balance(dataset):
    train_labels = dataset["train"]["target"]
    counts = Counter(train_labels)
    print("Class balance in training split:")
    for label, count in sorted(counts.items()):
        print(f"  label {label}: {count} examples")


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset = load_dataset(DATASET_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    inspect_class_balance(dataset)

    tokenized_dataset = dataset.map(tokenize_function(tokenizer, args.max_length), batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("target", "labels")
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]

    # We use class-weighted cross-entropy loss rather than a sampler because it works
    # naturally with the Trainer API and keeps the training loop simple in Colab/Kaggle.
    class_weights = compute_class_weights(list(train_dataset["labels"]))

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=args.logging_steps,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        class_weights=class_weights,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Training complete. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
