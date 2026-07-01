import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


MODEL_DIR = os.path.join(os.path.dirname(__file__), "output")
DATASET_NAME = "google/code_x_glue_cc_defect_detection"
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(os.path.dirname(__file__), "confusion_matrix.png")


def main():
    dataset = load_dataset(DATASET_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    test_texts = dataset["test"]["func"]
    true_labels = dataset["test"]["target"]
    predictions = classifier(test_texts, truncation=True, max_length=256)
    pred_labels = [int(item["label"].split("LABEL_")[-1]) for item in predictions]

    precision = precision_score(true_labels, pred_labels, average="binary")
    recall = recall_score(true_labels, pred_labels, average="binary")
    f1 = f1_score(true_labels, pred_labels, average="binary")
    roc_auc = roc_auc_score(true_labels, [item["score"] for item in predictions])

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["No Defect", "Defect"])
    plt.yticks(tick_marks, ["No Defect", "Defect"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    metrics = {
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "roc_auc": round(float(roc_auc), 6),
        "confusion_matrix": cm.tolist(),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Evaluation summary")
    print(f"- Precision: {metrics['precision']}")
    print(f"- Recall: {metrics['recall']}")
    print(f"- F1: {metrics['f1']}")
    print(f"- AUC-ROC: {metrics['roc_auc']}")
    print(f"- Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
    print(f"- Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
