"""
REV4RE - Model Evaluation
==========================
Computes classification metrics (precision, recall, F1, AUC)
and generates ROC curves for fine-tuned BERT-based models.

Expects preprocessed CSV with columns:
    - text: review text
    - label: numeric label (0=irrelevant, 1=explicit, 2=implicit)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import label_binarize
from simpletransformers.classification import ClassificationModel
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json

# Label configuration
LABEL_MAP = {'irrelevant': 0, 'explicit': 1, 'implicit': 2}
LABEL_NAMES = ['Irrelevant', 'Explicit', 'Implicit']
N_CLASSES = 3


def load_model(model_path, model_type='roberta'):
    """Load a fine-tuned model from checkpoint."""
    model = ClassificationModel(
        model_type, model_path,
        num_labels=N_CLASSES,
        args={'silent': True}
    )
    return model


def load_test_data(test_path):
    """
    Load test data. Handles both preprocessed format (text, label)
    and original REV4RE format (Review, RivewRelevance).
    """
    df = pd.read_csv(test_path)

    if 'RivewRelevance' in df.columns:
        df['label'] = df['RivewRelevance'].map(LABEL_MAP).astype(int)
        df['text'] = df['Cleaned_Review'].fillna(df['Review'])

    df = df[['text', 'label']].dropna()
    df['label'] = df['label'].astype(int)
    return df


def predict(model, test_df):
    """Generate predictions and probability outputs."""
    texts = test_df['text'].tolist()
    predictions, raw_outputs = model.predict(texts)
    probabilities = softmax(raw_outputs, axis=1)
    return predictions, probabilities


def compute_metrics(y_true, y_pred):
    """Compute per-class and aggregate classification metrics."""
    report = classification_report(
        y_true, y_pred,
        target_names=LABEL_NAMES,
        digits=4, output_dict=True
    )

    metrics = {
        'accuracy': report['accuracy'],
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
        'weighted_f1': report['weighted avg']['f1-score'],
    }

    for i, name in enumerate(LABEL_NAMES):
        metrics[f'{name.lower()}_precision'] = report[name]['precision']
        metrics[f'{name.lower()}_recall'] = report[name]['recall']
        metrics[f'{name.lower()}_f1'] = report[name]['f1-score']
        metrics[f'{name.lower()}_support'] = int(report[name]['support'])

    return metrics


def compute_auc(y_true, probabilities):
    """
    Compute per-class and macro-averaged AUC using one-vs-rest.
    AUC_macro = (1/|C|) * sum(AUC_c) for c in C
    """
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    auc_scores = {}

    for i, name in enumerate(LABEL_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
        auc_scores[f'{name.lower()}_auc'] = float(auc(fpr, tpr))

    auc_scores['macro_auc'] = float(np.mean(
        [v for v in auc_scores.values()]
    ))
    return auc_scores


def plot_roc_curves(y_true, probabilities,
                    output_path='figures/roc_curves.png'):
    """Generate ROC curves for all three classes (one-vs-rest)."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)

        axes[i].plot(fpr, tpr, color=color, lw=2,
                     label=f'AUC = {roc_auc:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        axes[i].set_xlabel('False Positive Rate', fontsize=12)
        axes[i].set_ylabel('True Positive Rate', fontsize=12)
        axes[i].set_title(f'{name} (One-vs-Rest)', fontsize=14)
        axes[i].legend(loc='lower right', fontsize=11)
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {output_path}")


def plot_confusion_matrix(y_true, y_pred,
                          output_path='figures/confusion_matrix.png'):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate fine-tuned BERT models on REV4RE'
    )
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model directory')
    parser.add_argument('--model_type', type=str, default='roberta',
                        choices=['bert', 'distilbert', 'albert', 'roberta'],
                        help='Model architecture type')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test CSV')
    parser.add_argument('--output_dir', type=str, default='outputs/',
                        help='Directory for results and figures')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data and model
    test_df = load_test_data(args.test_data)
    model = load_model(args.model_path, args.model_type)
    y_true = test_df['label'].tolist()

    # Predict
    print("Generating predictions...")
    predictions, probabilities = predict(model, test_df)

    # Classification report
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_true, predictions,
                                target_names=LABEL_NAMES, digits=4))

    metrics = compute_metrics(y_true, predictions)

    # AUC
    auc_scores = compute_auc(y_true, probabilities)
    metrics.update(auc_scores)
    print(f"\nAUC Scores:")
    for k, v in auc_scores.items():
        print(f"  {k}: {v:.4f}")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Generate plots
    figures_dir = os.path.join(args.output_dir, 'figures')
    plot_roc_curves(y_true, probabilities,
                    os.path.join(figures_dir, 'roc_curves.png'))
    plot_confusion_matrix(y_true, predictions,
                          os.path.join(figures_dir, 'confusion_matrix.png'))


if __name__ == '__main__':
    main()
