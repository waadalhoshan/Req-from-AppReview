"""
REV4RE - BERT-Based Model Fine-Tuning
=======================================
Fine-tunes BERT-family models for three-class requirements
classification (explicit, implicit, irrelevant).

Expects preprocessed CSV files with columns:
    - text: review text (from preprocess.py)
    - label: numeric label (0=irrelevant, 1=explicit, 2=implicit)

Uses the Simple Transformers library built on HuggingFace Transformers.
"""

import pandas as pd
import numpy as np
from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report
from sklearn.utils import resample
import argparse
import os
import warnings
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label mapping
LABEL_MAP = {'irrelevant': 0, 'explicit': 1, 'implicit': 2}
LABEL_NAMES = ['Irrelevant', 'Explicit', 'Implicit']

# Supported BERT-based models
MODEL_MAP = {
    'bert': ('bert', 'bert-base-uncased'),
    'distilbert': ('distilbert', 'distilbert-base-uncased'),
    'albert': ('albert', 'albert-base-v2'),
    'roberta': ('roberta', 'roberta-base'),
}


def load_data(train_path, test_path):
    """
    Load preprocessed training and test data.
    If the data has not been preprocessed (i.e., contains the original
    REV4RE columns), it will be converted automatically.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # If data comes directly from REV4RE (not preprocessed)
    if 'RivewRelevance' in train_df.columns:
        for df in [train_df, test_df]:
            df['label'] = df['RivewRelevance'].map(LABEL_MAP).astype(int)
            df['text'] = df['Cleaned_Review'].fillna(df['Review'])

    train_df = train_df[['text', 'label']].dropna()
    test_df = test_df[['text', 'label']].dropna()
    train_df['label'] = train_df['label'].astype(int)
    test_df['label'] = test_df['label'].astype(int)

    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    logger.info(f"Train distribution:\n{train_df['label'].value_counts().sort_index()}")
    return train_df, test_df


def get_class_weights(train_df):
    """Compute class weights inversely proportional to frequency."""
    counts = train_df['label'].value_counts().sort_index()
    total = len(train_df)
    n_classes = len(counts)
    weights = [total / (n_classes * counts[i]) for i in range(n_classes)]
    logger.info(f"Class weights: {dict(zip(LABEL_NAMES, weights))}")
    return weights


def apply_undersampling(df, random_state=42):
    """
    Apply random under-sampling to balance classes.
    Reduces majority classes to match minority class size.
    """
    min_count = df['label'].value_counts().min()
    balanced_dfs = []
    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        downsampled = resample(
            label_df, replace=False,
            n_samples=min_count, random_state=random_state
        )
        balanced_dfs.append(downsampled)
    balanced_df = pd.concat(balanced_dfs).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    logger.info(f"After under-sampling: {len(balanced_df)} reviews")
    logger.info(f"Distribution:\n{balanced_df['label'].value_counts().sort_index()}")
    return balanced_df


def create_model(model_name, num_labels=3, use_class_weights=False,
                 class_weights=None, args_dict=None):
    """Initialize a classification model with specified configuration."""
    if model_name not in MODEL_MAP:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Choose from: {list(MODEL_MAP.keys())}"
        )

    model_type, model_id = MODEL_MAP[model_name]

    default_args = {
        'num_train_epochs': 2,
        'learning_rate': 5e-5,
        'train_batch_size': 32,
        'eval_batch_size': 32,
        'max_seq_length': 512,
        'early_stopping_patience': 2,
        'use_early_stopping': True,
        'early_stopping_metric': 'eval_loss',
        'early_stopping_metric_minimize': True,
        'evaluate_during_training': True,
        'overwrite_output_dir': True,
        'save_model_every_epoch': False,
        'manual_seed': 42,
        'output_dir': f'outputs/{model_name}/',
        'best_model_dir': f'outputs/{model_name}/best_model/',
    }

    if args_dict:
        default_args.update(args_dict)

    if use_class_weights and class_weights:
        default_args['weight'] = class_weights

    model = ClassificationModel(
        model_type, model_id,
        num_labels=num_labels,
        args=default_args,
    )

    logger.info(f"Initialized {model_name} ({model_id})")
    return model


def train_and_evaluate(model, train_df, test_df, model_name):
    """Fine-tune the model and evaluate on the test set."""
    logger.info(f"Training {model_name}...")

    # Train
    model.train_model(train_df)

    # Predict
    predictions, raw_outputs = model.predict(test_df['text'].tolist())

    # Evaluate
    labels = test_df['label'].tolist()

    report = classification_report(
        labels, predictions,
        target_names=LABEL_NAMES,
        digits=4, output_dict=True
    )

    print(f"\n{'='*60}")
    print(f"Results: {model_name}")
    print(f"{'='*60}")
    print(classification_report(
        labels, predictions,
        target_names=LABEL_NAMES, digits=4
    ))

    results = {
        'model': model_name,
        'accuracy': report['accuracy'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
    }
    for i, name in enumerate(LABEL_NAMES):
        results[f'{name.lower()}_precision'] = report[name]['precision']
        results[f'{name.lower()}_recall'] = report[name]['recall']
        results[f'{name.lower()}_f1'] = report[name]['f1-score']

    return results, predictions, raw_outputs


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune BERT-based models for REV4RE classification'
    )
    parser.add_argument('--model', type=str, default='roberta',
                        choices=list(MODEL_MAP.keys()),
                        help='Model to fine-tune (default: roberta)')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training CSV')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test CSV')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs (default: 2)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate (default: 5e-5)')
    parser.add_argument('--balance_strategy', type=str, default='none',
                        choices=['none', 'class_weight', 'undersample'],
                        help='Class balancing strategy')
    parser.add_argument('--output_dir', type=str, default='outputs/',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    # Load data
    train_df, test_df = load_data(args.train_data, args.test_data)

    # Apply balancing strategy to training set
    if args.balance_strategy == 'undersample':
        train_df = apply_undersampling(train_df, random_state=args.seed)
        use_weights = False
        weights = None
    elif args.balance_strategy == 'class_weight':
        use_weights = True
        weights = get_class_weights(train_df)
    else:
        use_weights = False
        weights = None

    # Model configuration
    model_args = {
        'num_train_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'train_batch_size': args.batch_size,
        'manual_seed': args.seed,
        'output_dir': os.path.join(args.output_dir, args.model),
        'best_model_dir': os.path.join(args.output_dir, args.model,
                                       'best_model'),
    }

    # Create and train model
    model = create_model(
        args.model,
        use_class_weights=use_weights,
        class_weights=weights,
        args_dict=model_args
    )

    results, predictions, raw_outputs = train_and_evaluate(
        model, train_df, test_df, args.model
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_df = pd.DataFrame([results])
    results_path = os.path.join(args.output_dir, f'{args.model}_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")

    # Save predictions
    pred_df = test_df.copy()
    pred_df['prediction'] = predictions
    pred_path = os.path.join(args.output_dir, f'{args.model}_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved to {pred_path}")


if __name__ == '__main__':
    main()
