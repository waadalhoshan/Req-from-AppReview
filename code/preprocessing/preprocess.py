"""
REV4RE - Data Preprocessing
============================
Prepares app reviews for BERT-based fine-tuning.
Handles cleaning, splitting, and balancing strategies.

Dataset columns used:
    - Review: raw review text
    - Cleaned_Review: preprocessed review text
    - RivewRelevance: label (explicit, implicit, irrelevant)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import argparse
import os

# Label mapping: text labels to numeric
LABEL_MAP = {'irrelevant': 0, 'explicit': 1, 'implicit': 2}
LABEL_NAMES = ['irrelevant', 'explicit', 'implicit']


def load_dataset(filepath):
    """Load the REV4RE dataset from CSV."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} reviews")
    print(f"Class distribution:\n{df['RivewRelevance'].value_counts()}")
    return df


def prepare_for_training(df, use_cleaned=True):
    """
    Prepare dataset for Simple Transformers.
    Maps text labels to numeric and selects the text column.

    Args:
        df: DataFrame with REV4RE columns
        use_cleaned: if True, use Cleaned_Review; otherwise use Review
    """
    df = df.copy()

    # Select text column
    text_col = 'Cleaned_Review' if use_cleaned else 'Review'

    # Drop rows with missing text
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].str.strip().str.len() > 0]

    # Map labels to numeric
    df['label'] = df['RivewRelevance'].map(LABEL_MAP)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Rename text column for Simple Transformers compatibility
    df = df.rename(columns={text_col: 'text'})

    # Keep only required columns
    df = df[['text', 'label']].reset_index(drop=True)

    print(f"After preparation: {len(df)} reviews")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    return df


def split_dataset(df, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets (80/20).
    Uses random_state=42 for reproducibility.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")
    print(f"Train distribution:\n{train_df['label'].value_counts().sort_index()}")
    print(f"Test distribution:\n{test_df['label'].value_counts().sort_index()}")
    return train_df, test_df


def apply_class_weights(df):
    """
    Compute class weights inversely proportional to class frequency.
    Returns a list of weights ordered by label index [0, 1, 2].
    """
    counts = df['label'].value_counts().sort_index()
    total = len(df)
    n_classes = len(counts)
    weights = [total / (n_classes * counts[i]) for i in range(n_classes)]
    print(f"Class weights: {dict(zip(LABEL_NAMES, weights))}")
    return weights


def apply_undersampling(df, random_state=42):
    """
    Apply random under-sampling to balance classes.
    Reduces majority classes to match the size of the minority class.
    """
    min_count = df['label'].value_counts().min()
    balanced_dfs = []
    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        downsampled = resample(
            label_df,
            replace=False,
            n_samples=min_count,
            random_state=random_state
        )
        balanced_dfs.append(downsampled)
    balanced_df = pd.concat(balanced_dfs).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    print(f"\nAfter under-sampling: {len(balanced_df)} reviews")
    print(f"Distribution:\n{balanced_df['label'].value_counts().sort_index()}")
    return balanced_df


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess REV4RE dataset for BERT fine-tuning'
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to REV4RE CSV file')
    parser.add_argument('--output_dir', type=str, default='dataset/',
                        help='Output directory for processed files')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion (default: 0.2)')
    parser.add_argument('--balance', type=str, default='none',
                        choices=['none', 'undersample', 'class_weight'],
                        help='Balancing strategy for training set')
    parser.add_argument('--use_raw', action='store_true',
                        help='Use raw Review text instead of Cleaned_Review')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and prepare
    df = load_dataset(args.data)
    df = prepare_for_training(df, use_cleaned=not args.use_raw)

    # Split
    train_df, test_df = split_dataset(df, args.test_size, args.seed)

    # Apply balancing strategy to training set only
    if args.balance == 'undersample':
        train_df = apply_undersampling(train_df, random_state=args.seed)
    elif args.balance == 'class_weight':
        weights = apply_class_weights(train_df)
        pd.DataFrame([dict(zip(LABEL_NAMES, weights))]).to_csv(
            os.path.join(args.output_dir, 'class_weights.csv'), index=False
        )

    # Save processed files
    train_df.to_csv(os.path.join(args.output_dir, 'rev4re_train.csv'),
                    index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'rev4re_test.csv'),
                   index=False)
    print(f"\nFiles saved to {args.output_dir}")


if __name__ == '__main__':
    main()
