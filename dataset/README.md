# REV4RE Dataset

## Overview

REV4RE (Reviews for Requirements Engineering) is a gold-standard dataset of manually annotated app reviews for requirements classification. Reviews were collected from the Apple App Store and Google Play Store between February–May 2022.

## File Format

The dataset is provided in CSV format with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `ID` | int | Unique review identifier |
| `Date` | string | Review posting date |
| `AppName` | string | Name of the source application |
| `Review` | string | Raw review text |
| `size` | string | Review size (if available) |
| `Rating` | int | Star rating (1–5) |
| `RivewRelevance` | string | Classification label: `explicit`, `implicit`, or `irrelevant` |
| `Annonater` | string | Annotator code |
| `category` | string | App Store category |
| `Tokens` | int | Number of tokens in the review |
| `Sentences` | int | Number of sentences in the review |
| `Cleaned_Review` | string | Preprocessed review text (lowercased, cleaned) |
| `Frequency` | string | Token frequency distribution |
| `Reviewer_Sentiment` | string | Reviewer sentiment (Pos/Neg) |
| `Transformer_HF_Sentiment` | string | HuggingFace transformer sentiment scores |
| `TextBlob_Sentiment` | string | TextBlob sentiment analysis output |
| `Vader_Sentiment` | string | VADER sentiment analysis output |

## Label Definitions

| Label | Class | Description |
|-------|-------|-------------|
| `explicit` | Explicit Requirements | Reviews that directly state a requirement — feature requests, bug reports, or help requests. Characteristics: ER1–ER4. |
| `implicit` | Implicit Requirements | Reviews where requirements are implied through narratives, comparisons, or complaints without direct statement. Characteristics: MR1–MR6. |
| `irrelevant` | Irrelevant Reviews | Reviews focused on services, pricing, community, or content unrelated to app functionality. Characteristics: IR1–IR4. |

## Usage

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('rev4re.csv')

# Class distribution
print(df['RivewRelevance'].value_counts())

# Filter by category
shopping = df[df['category'] == 'Shopping']

# Map labels to numeric for model training
label_map = {'irrelevant': 0, 'explicit': 1, 'implicit': 2}
df['label'] = df['RivewRelevance'].map(label_map)

# Split for training/testing (80/20, seed=42)
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)
```

## Ethical Considerations

- Reviews were publicly available at the time of collection
- Personally identifiable information has been removed where identified
- Reviews with language challenges (grammar errors, colloquialisms) were deliberately retained for authenticity

## Citation

Please cite the accompanying paper if you use this dataset.
