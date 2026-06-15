# Pre-Trained Models

## Base Models (from HuggingFace)

The following BERT-based models were used for fine-tuning:

| Model | HuggingFace ID | Parameters |
|-------|----------------|------------|
| BERT-base | [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) | 110M |
| DistilBERT | [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) | 60M |
| ALBERT | [`albert-base-v2`](https://huggingface.co/albert-base-v2) | 11M |
| RoBERTa | [`roberta-base`](https://huggingface.co/roberta-base) | 125M |

## Fine-Tuned Models

Fine-tuned model checkpoints will be made available upon paper acceptance.

<!-- 
Uncomment and update links when models are uploaded:

| Model | Link | Best Configuration |
|-------|------|--------------------|
| BERT-REV4RE | [Download](#) | Under-sampled |
| DistilBERT-REV4RE | [Download](#) | Under-sampled |
| ALBERT-REV4RE | [Download](#) | Under-sampled |
| RoBERTa-REV4RE | [Download](#) | Under-sampled |
-->

## Training Configuration

All models were fine-tuned with:
- Learning rate: 5e-5
- Batch size: 32
- Max epochs: 2
- Optimizer: Adam
- Early stopping patience: 2
- Max sequence length: 512 tokens
- Random seed: 42
