📘 Fine-Tuning Strategy:

We fine-tuned BERT (prajjwal1/bert-tiny) and DeBERTa-v3 using a two-stage training procedure:

Stage 1 – Classifier training:

All encoder layers frozen.

Only the top classification layer trained.

Learning rate: 5e-5

Dropout: 0.2

Stage 2 – Full fine-tuning:

All encoder layers unfrozen.

Entire model trained jointly.

Learning rate: 2e-5

Dropout: 0.2 (optimized on split 1).

📂 Data Splits

The dataset was divided into three train/validation/test splits.

Hyperparameters (learning rate, dropout, number of epochs) were optimized only on split 1.

The same hyperparameters were then applied to splits 2 and 3 to ensure fair comparison, reproducibility, and unbiased evaluation.

Final results are reported as the mean across the three splits.

⚖️ Hyperparameters: We followed established NLP/ML best practices:

Using fixed hyperparameters across splits (instead of re-optimizing per split).

This avoids information leakage and inflated performance.

Ensures comparability between models (BERT vs DeBERTa-v3).

Aligned with recommendations from:
