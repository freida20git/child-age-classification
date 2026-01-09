# Child Age Classification

## Short summary
This repository contains a data-analysis + NLP / LLM project to predict child age (and age ranges) from conversational data. The work combines engineered speaking/linguistic features and classical ML models, transformer-based fine-tuning, and LLM-driven data refinement (prompt-based corrections). Notebooks and scripts include feature extraction, data splits, model training, evaluation, and result artifacts (including a PDF with BERT results).

This README reports averaged metrics across the three experimental splits. Per-split numbers and full tables are available in the notebooks and the PDF artifact in the repo.

---

## Quick links (important files)
- Notebooks:
  - Speak_Age_Features.ipynb — feature extraction and preprocessing
  - Speak_Age_Promts.ipynb — LLM prompts & data correction (typo: "Promts")
  - Speak_Age_Train_and_Test_All_Models.ipynb — classical ML experiments and analyses
  - splits.ipynb — split generation and protocol
  - weighted_splits_inference_training_model.ipynb — additional split/model experiments
  - TinyBERT classifier (notebook referenced)
  - Regression_BERT (notebook referenced)
  - Embeddings and LLM (notebook referenced; uses Gemini embeddings)
- Scripts / models:
  - models training and inference codes/DeBERTav3_focalLoss_llmData.py — DeBERTa-v3 fine-tuning with focal loss and LLM-refined data
- Reports:
  - models training and inference codes/results of runing bert on raw data 3 splits.pdf — PDF with BERT per-split results (summarized below)
- Data (external):
  - Dataset and embeddings are referenced on Google Drive in the notebooks (not stored in this public repo). Splits link and division code are referenced in notebooks/PDF.

---

## What was evaluated
Tasks:
- Classification into age ranges: 2–3, 3–5, 5–6 (three-class classification).
- Regression to predict continuous child_age (months).
- Embedding-based regression using Gemini LLM embeddings + classical regressors.

Method highlights:
- Transformer fine-tuning (Tiny-BERT and DeBERTa‑v3) using a two-stage approach: (1) freeze encoder and train classifier head, (2) unfreeze and fine-tune full model.
- DeBERTa experiments use a custom focal loss implementation (see DeBERTav3_focalLoss_llmData.py).
- Classical ML: Logistic Regression, Random Forest, HistGradientBoostingRegressor, CatBoost, LightGBM, etc.
- LLM usage: prompt-based label/data refinement and use of LLM embeddings (Gemini API) for downstream regression.

---

## Results (averages across the three splits)
All classification accuracy/F1 are reported as weighted metrics unless otherwise noted. Regression metrics: MAE and RMSE are denormalized to months; R² is reported as-is.

1) Tiny‑BERT — classification
- Accuracy (avg): 0.69
- F1 (avg): 0.69

2) DeBERTa‑v3 — classification
- Accuracy (avg): 0.71
- F1 (avg): 0.70

3) Tiny‑BERT — regression (regressor head)
- RMSE (months, avg): 8.91
- MAE  (months, avg): 7.09
- R² (avg): 0.58

4) Gemini embeddings + classical regressors (avg across splits)
- Random Forest ≈ [MAE 8.17, RMSE 9.68, R² 0.512]
- LightGBM     ≈ [MAE 6.95, RMSE 8.71, R² 0.606]
- CatBoost     ≈ [MAE 6.70, RMSE 8.32, R² 0.637]
- CatBoost+PCA ≈ [MAE 6.55, RMSE 8.24, R² 0.647]

Notes: MAE and RMSE are in months for regression tasks.

---

## Per-split behaviour & qualitative observations
- Classification: 2–3 age group often shows higher recall; 3–5 is commonly confused with adjacent groups.
- Regression: R² values typically range ~0.50–0.65; embedding-based models give the strongest R² (~0.64–0.65).
- Feature importance: consistent top features include kc, narrative_complexity, mlum/MLU, child_participation, child_word_ratio, mtld/mattr, ipsyn metrics.

---

## Per-notebook one-paragraph summaries
- Speak_Age_Features.ipynb: Extracts and computes engineered conversation-level features (lexical diversity, MLU, clause density, narrative complexity, phonological features) and prepares a features CSV used by experiments.
- Speak_Age_Promts.ipynb: Contains prompt templates and procedures used with an LLM to correct/refine labels and text; used to produce the LLM-refined dataset (llmData) referenced by transformer scripts.
- Speak_Age_Train_and_Test_All_Models.ipynb: Main analysis notebook that runs multiple classical ML models (logistic regression, linear regression, random forest, HGB, etc.), reports classification and regression metrics, shows confusion matrices, and prints top feature importances.
- splits.ipynb: Script/notebook to create and document the three train/validation/test splits used across experiments; referenced by other notebooks and the PDF.
- weighted_splits_inference_training_model.ipynb: Loads the LLM-labeled CSV (≈1831 rows) and runs split-aware experiments with weighted training/inference; includes exploratory outputs and experiment harness used for ablations.
- TinyBERT classifier (notebook): Implements Tiny-BERT fine-tuning for classification; prints per-split accuracy/F1 used for the Tiny-BERT rows in the results table.
- Regression_BERT (notebook): Implements BERT-based regression (regressor head) and reports RMSE, MAE, and R² across splits.
- Embeddings and LLM (notebook): Produces and saves Gemini embeddings (.pkl) and trains classical regressors (RandomForest, LightGBM, CatBoost, CatBoost+PCA) on top of embeddings; reports denormalized MAE/RMSE and R² in months.

---

## Files & code pointers
- Transformer fine-tuning:
  - models training and inference codes/DeBERTav3_focalLoss_llmData.py — DeBERTa-v3 with focal loss, two-stage training, plotting and saving of final metrics.
- Classical & regression:
  - Speak_Age_Train_and_Test_All_Models.ipynb — classical ML experiments and analyses.
  - weighted_splits_inference_training_model.ipynb — split-aware training/inference experiments.
- Embeddings experiments:
  - Embeddings and LLM (notebook) — uses Gemini embeddings saved to Drive.
- Results artifact:
  - models training and inference codes/results of runing bert on raw data 3 splits.pdf — source of per-split transformer numbers.

---

## Reproducibility notes & suggestions
- Data: Notebooks reference a Google Drive CSV and embeddings on Drive. To reproduce, ensure Drive access or include a small sample dataset in the repo.
- Environment: I added a candidate requirements.txt in the repo (see requirements.txt). It was generated from imports found across notebooks and scripts; please verify versions and adjust as needed.
- Organization: Consider moving notebooks to /notebooks, results to /reports or /artifacts, and scripts to /src (avoid spaces in directory names).

---

## Next steps completed per your request
- Added per-notebook one-paragraph summaries.
- Produced a candidate requirements.txt based on imports in notebooks and scripts.
- Prepared this cleaned README ready to commit (committed alongside requirements.txt).

---

## Contact
If you want any phrasing edits, a different presentation of numeric results (per-split tables vs. averages), or additional artifacts extracted into the repo (e.g., small sample data or CI), tell me and I will update and commit accordingly.
