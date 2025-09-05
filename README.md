# Fine-Tuning Strategy

## Models
- **BERT (prajjwal1/bert-tiny)**
- **DeBERTa-v3**

Both models were fine-tuned using a two-stage training procedure.

## Training Procedure

### Stage 1 – Classifier Training
- Encoder layers frozen  
- Only the classification head trained  
- Learning rate: `5e-5`  
- Dropout: `0.2`  

### Stage 2 – Full Fine-Tuning
- All encoder layers unfrozen  
- Entire model trained jointly  
- Learning rate: `2e-5`  
- Dropout: `0.2` (optimized on split 1)  

---

## Data Splits
- Dataset divided into **three train/validation/test splits**  
- Hyperparameters (learning rate, dropout, epochs) were **optimized only on split 1**  
- The **same hyperparameters** were then applied to splits 2 and 3  
- This ensures **fair comparison, reproducibility, and unbiased evaluation**  
- Final results are reported as the **mean across the three splits**  

---

## Results

### BERT (with class weights for class imbalance)
- Accuracy: `0.69`  
- F1: `0.69`  

### DeBERTa-v3 (with focal loss for class imbalance)
- Accuracy: `0.71`  
- F1: `0.72`  

---

## Notes
- Raw data was refined and corrected using an LLM, which improved inference results.  

---

## Hyperparameter Policy
We followed established best practices in NLP and ML research:

- **Fixed hyperparameters across splits**  
  - Prevents information leakage  
  - Avoids inflated performance estimates  
  - Ensures comparability between models  

**References**  
- Mosbach et al. (2021), *On the Stability of Fine-Tuning BERT*  
- Cawley & Talbot (2010), *On Overfitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*  
- Scikit-learn Documentation, *Nested Cross-Validation*  
