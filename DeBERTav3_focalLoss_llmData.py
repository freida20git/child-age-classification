
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ------------------------
# Configuration
# ------------------------
model_name = "microsoft/deberta-v3-base"
num_labels = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Load tokenizer & model
# ------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

config = AutoConfig.from_pretrained(
    model_name,
    num_labels=num_labels,
    hidden_dropout_prob=0.2,
    attention_probs_dropout_prob=0.2
)

#model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config).to(device)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    use_safetensors=True
).to(device)

model.to(device)  # <-- move model to GPU (or CPU fallback)

# ------------------------
# Load dataset
# ------------------------

data_path = os.getcwd()

dataset = load_dataset(
    'csv',
    data_files={
        'train': os.path.join(data_path, 'llmData_train.csv'),
        'validation': os.path.join(data_path, 'llmData_validation.csv'),
        'test': os.path.join(data_path, 'llmData_test.csv')
    }
)

label_map = {"2_3": 0, "3_4": 1, "4_6": 2}

def tokenize_function(examples):
    return tokenizer(examples["child_text"], padding="max_length", truncation=True, max_length=512)

def encode_labels(example):
    example["labels"] = label_map[example["class_range"]]
    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.map(encode_labels)
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# ------------------------
# Compute class weights
# ------------------------
train_labels = [int(x) for x in tokenized_datasets['train']["labels"]]
class_counts = Counter(train_labels)
num_classes = len(class_counts)
total_samples = sum(class_counts.values())
class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class counts:", class_counts)
print("Class weights:", class_weights)

#--------------------------------
#  Focal Loss
#--------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # standard CE loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
            
class FocalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"]
        
        loss_fct = FocalLoss(alpha=1, gamma=2)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ------------------------
# Metrics
# ------------------------
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, average="weighted")
    }

early_stopping = EarlyStoppingCallback(early_stopping_patience=4)

# ==========================================================
# STEP 1: Train classifier head only (freeze encoder)
# =========================================================
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

for param in model.deberta.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

training_args_head = TrainingArguments(
    output_dir="./focal_results_head",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=30,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    dataloader_pin_memory=False, 
    report_to=[]
)

trainer_head = FocalTrainer(
    model=model,
    args=training_args_head,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)

print("\n======== Training classifier head only ========")
trainer_head.train()

trainer_head.save_model("./focal_results_head_checkpoint")

# ==========================================================
# STEP 2: Fine-tune full model (unfreeze encoder)
# ==========================================================
model = AutoModelForSequenceClassification.from_pretrained(
    "./focal_results_head_checkpoint",
    config=config
).to(device)


for param in model.deberta.parameters():
    param.requires_grad = True

training_args_full = TrainingArguments(
    output_dir="./focal_results_full",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=80,
    learning_rate=2e-5,  # smaller LR for stability
    load_best_model_at_end=True,
    dataloader_pin_memory=False, 
    report_to=[]
)

trainer_full = FocalTrainer(
    model=model,
    args=training_args_full,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)

print("\n======== Fine-tuning full model ========")
trainer_full.train()
# ==========================================================
# Plot training & validation loss curves
# ==========================================================
def plot_loss_curves(trainer, save_path="focal_loss_curve_llm.png"):
    logs = trainer.state.log_history
    train_loss = [x["loss"] for x in logs if "loss" in x]
    eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]

    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label="Training Loss")
    plt.plot(eval_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

plot_loss_curves(trainer_full)

# ------------------------
# Save final model
# ------------------------
trainer_full.save_model("./focal_deberta_llm")
# ==========================================================
# Confusion Matrix on validation set
# ==========================================================
preds_output = trainer_full.predict(tokenized_datasets['validation'])
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix (Validation)")
plt.savefig("confusion_matrix_llm.png")
plt.close()

# ==========================================================
# Final evaluation on TEST set
# ==========================================================
from sklearn.metrics import accuracy_score, f1_score, classification_report

print("\n======== Final Evaluation on Test Set ========")
preds_output = trainer_full.predict(tokenized_datasets['test'])
y_true = preds_output.label_ids
y_pred = np.argmax(preds_output.predictions, axis=1)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")

print(f"Accuracy: {acc}")
print(f"F1: {f1}")
print(classification_report(y_true, y_pred, digits=4))

# Optionally save to file
with open("focal_test_metrics_llm.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"F1: {f1}\n\n")
    f.write(classification_report(y_true, y_pred, digits=4))
