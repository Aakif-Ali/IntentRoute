import os
import random
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\aakif\final year"

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "classification_dataset.csv"
)

MODEL_NAME = "roberta-base"

FINAL_MODEL_DIR = os.path.join(
    BASE_DIR,
    "classification",
    "roberta_final_model"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "classification",
    "roberta_final_results"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "roberta_final_test_results.txt"
)

RANDOM_SEED = 42

MAX_LENGTH = 128

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16

LEARNING_RATE = 2e-5

NUM_EPOCHS = 3

WEIGHT_DECAY = 0.01


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# GPU INFORMATION
# ============================================================

print("=" * 60)
print("FINAL RoBERTa CLASSIFICATION TRAINING")
print("=" * 60)

if torch.cuda.is_available():

    print("Device: CUDA")
    print("GPU:", torch.cuda.get_device_name(0))

    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    print(
        "GPU memory:",
        round(gpu_memory / (1024 ** 3), 2),
        "GB"
    )

else:

    print("Device: CPU")


print("=" * 60)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(FINAL_MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

import pandas as pd

df = pd.read_csv(DATASET_PATH)

print("Total samples:", len(df))

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# CLEAN DATA
# ============================================================

df = df[["question", "label"]].copy()

df["question"] = df["question"].astype(str).str.strip()

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(subset=["question", "label"])

df["label"] = df["label"].astype(int)

df = df[df["label"].isin([0, 1])]

df = df.drop_duplicates(
    subset=["question"]
).reset_index(drop=True)


print("\nAfter cleaning:")
print("Total samples:", len(df))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
    .rename(
        index={
            0: "0 (Simple)",
            1: "1 (Complex)"
        }
    )
)


# ============================================================
# SPLIT DATA
#
# 80% TRAIN
# 10% VALIDATION
# 10% TEST
#
# STRATIFIED
# ============================================================

print("\nCreating stratified dataset split...")

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_df["label"]
)


print("\nDataset split:")

print("Training   :", len(train_df))
print("Validation :", len(val_df))
print("Test       :", len(test_df))


# ============================================================
# CONVERT TO HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[["question", "label"]],
    preserve_index=False
)

val_dataset = Dataset.from_pandas(
    val_df[["question", "label"]],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[["question", "label"]],
    preserve_index=False
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading RoBERTa tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Tokenizer loaded successfully.")


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["question"],
        truncation=True,
        max_length=MAX_LENGTH
    )


print("\nTokenizing training data...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing train"
)


print("\nTokenizing validation data...")

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing validation"
)


print("\nTokenizing test data...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing test"
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# LOAD RoBERTa MODEL
# ============================================================

print("\nLoading RoBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

print("RoBERTa model loaded successfully.")


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=FINAL_MODEL_DIR,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=100,

    learning_rate=LEARNING_RATE,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    num_train_epochs=NUM_EPOCHS,

    weight_decay=WEIGHT_DECAY,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    save_total_limit=2,

    fp16=False,

    dataloader_pin_memory=torch.cuda.is_available(),

    seed=RANDOM_SEED,

    report_to="none"
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 60)
print("STARTING FINAL RoBERTa TRAINING")
print("=" * 60)

print("\nTraining samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

print("\nEpochs:", NUM_EPOCHS)
print("Train batch size:", TRAIN_BATCH_SIZE)
print("Evaluation batch size:", EVAL_BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Maximum sequence length:", MAX_LENGTH)

print("=" * 60)


train_result = trainer.train()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n")
print("=" * 60)
print("FINAL RoBERTa TRAINING COMPLETE")
print("=" * 60)

print("\nSaving final RoBERTa model...")

trainer.save_model(FINAL_MODEL_DIR)

tokenizer.save_pretrained(FINAL_MODEL_DIR)

print("\nFinal model saved to:")
print(FINAL_MODEL_DIR)


# ============================================================
# VALIDATION RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL VALIDATION RESULTS")
print("=" * 60)

validation_results = trainer.evaluate(
    eval_dataset=val_dataset
)

for key, value in validation_results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )

    else:

        print(
            f"{key}: {value}"
        )


# ============================================================
# FINAL TEST PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

test_output = trainer.predict(
    test_dataset
)

test_logits = test_output.predictions

test_labels = test_output.label_ids

test_predictions = np.argmax(
    test_logits,
    axis=1
)


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

test_precision, test_recall, test_f1, _ = (
    precision_recall_fscore_support(
        test_labels,
        test_predictions,
        average="binary",
        zero_division=0
    )
)


print("\nAccuracy :", f"{test_accuracy:.4f}")
print("Precision:", f"{test_precision:.4f}")
print("Recall   :", f"{test_recall:.4f}")
print("F1 Score :", f"{test_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    test_labels,
    test_predictions,
    target_names=[
        "Simple",
        "Complex"
    ],
    digits=4,
    zero_division=0
)

print("\n")
print("=" * 60)
print("FINAL CLASSIFICATION REPORT")
print("=" * 60)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print("\n")
print("=" * 60)
print("FINAL CONFUSION MATRIX")
print("=" * 60)

print("\n                 Predicted")
print("                 Simple  Complex")
print(
    f"Actual Simple    {cm[0][0]:6d}  {cm[0][1]:7d}"
)
print(
    f"Actual Complex   {cm[1][0]:6d}  {cm[1][1]:7d}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 60 + "\n")
    f.write("FINAL RoBERTa CLASSIFICATION RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write("MODEL\n")
    f.write("RoBERTa-base\n\n")

    f.write("DATASET\n")
    f.write(f"Total samples: {len(df)}\n")
    f.write(f"Training samples: {len(train_df)}\n")
    f.write(f"Validation samples: {len(val_df)}\n")
    f.write(f"Test samples: {len(test_df)}\n\n")

    f.write("FINAL TEST RESULTS\n")
    f.write("-" * 40 + "\n")

    f.write(
        f"Accuracy : {test_accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {test_precision:.4f}\n"
    )

    f.write(
        f"Recall   : {test_recall:.4f}\n"
    )

    f.write(
        f"F1 Score : {test_f1:.4f}\n"
    )

    f.write("\n\n")

    f.write("CLASSIFICATION REPORT\n")
    f.write("-" * 40 + "\n")

    f.write(report)

    f.write("\n\n")

    f.write("CONFUSION MATRIX\n")
    f.write("-" * 40 + "\n")

    f.write(
        f"Simple  -> Simple: {cm[0][0]}\n"
    )

    f.write(
        f"Simple  -> Complex: {cm[0][1]}\n"
    )

    f.write(
        f"Complex -> Simple: {cm[1][0]}\n"
    )

    f.write(
        f"Complex -> Complex: {cm[1][1]}\n"
    )

    f.write("\n\n")

    f.write("MODEL LOCATION\n")
    f.write(FINAL_MODEL_DIR)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("FINAL RoBERTa CLASSIFICATION FINISHED")
print("=" * 60)

print("\nFinal Test F1:", f"{test_f1:.4f}")

print("\nResults saved to:")
print(RESULTS_FILE)

print("\nFinal model saved to:")
print(FINAL_MODEL_DIR)

if torch.cuda.is_available():

    print("\nGPU used successfully:")
    print(torch.cuda.get_device_name(0))

print("\nDone.")