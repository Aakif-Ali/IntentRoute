import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "classification",
    "roberta_final_model"
)

RANDOM_SEED = 42
MAX_LENGTH = 128


# ============================================================
# DEVICE
# ============================================================

print("=" * 60)
print("RoBERTa SAVED MODEL VERIFICATION")
print("=" * 60)

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("Device: CUDA")
    print("GPU:", torch.cuda.get_device_name(0))

else:

    DEVICE = torch.device("cpu")

    print("Device: CPU")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

df = df[["question", "label"]].copy()

df["question"] = df["question"].astype(str).str.strip()

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(
    subset=["question", "label"]
)

df["label"] = df["label"].astype(int)

df = df[df["label"].isin([0, 1])]

df = df.drop_duplicates(
    subset=["question"]
).reset_index(drop=True)

print("Total samples:", len(df))


# ============================================================
# RECREATE EXACT SAME SPLIT
# ============================================================

print("\nRecreating test split...")

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

print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

print("Tokenizer loaded.")


# ============================================================
# LOAD SAVED MODEL
# ============================================================

print("\nLoading SAVED RoBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)

model.eval()

print("Saved model loaded successfully.")


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\nModel configuration:")

print("Number of labels:", model.config.num_labels)

print("Hidden size:", model.config.hidden_size)

print(
    "Classifier:",
    model.classifier
)

print(
    "Model type:",
    model.config.model_type
)


# ============================================================
# TOKENIZE TEST DATA
# ============================================================

test_dataset = Dataset.from_pandas(
    test_df[["question", "label"]],
    preserve_index=False
)


def tokenize_function(examples):

    return tokenizer(
        examples["question"],
        truncation=True,
        max_length=MAX_LENGTH
    )


print("\nTokenizing test dataset...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

print("Tokenization complete.")


# ============================================================
# REMOVE QUESTION COLUMN
# ============================================================

test_dataset = test_dataset.remove_columns(
    ["question"]
)


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# TRAINER FOR DIRECT EVALUATION
# ============================================================

training_args = TrainingArguments(
    output_dir=os.path.join(
        BASE_DIR,
        "classification",
        "verification_temp"
    ),
    per_device_eval_batch_size=16,
    report_to="none",
    fp16=False
)


trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=data_collator
)


# ============================================================
# RUN PREDICTION
# ============================================================

print("\n")
print("=" * 60)
print("TESTING SAVED MODEL ON 6,000 TEST QUESTIONS")
print("=" * 60)

prediction_output = trainer.predict(
    test_dataset
)

logits = prediction_output.predictions

labels = prediction_output.label_ids

predictions = np.argmax(
    logits,
    axis=1
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    labels,
    predictions
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0
    )
)


print("\n")
print("=" * 60)
print("SAVED MODEL RESULTS")
print("=" * 60)

print(
    "Accuracy :",
    f"{accuracy:.4f}"
)

print(
    "Precision:",
    f"{precision:.4f}"
)

print(
    "Recall   :",
    f"{recall:.4f}"
)

print(
    "F1 Score :",
    f"{f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    labels,
    predictions
)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
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
# PREDICTION DISTRIBUTION
# ============================================================

simple_count = np.sum(
    predictions == 0
)

complex_count = np.sum(
    predictions == 1
)

print("\n")
print("=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

print(
    "Predicted Simple :",
    simple_count
)

print(
    "Predicted Complex:",
    complex_count
)


# ============================================================
# CHECK INDIVIDUAL COMPLEX QUESTIONS
# ============================================================

print("\n")
print("=" * 60)
print("SAMPLE COMPLEX QUESTIONS")
print("=" * 60)

complex_indices = np.where(
    labels == 1
)[0][:10]


for index in complex_indices:

    question = test_df.iloc[index]["question"]

    prediction = predictions[index]

    probabilities = torch.softmax(
        torch.tensor(logits[index]),
        dim=0
    )

    simple_prob = probabilities[0].item()

    complex_prob = probabilities[1].item()

    print("\nQuestion:")
    print(question)

    print(
        "Actual:",
        "Complex"
    )

    print(
        "Predicted:",
        "Simple" if prediction == 0 else "Complex"
    )

    print(
        "Simple probability:",
        f"{simple_prob * 100:.2f}%"
    )

    print(
        "Complex probability:",
        f"{complex_prob * 100:.2f}%"
    )

    print("-" * 60)


print("\n")
print("=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)