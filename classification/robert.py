import os
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset, ClassLabel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "classification",
    "roberta_results"
)

MODEL_OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "classification",
    "roberta_model"
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
print("RoBERTa CLASSIFICATION")
print("=" * 60)

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("Device: cuda")

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )

else:

    DEVICE = torch.device("cpu")

    print("Device: cpu")
    print("WARNING: GPU not detected.")

print("=" * 60)


# ============================================================
# CHECK DATASET
# ============================================================

print("\nLoading dataset...")

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        "\nDataset not found:\n"
        + DATASET_PATH
    )

df = pd.read_csv(DATASET_PATH)

print(
    "Total samples:",
    len(df)
)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "question",
    "label"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"\nRequired column missing: {column}"
        )


# ============================================================
# CLEAN DATA
# ============================================================

df["question"] = (
    df["question"]
    .astype(str)
    .str.strip()
)

df["label"] = pd.to_numeric(
    df["label"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "question",
        "label"
    ]
)

df["label"] = (
    df["label"]
    .astype(int)
)

df = df[
    df["question"].str.len() > 0
]

df = df[
    df["label"].isin([0, 1])
]

df = df.drop_duplicates(
    subset=["question"]
).reset_index(drop=True)


print("\nAfter cleaning:")

print(
    "Total samples:",
    len(df)
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

class_counts = (
    df["label"]
    .value_counts()
    .sort_index()
)

for label, count in class_counts.items():

    if label == 0:
        name = "Simple"

    else:
        name = "Complex"

    print(
        f"{label} ({name}): {count}"
    )

if len(class_counts) != 2:

    raise ValueError(
        "\nDataset must contain both classes."
    )


# ============================================================
# CONVERT TO HUGGING FACE DATASET
# ============================================================

dataset = Dataset.from_pandas(
    df[
        [
            "question",
            "label"
        ]
    ],
    preserve_index=False
)


# ============================================================
# CONVERT LABEL TO CLASSLABEL
# ============================================================

print("\nPreparing labels...")

dataset = dataset.cast_column(
    "label",
    ClassLabel(
        num_classes=2,
        names=[
            "Simple",
            "Complex"
        ]
    )
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print(
    "\nCreating train/validation/test split..."
)


# 80% training
# 20% temporary

split_1 = dataset.train_test_split(
    test_size=0.20,
    seed=RANDOM_SEED,
    stratify_by_column="label"
)

train_dataset = split_1["train"]

temp_dataset = split_1["test"]


# 50% validation
# 50% test

split_2 = temp_dataset.train_test_split(
    test_size=0.50,
    seed=RANDOM_SEED,
    stratify_by_column="label"
)

validation_dataset = split_2["train"]

test_dataset = split_2["test"]


print(
    f"Training samples:   {len(train_dataset)}"
)

print(
    f"Validation samples: {len(validation_dataset)}"
)

print(
    f"Test samples:       {len(test_dataset)}"
)


# ============================================================
# TOKENIZER
# ============================================================

print(
    "\nLoading RoBERTa tokenizer..."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print(
    "Tokenizer loaded successfully."
)


# ============================================================
# TOKENIZATION
# ============================================================

print(
    "\nTokenizing dataset..."
)


def tokenize_function(examples):

    return tokenizer(
        examples["question"],
        truncation=True,
        max_length=MAX_LENGTH
    )


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing training data"
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing validation data"
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    desc="Tokenizing test data"
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

print(
    "\nLoading RoBERTa model..."
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label={
        0: "Simple",
        1: "Complex"
    },
    label2id={
        "Simple": 0,
        "Complex": 1
    }
)

model.to(DEVICE)

print(
    "RoBERTa model loaded successfully."
)


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

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0
        )
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

print(
    "\nPreparing training configuration..."
)

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    num_train_epochs=NUM_EPOCHS,

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=100,

    load_best_model_at_end=True,

    metric_for_best_model="f1",

    greater_is_better=True,

    fp16=torch.cuda.is_available(),

    dataloader_pin_memory=torch.cuda.is_available(),

    seed=RANDOM_SEED,

    save_total_limit=2,

    report_to="none"
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ============================================================
# START TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("STARTING RoBERTa TRAINING")
print("=" * 60)

if torch.cuda.is_available():

    torch.cuda.empty_cache()

trainer.train()


print("\n")
print("=" * 60)
print("RoBERTa TRAINING COMPLETE")
print("=" * 60)


# ============================================================
# SAVE MODEL
# ============================================================

print(
    "\nSaving trained RoBERTa model..."
)

os.makedirs(
    MODEL_OUTPUT_DIR,
    exist_ok=True
)

trainer.save_model(
    MODEL_OUTPUT_DIR
)

tokenizer.save_pretrained(
    MODEL_OUTPUT_DIR
)

print(
    "Model saved to:"
)

print(
    MODEL_OUTPUT_DIR
)


# ============================================================
# VALIDATION RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

validation_results = trainer.evaluate(
    eval_dataset=validation_dataset
)

for key, value in validation_results.items():

    if isinstance(value, float):

        print(
            f"{key}: {value:.4f}"
        )


# ============================================================
# TEST RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

test_results = trainer.predict(
    test_dataset
)

test_predictions = np.argmax(
    test_results.predictions,
    axis=1
)

test_labels = np.array(
    test_results.label_ids
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

print(
    f"\nAccuracy : {test_accuracy:.4f}"
)

print(
    f"Precision: {test_precision:.4f}"
)

print(
    f"Recall   : {test_recall:.4f}"
)

print(
    f"F1 Score : {test_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

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

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    test_labels,
    test_predictions
)

print(
    "\n                 Predicted"
)

print(
    "                 Simple  Complex"
)

print(
    f"Actual Simple    {cm[0,0]:6d}  {cm[0,1]:7d}"
)

print(
    f"Actual Complex   {cm[1,0]:6d}  {cm[1,1]:7d}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "roberta_test_results.txt"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

with open(
    results_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "RoBERTa CLASSIFICATION RESULTS\n"
    )

    f.write(
        "=" * 60
        + "\n\n"
    )

    f.write(
        f"Dataset samples: {len(df)}\n"
    )

    f.write(
        f"Training samples: {len(train_dataset)}\n"
    )

    f.write(
        f"Validation samples: {len(validation_dataset)}\n"
    )

    f.write(
        f"Test samples: {len(test_dataset)}\n\n"
    )

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
        f"F1 Score : {test_f1:.4f}\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write(
        "-" * 60
        + "\n"
    )

    f.write(report)

    f.write(
        "\nConfusion Matrix\n"
    )

    f.write(
        "-" * 60
        + "\n"
    )

    f.write(
        str(cm)
    )


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 60)
print("RoBERTa CLASSIFICATION FINISHED")
print("=" * 60)

print(
    "\nFinal Test F1:",
    f"{test_f1:.4f}"
)

print(
    "\nResults saved to:"
)

print(
    results_path
)

print(
    "\nTrained model saved to:"
)

print(
    MODEL_OUTPUT_DIR
)

if torch.cuda.is_available():

    print(
        "\nGPU was used successfully:"
    )

    print(
        torch.cuda.get_device_name(0)
    )

print("\nDone.")