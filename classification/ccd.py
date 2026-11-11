import os
import pandas as pd
from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\aakif\final year"

HOTPot_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "hotpot_qa",
    "distractor"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "classification_dataset.csv"
)

RANDOM_SEED = 42

# Maximum number of examples from each class.
# None = use all available balanced examples.
MAX_PER_CLASS = 30000


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CREATING SIMPLE / COMPLEX CLASSIFICATION DATASET")
print("=" * 70)


# ============================================================
# 1. LOAD SIMPLE QUESTIONS FROM SQuAD
# ============================================================

print("\n[1/5] Loading SQuAD dataset...")
print("SQuAD questions will be labeled as SIMPLE (0).")

try:

    squad = load_dataset(
        "rajpurkar/squad"
    )

except Exception as e:

    print("\nERROR while downloading/loading SQuAD:")
    print(e)

    print(
        "\nMake sure you have an internet connection "
        "and the 'datasets' package is installed."
    )

    raise


# Combine SQuAD train and validation
squad_train = squad["train"]
squad_validation = squad["validation"]


squad_train_df = pd.DataFrame({
    "question": squad_train["question"]
})

squad_validation_df = pd.DataFrame({
    "question": squad_validation["question"]
})


simple_df = pd.concat(
    [
        squad_train_df,
        squad_validation_df
    ],
    ignore_index=True
)


print(
    "SQuAD questions loaded:",
    len(simple_df)
)


# ============================================================
# 2. CLEAN SQuAD
# ============================================================

simple_df["question"] = (
    simple_df["question"]
    .astype(str)
    .str.strip()
)


simple_df = simple_df[
    simple_df["question"].str.len() > 0
]


simple_df = simple_df.drop_duplicates(
    subset=["question"]
).reset_index(drop=True)


# Label:
# 0 = Simple

simple_df["label"] = 0


print(
    "Unique Simple questions:",
    len(simple_df)
)


# ============================================================
# 3. LOAD HOTPOTQA
# ============================================================

print("\n[2/5] Loading HotpotQA...")
print("HotpotQA questions will be labeled as COMPLEX (1).")


if not os.path.exists(HOTPot_DIR):

    raise FileNotFoundError(
        "\nHotpotQA directory was not found:\n"
        + HOTPot_DIR
    )


parquet_files = []

for filename in os.listdir(HOTPot_DIR):

    if filename.endswith(".parquet"):

        parquet_files.append(
            os.path.join(
                HOTPot_DIR,
                filename
            )
        )


parquet_files.sort()


if len(parquet_files) == 0:

    raise FileNotFoundError(
        "\nNo .parquet files were found in:\n"
        + HOTPot_DIR
    )


print("\nHotpotQA files found:")

for file in parquet_files:

    print(
        " -",
        os.path.basename(file)
    )


# ============================================================
# LOAD ALL HOTPOTQA PARQUET FILES
# ============================================================

hotpot_frames = []


for file in parquet_files:

    print(
        "\nReading:",
        os.path.basename(file)
    )

    temp_df = pd.read_parquet(
        file
    )

    print(
        "Samples:",
        len(temp_df)
    )

    if "question" not in temp_df.columns:

        raise ValueError(
            "The file does not contain a 'question' column:\n"
            + file
        )

    hotpot_frames.append(
        temp_df[["question"]]
    )


complex_df = pd.concat(
    hotpot_frames,
    ignore_index=True
)


print(
    "\nTotal HotpotQA questions:",
    len(complex_df)
)


# ============================================================
# 4. CLEAN HOTPOTQA
# ============================================================

complex_df["question"] = (
    complex_df["question"]
    .astype(str)
    .str.strip()
)


complex_df = complex_df[
    complex_df["question"].str.len() > 0
]


complex_df = complex_df.drop_duplicates(
    subset=["question"]
).reset_index(drop=True)


# Label:
# 1 = Complex

complex_df["label"] = 1


print(
    "Unique Complex questions:",
    len(complex_df)
)


# ============================================================
# 5. BALANCE DATASET
# ============================================================

print("\n[3/5] Balancing Simple and Complex classes...")


available = min(
    len(simple_df),
    len(complex_df)
)


if MAX_PER_CLASS is None:

    samples_per_class = available

else:

    samples_per_class = min(
        MAX_PER_CLASS,
        available
    )


if samples_per_class == 0:

    raise ValueError(
        "Could not create a balanced dataset."
    )


simple_df = simple_df.sample(
    n=samples_per_class,
    random_state=RANDOM_SEED
)


complex_df = complex_df.sample(
    n=samples_per_class,
    random_state=RANDOM_SEED
)


print(
    "Samples per class:",
    samples_per_class
)


# ============================================================
# 6. COMBINE
# ============================================================

print("\n[4/5] Combining datasets...")


classification_df = pd.concat(
    [
        simple_df,
        complex_df
    ],
    ignore_index=True
)


# Keep only required columns

classification_df = classification_df[
    [
        "question",
        "label"
    ]
]


# Shuffle

classification_df = classification_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# Final duplicate removal

classification_df = classification_df.drop_duplicates(
    subset=["question"]
).reset_index(drop=True)


# ============================================================
# 7. SAVE
# ============================================================

print("\n[5/5] Saving dataset...")


os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)


classification_df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8"
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION DATASET CREATED SUCCESSFULLY")
print("=" * 70)


print(
    "\nOutput file:"
)

print(
    OUTPUT_PATH
)


print(
    "\nTotal questions:",
    len(classification_df)
)


print(
    "\nClass distribution:"
)


counts = (
    classification_df["label"]
    .value_counts()
    .sort_index()
)


for label, count in counts.items():

    if label == 0:

        class_name = "Simple"

    else:

        class_name = "Complex"


    print(
        f"  {label} = {class_name}: {count}"
    )


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\n")
print("=" * 70)
print("SAMPLE QUESTIONS")
print("=" * 70)


print("\nSIMPLE QUESTIONS (0):")


simple_examples = classification_df[
    classification_df["label"] == 0
]["question"].head(5)


for i, question in enumerate(
    simple_examples,
    start=1
):

    print(
        f"{i}. {question}"
    )


print("\nCOMPLEX QUESTIONS (1):")


complex_examples = classification_df[
    classification_df["label"] == 1
]["question"].head(5)


for i, question in enumerate(
    complex_examples,
    start=1
):

    print(
        f"{i}. {question}"
    )


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)