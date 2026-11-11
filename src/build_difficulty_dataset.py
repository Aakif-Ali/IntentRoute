import os
import json
import time
import requests
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

MATH_DIR = r".\dataset\hendrycks_math\balanced"

QA_TRAIN = r".\dataset\combined_qa\train.parquet"
QA_VAL = r".\dataset\combined_qa\validation.parquet"

OUTPUT_DIR = r".\dataset\difficulty_general"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

# Number of QA questions to use
TOTAL_QA = 2000

# Smaller batch = more reliable structured output
BATCH_SIZE = 20

# Maximum number of retries for one batch
MAX_RETRIES = 3

# Final balanced dataset
TARGET_PER_CLASS = 3000

RANDOM_STATE = 42


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MATH DATA
# ============================================================

def load_math():

    print("\n" + "=" * 70)
    print("LOADING MATH DATA")
    print("=" * 70)

    files = [
        os.path.join(MATH_DIR, "train.parquet"),
        os.path.join(MATH_DIR, "validation.parquet"),
        os.path.join(MATH_DIR, "test.parquet")
    ]

    frames = []

    for file in files:

        if os.path.exists(file):

            df = pd.read_parquet(file)

            print(
                f"{os.path.basename(file):20s} -> "
                f"{len(df):6d} rows"
            )

            frames.append(df)

    if not frames:

        raise FileNotFoundError(
            f"No MATH parquet files found in:\n{MATH_DIR}"
        )

    math_df = pd.concat(
        frames,
        ignore_index=True
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use the existing MATH labels.
    # DO NOT recalculate them.
    # --------------------------------------------------------

    if "label" not in math_df.columns:

        raise ValueError(
            "MATH dataset does not contain a 'label' column."
        )

    math_df = math_df[
        math_df["problem"].notna()
    ].copy()

    math_df["question"] = (
        math_df["problem"]
        .astype(str)
        .str.strip()
    )

    math_df["label"] = pd.to_numeric(
        math_df["label"],
        errors="coerce"
    )

    math_df = math_df[
        math_df["label"].isin([0, 1])
    ].copy()

    math_df["label"] = (
        math_df["label"]
        .astype(int)
    )

    math_df["source"] = "MATH"

    math_df = math_df[
        [
            "question",
            "label",
            "source"
        ]
    ]

    math_df = math_df.drop_duplicates(
        subset=["question"]
    )

    print("\nMATH distribution:")

    distribution = (
        math_df["label"]
        .value_counts()
        .sort_index()
    )

    print(
        f"SIMPLE  : {distribution.get(0, 0)}"
    )

    print(
        f"COMPLEX : {distribution.get(1, 0)}"
    )

    print(
        f"\nTotal MATH questions: {len(math_df)}"
    )

    return math_df


# ============================================================
# LOAD QA DATA
# ============================================================

def load_qa():

    print("\n" + "=" * 70)
    print("LOADING QA DATA")
    print("=" * 70)

    if not os.path.exists(QA_TRAIN):

        raise FileNotFoundError(
            f"Missing:\n{QA_TRAIN}"
        )

    if not os.path.exists(QA_VAL):

        raise FileNotFoundError(
            f"Missing:\n{QA_VAL}"
        )

    train = pd.read_parquet(
        QA_TRAIN
    )

    validation = pd.read_parquet(
        QA_VAL
    )

    qa = pd.concat(
        [
            train,
            validation
        ],
        ignore_index=True
    )

    print(
        f"Combined QA rows: {len(qa)}"
    )

    if "question" not in qa.columns:

        raise ValueError(
            "QA dataset does not contain 'question' column."
        )

    qa = qa[
        qa["question"].notna()
    ].copy()

    qa["question"] = (
        qa["question"]
        .astype(str)
        .str.strip()
    )

    qa = qa[
        qa["question"].str.len() > 5
    ]

    qa = qa.drop_duplicates(
        subset=["question"]
    )

    # --------------------------------------------------------
    # Random sample
    # --------------------------------------------------------

    sample_size = min(
        TOTAL_QA,
        len(qa)
    )

    qa = qa.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    print(
        f"QA selected for teacher labeling: "
        f"{len(qa)}"
    )

    return qa


# ============================================================
# TEST OLLAMA
# ============================================================

def test_ollama():

    print("\n" + "=" * 70)
    print("TESTING OLLAMA")
    print("=" * 70)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": (
            'Return exactly this JSON: '
            '{"label":"SIMPLE"}'
        ),
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 20
        }
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120
        )

        response.raise_for_status()

        result = response.json()

        output = result.get(
            "response",
            ""
        )

        print(
            "Ollama connection: OK"
        )

        print(
            f"Model: {OLLAMA_MODEL}"
        )

        print(
            f"Test response: {output}"
        )

    except Exception as e:

        print(
            "\nERROR connecting to Ollama:"
        )

        print(e)

        raise SystemExit(
            "\nMake sure Ollama is running."
        )


# ============================================================
# QWEN3 BATCH LABELING
# ============================================================

def label_batch(questions):

    """
    Send multiple questions to Qwen3.

    Qwen3 is forced to return JSON.

    Example:

    {
        "labels": [
            "SIMPLE",
            "COMPLEX",
            "SIMPLE"
        ]
    }
    """

    numbered_questions = []

    for i, question in enumerate(
        questions,
        start=1
    ):

        numbered_questions.append(
            f"{i}. {question}"
        )

    question_text = "\n".join(
        numbered_questions
    )

    prompt = f"""
You are a dataset annotation expert.

Your task is to classify the difficulty of each question.

Use ONLY these two labels:

SIMPLE
COMPLEX

Definition of SIMPLE:
- Direct factual question
- Definition
- Identification
- Basic explanation
- Straightforward calculation
- Usually answerable with little reasoning
- Does not require combining several reasoning steps

Definition of COMPLEX:
- Requires multiple reasoning steps
- Requires analysis or inference
- Requires combining multiple pieces of information
- Requires derivation or substantial problem solving
- Requires planning or multi-step reasoning

IMPORTANT:

Judge the difficulty of the QUESTION itself.

Do NOT classify a question as complex simply because its topic is advanced.

For example:

"What is photosynthesis?"
= SIMPLE

"Who is Lionel Messi?"
= SIMPLE

"How does photosynthesis convert light energy into chemical energy?"
= SIMPLE or COMPLEX depending on the reasoning required.

"How would you design an experiment to determine which environmental variables most strongly affect photosynthesis?"
= COMPLEX

Return exactly one label for every question.

Questions:

{question_text}
"""

    schema = {
        "type": "object",
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "SIMPLE",
                        "COMPLEX"
                    ]
                }
            }
        },
        "required": [
            "labels"
        ]
    }

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,

        # Structured JSON output
        "format": schema,

        "options": {
            "temperature": 0,
            "num_predict": BATCH_SIZE * 5
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    output = result.get(
        "response",
        ""
    )

    if not output:

        return None

    try:

        parsed = json.loads(
            output
        )

    except json.JSONDecodeError:

        print(
            "\nQwen3 returned invalid JSON:"
        )

        print(output[:1000])

        return None

    labels = parsed.get(
        "labels"
    )

    if not isinstance(
        labels,
        list
    ):

        return None

    if len(labels) != len(questions):

        print(
            f"\nWrong number of labels: "
            f"expected {len(questions)}, "
            f"received {len(labels)}"
        )

        return None

    cleaned_labels = []

    for label in labels:

        label = str(
            label
        ).upper().strip()

        if label not in [
            "SIMPLE",
            "COMPLEX"
        ]:

            return None

        cleaned_labels.append(
            label
        )

    return cleaned_labels


# ============================================================
# ANNOTATE QA
# ============================================================

def annotate_qa(qa):

    print("\n" + "=" * 70)
    print("QWEN3 QA DIFFICULTY ANNOTATION")
    print("=" * 70)

    checkpoint_file = os.path.join(
        OUTPUT_DIR,
        "qa_teacher_annotations_checkpoint.parquet"
    )

    # --------------------------------------------------------
    # Load existing checkpoint
    # --------------------------------------------------------

    annotations = []

    if os.path.exists(
        checkpoint_file
    ):

        old = pd.read_parquet(
            checkpoint_file
        )

        print(
            f"Existing checkpoint found: "
            f"{len(old)} labeled"
        )

        annotations = old.to_dict(
            orient="records"
        )

        labeled_questions = set(
            old["question"].astype(str)
        )

        qa = qa[
            ~qa["question"].isin(
                labeled_questions
            )
        ].copy()

    print(
        f"Remaining questions: {len(qa)}"
    )

    if len(qa) == 0:

        return pd.DataFrame(
            annotations
        )

    total = len(qa)

    successful_batches = 0
    failed_batches = 0

    # --------------------------------------------------------
    # Process batches
    # --------------------------------------------------------

    for start in range(
        0,
        total,
        BATCH_SIZE
    ):

        batch = qa.iloc[
            start:start + BATCH_SIZE
        ]

        questions = batch[
            "question"
        ].tolist()

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        total_batches = (
            (total + BATCH_SIZE - 1)
            // BATCH_SIZE
        )

        print(
            f"\nBatch {batch_number}/{total_batches}"
        )

        print(
            f"Questions: "
            f"{start + 1}-"
            f"{start + len(questions)}"
        )

        labels = None

        # ----------------------------------------------------
        # Retry
        # ----------------------------------------------------

        for attempt in range(
            1,
            MAX_RETRIES + 1
        ):

            try:

                labels = label_batch(
                    questions
                )

                if labels is not None:

                    break

                print(
                    f"Invalid response "
                    f"(attempt "
                    f"{attempt}/{MAX_RETRIES})"
                )

            except Exception as e:

                print(
                    f"Request error "
                    f"(attempt "
                    f"{attempt}/{MAX_RETRIES}):"
                )

                print(e)

            time.sleep(2)

        # ----------------------------------------------------
        # Successful batch
        # ----------------------------------------------------

        if labels is not None:

            for question, label in zip(
                questions,
                labels
            ):

                annotations.append({

                    "question": question,

                    "label": (
                        0
                        if label == "SIMPLE"
                        else 1
                    ),

                    "teacher_label": label,

                    "source": "QA_TEACHER"

                })

            successful_batches += 1

            print(
                "Status: SUCCESS"
            )

        # ----------------------------------------------------
        # Failed batch
        # ----------------------------------------------------

        else:

            failed_batches += 1

            print(
                "Status: FAILED"
            )

            print(
                "This batch will NOT be "
                "marked as completed."
            )

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        checkpoint_df = pd.DataFrame(
            annotations
        )

        checkpoint_df.to_parquet(
            checkpoint_file,
            index=False
        )

        completed_questions = min(
            start + BATCH_SIZE,
            total
        )

        labeled_count = len(
            annotations
        )

        print(
            f"Progress: "
            f"{completed_questions}/{total} "
            f"batches processed"
        )

        print(
            f"Successfully labeled: "
            f"{labeled_count}"
        )

        print(
            f"Successful batches: "
            f"{successful_batches}"
        )

        print(
            f"Failed batches: "
            f"{failed_batches}"
        )

        # ----------------------------------------------------
        # Small cooling pause
        # ----------------------------------------------------

        time.sleep(1)

    print("\n" + "-" * 70)

    print(
        f"Annotation finished."
    )

    print(
        f"Successful batches: "
        f"{successful_batches}"
    )

    print(
        f"Failed batches: "
        f"{failed_batches}"
    )

    print(
        f"Total labeled questions: "
        f"{len(annotations)}"
    )

    return pd.DataFrame(
        annotations
    )


# ============================================================
# BALANCE DATASET
# ============================================================

def balance_dataset(
    math_df,
    qa_df
):

    print("\n" + "=" * 70)
    print("BUILDING FINAL DATASET")
    print("=" * 70)

    qa_df = qa_df[
        [
            "question",
            "label",
            "source"
        ]
    ].copy()

    combined = pd.concat(
        [
            math_df,
            qa_df
        ],
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=["question"]
    )

    print("\nCombined distribution:")

    print(
        f"SIMPLE  : "
        f"{len(combined[combined['label'] == 0])}"
    )

    print(
        f"COMPLEX : "
        f"{len(combined[combined['label'] == 1])}"
    )

    simple = combined[
        combined["label"] == 0
    ]

    complex_ = combined[
        combined["label"] == 1
    ]

    available = min(
        len(simple),
        len(complex_)
    )

    target = min(
        TARGET_PER_CLASS,
        available
    )

    print(
        f"\nBalancing to "
        f"{target} samples per class."
    )

    simple = simple.sample(
        n=target,
        random_state=RANDOM_STATE
    )

    complex_ = complex_.sample(
        n=target,
        random_state=RANDOM_STATE
    )

    final_df = pd.concat(
        [
            simple,
            complex_
        ],
        ignore_index=True
    )

    final_df = final_df.sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(
        drop=True
    )

    print("\nFINAL DISTRIBUTION:")

    print(
        f"SIMPLE  : "
        f"{len(final_df[final_df['label'] == 0])}"
    )

    print(
        f"COMPLEX : "
        f"{len(final_df[final_df['label'] == 1])}"
    )

    print(
        f"TOTAL   : {len(final_df)}"
    )

    return final_df


# ============================================================
# CREATE TRAIN / VALIDATION / TEST
# ============================================================

def create_splits(df):

    print("\n" + "=" * 70)
    print("CREATING DATASET SPLITS")
    print("=" * 70)

    train, temp = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

    validation, test = train_test_split(
        temp,
        test_size=0.50,
        stratify=temp["label"],
        random_state=RANDOM_STATE
    )

    train_file = os.path.join(
        OUTPUT_DIR,
        "train.parquet"
    )

    validation_file = os.path.join(
        OUTPUT_DIR,
        "validation.parquet"
    )

    test_file = os.path.join(
        OUTPUT_DIR,
        "test.parquet"
    )

    train.to_parquet(
        train_file,
        index=False
    )

    validation.to_parquet(
        validation_file,
        index=False
    )

    test.to_parquet(
        test_file,
        index=False
    )

    print(
        f"Train      : {len(train)}"
    )

    print(
        f"Validation : {len(validation)}"
    )

    print(
        f"Test       : {len(test)}"
    )

    print("\nSaved files:")

    print(
        train_file
    )

    print(
        validation_file
    )

    print(
        test_file
    )

    return (
        train,
        validation,
        test
    )


# ============================================================
# SHOW EXAMPLES
# ============================================================

def show_samples(df):

    print("\n" + "=" * 70)
    print("DATASET QUALITY CHECK")
    print("=" * 70)

    for label, name in [
        (0, "SIMPLE"),
        (1, "COMPLEX")
    ]:

        print(
            f"\n--- {name} EXAMPLES ---"
        )

        subset = df[
            df["label"] == label
        ]

        if len(subset) == 0:
            continue

        sample = subset.sample(
            n=min(10, len(subset)),
            random_state=RANDOM_STATE
        )

        for i, (_, row) in enumerate(
            sample.iterrows(),
            start=1
        ):

            question = str(
                row["question"]
            )

            if len(question) > 300:

                question = (
                    question[:300]
                    + "..."
                )

            print(
                f"{i:2d}. "
                f"[{row['source']}] "
                f"{question}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)
    print("GENERAL DIFFICULTY DATASET BUILDER")
    print("=" * 70)

    print(
        "\nConfiguration:"
    )

    print(
        f"  MATH questions : existing labels"
    )

    print(
        f"  QA questions   : {TOTAL_QA}"
    )

    print(
        f"  Batch size     : {BATCH_SIZE}"
    )

    print(
        f"  Teacher        : {OLLAMA_MODEL}"
    )

    print(
        f"  Target/class   : {TARGET_PER_CLASS}"
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    math_df = load_math()

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    qa = load_qa()

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    test_ollama()

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    qa_annotations = annotate_qa(
        qa
    )

    if len(qa_annotations) == 0:

        raise RuntimeError(
            "\nNo QA annotations were created.\n"
            "Dataset cannot be built."
        )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    annotation_file = os.path.join(
        OUTPUT_DIR,
        "qa_teacher_annotations.parquet"
    )

    qa_annotations.to_parquet(
        annotation_file,
        index=False
    )

    print(
        "\nTeacher annotations saved:"
    )

    print(
        annotation_file
    )

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    print(
        "\nQA teacher distribution:"
    )

    simple_count = len(
        qa_annotations[
            qa_annotations["label"] == 0
        ]
    )

    complex_count = len(
        qa_annotations[
            qa_annotations["label"] == 1
        ]
    )

    print(
        f"SIMPLE  : {simple_count}"
    )

    print(
        f"COMPLEX : {complex_count}"
    )

    # --------------------------------------------------------
    # STEP 7
    # --------------------------------------------------------

    final_df = balance_dataset(
        math_df,
        qa_annotations
    )

    # --------------------------------------------------------
    # STEP 8
    # --------------------------------------------------------

    train, validation, test = create_splits(
        final_df
    )

    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    show_samples(
        final_df
    )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DATASET BUILD COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Do NOT train RoBERTa yet."
    )

    print(
        "First inspect the examples above."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()