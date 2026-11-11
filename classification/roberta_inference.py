import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\aakif\final year"

MODEL_DIR = os.path.join(
    BASE_DIR,
    "classification",
    "roberta_final_model"
)

MAX_LENGTH = 128


# ============================================================
# CLASS LABELS
# ============================================================

LABELS = {
    0: "Simple",
    1: "Complex"
}


# ============================================================
# DEVICE
# ============================================================

print("=" * 60)
print("RoBERTa FINAL INFERENCE")
print("=" * 60)

if torch.cuda.is_available():

    DEVICE = torch.device("cuda")

    print("Device: CUDA")
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

else:

    DEVICE = torch.device("cpu")

    print("Device: CPU")


# ============================================================
# CHECK MODEL DIRECTORY
# ============================================================

if not os.path.exists(MODEL_DIR):

    print("\nERROR: Model directory not found!")
    print(MODEL_DIR)

    raise SystemExit


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

print("Tokenizer loaded successfully.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading final RoBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)

model.eval()

print("Final RoBERTa model loaded successfully.")


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

print("\nModel information:")
print("Number of labels:", model.config.num_labels)

print(
    "ID 0:",
    model.config.id2label.get(0, "Simple")
)

print(
    "ID 1:",
    model.config.id2label.get(1, "Complex")
)

print("Maximum sequence length:", MAX_LENGTH)

print("=" * 60)


# ============================================================
# CLASSIFICATION FUNCTION
# ============================================================

def classify_question(question):

    # --------------------------------------------------------
    # Tokenize question
    # --------------------------------------------------------

    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    # --------------------------------------------------------
    # Move tensors to GPU/CPU
    # --------------------------------------------------------

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    # --------------------------------------------------------
    # Get probabilities
    # --------------------------------------------------------

    simple_probability = probabilities[0].item()

    complex_probability = probabilities[1].item()

    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    predicted_class = torch.argmax(
        probabilities
    ).item()

    predicted_label = LABELS[
        predicted_class
    ]

    confidence = probabilities[
        predicted_class
    ].item()

    return (
        predicted_label,
        confidence,
        simple_probability,
        complex_probability
    )


# ============================================================
# INTERACTIVE CLASSIFIER
# ============================================================

print("\n")
print("=" * 60)
print("QUESTION CLASSIFIER")
print("=" * 60)

print("\nEnter a question to classify.")
print("Type 'exit' to stop.\n")


while True:

    # --------------------------------------------------------
    # Get question
    # --------------------------------------------------------

    question = input("Question: ").strip()

    # --------------------------------------------------------
    # Exit
    # --------------------------------------------------------

    if question.lower() == "exit":

        print("\nClassifier stopped.")

        break

    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if not question:

        print("\nPlease enter a question.")
        print("-" * 60)

        continue

    # --------------------------------------------------------
    # Classify
    # --------------------------------------------------------

    try:

        (
            prediction,
            confidence,
            simple_probability,
            complex_probability
        ) = classify_question(question)

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        print("\nPrediction :", prediction)

        print(
            "Confidence :",
            f"{confidence * 100:.2f}%"
        )

        print(
            "Simple probability :",
            f"{simple_probability * 100:.2f}%"
        )

        print(
            "Complex probability:",
            f"{complex_probability * 100:.2f}%"
        )

        print("-" * 60)

    except Exception as e:

        print("\nERROR during classification:")
        print(e)

        print("-" * 60)


# ============================================================
# END
# ============================================================

print("\n")
print("=" * 60)
print("RoBERTa INFERENCE FINISHED")
print("=" * 60)