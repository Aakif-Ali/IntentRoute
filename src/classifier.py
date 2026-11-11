

"""
src/classifier.py
-----------------
RoBERTa question complexity classifier for the Intelligent QA System.

Loads the trained roberta_final_model and provides:
  - classify(text)        -> single question classification
  - batch_classify(texts) -> batch classification for evaluation

Returns: label ("Simple"/"Complex"), confidence float, full probability dict.
"""

import os
import time
import logging
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

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

LABELS = {
    0: "Simple",
    1: "Complex"
}


# ============================================================
# ROBERTA CLASSIFIER
# ============================================================

class RoBERTaClassifier:
    """
    Encapsulated RoBERTa-based question complexity classifier.

    Attributes
    ----------
    device : torch.device
    model  : AutoModelForSequenceClassification
    tokenizer : AutoTokenizer
    """

    def __init__(self, model_dir: str = MODEL_DIR):

        self.model_dir = model_dir
        self._validate_model_dir()

        # --------------------------------------------------------
        # DEVICE SELECTION
        # Use CPU by default for RoBERTa (latency is only ~50ms), reserving
        # 100% of GPU VRAM for Ollama LLMs (qwen2.5:3b, qwen2.5-coder:3b).
        # --------------------------------------------------------
        use_cuda = os.getenv("ROBERTA_CUDA", "0") == "1"
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("RoBERTaClassifier: Using CUDA — %s", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            logger.info("RoBERTaClassifier: Using CPU (reserving 100%% VRAM for Ollama LLMs)")

        # --------------------------------------------------------
        # LOAD TOKENIZER
        # --------------------------------------------------------
        logger.info("Loading tokenizer from: %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # --------------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------------
        logger.info("Loading RoBERTa model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "RoBERTaClassifier ready. Labels: %s",
            {v: k for k, v in LABELS.items()}
        )

    def _validate_model_dir(self):
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"RoBERTa model directory not found:\n{self.model_dir}\n"
                "Expected at: classification/roberta_final_model"
            )
        required = ["config.json", "model.safetensors", "tokenizer.json"]
        for f in required:
            if not os.path.exists(os.path.join(self.model_dir, f)):
                raise FileNotFoundError(
                    f"Missing required file in model directory: {f}"
                )

    # --------------------------------------------------------
    # SINGLE CLASSIFICATION
    # --------------------------------------------------------

    def classify(self, text: str) -> Dict:
        """
        Classify a single question.

        Returns
        -------
        dict with keys:
            label      : "Simple" or "Complex"
            confidence : float (0.0–1.0), confidence in predicted label
            simple_prob: float, probability of Simple
            complex_prob: float, probability of Complex
            latency_ms : float, inference time in milliseconds
        """
        if not text or not text.strip():
            return {
                "label": "Simple",
                "confidence": 1.0,
                "simple_prob": 1.0,
                "complex_prob": 0.0,
                "latency_ms": 0.0,
            }

        t0 = time.perf_counter()

        inputs = self.tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        pred_idx = int(torch.argmax(probs).item())
        simple_prob  = float(probs[0].item())
        complex_prob = float(probs[1].item())
        confidence   = float(probs[pred_idx].item())

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "label":        LABELS[pred_idx],
            "confidence":   round(confidence, 4),
            "simple_prob":  round(simple_prob, 4),
            "complex_prob": round(complex_prob, 4),
            "latency_ms":   round(latency_ms, 2),
        }

    # --------------------------------------------------------
    # BATCH CLASSIFICATION
    # --------------------------------------------------------

    def batch_classify(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Classify a list of questions in batches.

        Returns a list of classification dicts (same structure as classify()).
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs_batch = torch.softmax(logits, dim=-1)

            for probs in probs_batch:
                pred_idx = int(torch.argmax(probs).item())
                results.append({
                    "label":        LABELS[pred_idx],
                    "confidence":   round(float(probs[pred_idx].item()), 4),
                    "simple_prob":  round(float(probs[0].item()), 4),
                    "complex_prob": round(float(probs[1].item()), 4),
                    "latency_ms":   None,
                })
        return results
