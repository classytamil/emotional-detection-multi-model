"""
RAVDESS Benchmark: Wav2Vec2 Emotion Recognition
Evaluates model accuracy on 20% test split with full metrics and graphs.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
RAVDESS_ROOT  = "/Users/leadtapmacmini2/Downloads/multimodel/RAVDESS"
OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "benchmark_results")
MODEL_NAME    = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
SAMPLE_RATE   = 16000

# RAVDESS filename position [2] → emotion label
EMOTION_MAP = {
    1: "neutral", 2: "calm",    3: "happy",    4: "sad",
    5: "angry",   6: "fearful", 7: "disgust",  8: "surprised"
}

# ── Step 1: Build dataset from filenames ──────────────────────────────────────
def build_dataset(root: str) -> pd.DataFrame:
    rows = []
    for actor_folder in sorted(os.listdir(root)):
        actor_path = os.path.join(root, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        for filename in os.listdir(actor_path):
            if not filename.endswith(".wav"):
                continue
            parts = filename.split("-")
            emotion_code = int(parts[2])
            actor_id     = int(parts[6].split(".")[0])
            emotion      = EMOTION_MAP[emotion_code]
            gender       = "male" if actor_id % 2 == 1 else "female"
            rows.append({
                "filename": filename,
                "path": os.path.join(actor_path, filename),
                "emotion": emotion,
                "actor":   actor_id,
                "gender":  gender,
            })
    df = pd.DataFrame(rows)
    print(f"Dataset loaded: {len(df)} files  |  "
          f"{df['emotion'].nunique()} emotion classes  |  "
          f"{df['actor'].nunique()} actors")
    print(df["emotion"].value_counts().to_string())
    return df


# ── Step 2: Load model ────────────────────────────────────────────────────────
def _remap_checkpoint(state_dict):
    """
    ehcalabres checkpoint uses classifier.dense/output naming;
    current Wav2Vec2ForSequenceClassification expects projector/classifier.
    """
    remap = {
        "classifier.dense.weight":  "projector.weight",
        "classifier.dense.bias":    "projector.bias",
        "classifier.output.weight": "classifier.weight",
        "classifier.output.bias":   "classifier.bias",
    }
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[remap.get(k, k)] = v
    return new_sd


def load_model():
    local = os.path.join(os.path.dirname(__file__), "audio_emotion_model")
    src = local if os.path.exists(os.path.join(local, "config.json")) else MODEL_NAME
    print(f"\nLoading model from: {src}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(src)

    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(src, "pytorch_model.bin")
    raw_sd   = torch.load(ckpt_path, map_location="cpu")
    fixed_sd = _remap_checkpoint(raw_sd)

    from transformers import Wav2Vec2Config
    cfg = Wav2Vec2Config.from_pretrained(src)
    cfg.classifier_proj_size = 1024  # checkpoint was trained with proj=1024 not 256
    model = Wav2Vec2ForSequenceClassification(cfg)
    model.load_state_dict(fixed_sd, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Model ready on {device}")
    return processor, model, device


# ── Step 3: Inference ─────────────────────────────────────────────────────────
def predict_batch(paths, processor, model, device):
    preds = []
    label_map = model.config.id2label  # model's own index→label
    for path in tqdm(paths, desc="Inference", unit="file"):
        try:
            audio, _ = librosa.load(path, sr=SAMPLE_RATE)
            inputs = processor(
                audio, sampling_rate=SAMPLE_RATE,
                return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_id = int(torch.argmax(logits, dim=-1).item())
            preds.append(label_map[pred_id])
        except Exception as e:
            preds.append("neutral")  # fallback on corrupt file
    return preds


# ── Step 4: Metrics & Plots ───────────────────────────────────────────────────
def save_report(y_true, y_pred, labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, digits=4)

    print("\n" + "="*60)
    print(f"  ACCURACY : {acc*100:.2f}%")
    print("="*60)
    print(report)

    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc*100:.2f}%\n\n")
        f.write(report)

    # Confusion matrix heatmap
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Wav2Vec2 Confusion Matrix — RAVDESS", fontsize=14)
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {cm_path}")

    # Per-class F1 bar chart
    from sklearn.metrics import f1_score
    f1_scores = f1_score(y_true, y_pred, labels=labels, average=None)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bars = ax2.bar(labels, f1_scores * 100, color=sns.color_palette("husl", len(labels)))
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("F1 Score (%)", fontsize=12)
    ax2.set_title("Per-Class F1 Score — Wav2Vec2 on RAVDESS", fontsize=14)
    for bar, val in zip(bars, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val*100:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    f1_path = os.path.join(out_dir, "f1_per_class.png")
    fig2.savefig(f1_path, dpi=150)
    plt.close(fig2)
    print(f"F1 chart saved       → {f1_path}")

    return acc, cm, report


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Dataset
    df = build_dataset(RAVDESS_ROOT)

    # 2. Train / test split (stratified)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df["emotion"]
    )
    print(f"\nSplit → Train: {len(train_df)}  |  Test: {len(test_df)}")

    # 3. Load model
    processor, model, device = load_model()

    # 4. Inference on test set only
    y_true = test_df["emotion"].tolist()
    y_pred = predict_batch(test_df["path"].tolist(), processor, model, device)

    # 5. Labels in consistent order
    labels = sorted(EMOTION_MAP.values())  # alphabetical

    # 6. Save metrics + plots
    acc, cm, report = save_report(y_true, y_pred, labels, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("DONE.")


if __name__ == "__main__":
    main()
