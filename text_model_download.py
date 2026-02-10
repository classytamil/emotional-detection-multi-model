from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Define save path for the text model
# Define save path for the text model
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotion_model_local")

# Switch to a different multilingual model that supports Tamil
model_name = "MilaNLProc/xlm-emo-t"

print(f"Downloading multilingual text emotion model: {model_name}...")
print("This may take a few minutes...")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)
print(f"[DONE] Tokenizer saved to {save_path}")

# Download model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.save_pretrained(save_path)
print(f"[DONE] Model saved to {save_path}")

print("\n[DONE] Multilingual text model download complete!")
print(f"Model location: {save_path}")
