from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Define save path for the text model
save_path = r"E:\Projects\emotion detection multi model\emotion_model_local"

# Use a powerful multilingual emotion model (supports 100+ languages including Tamil and English)
model_name = "MilaNLProc/xlm-roberta-base-emotion"

print(f"Downloading multilingual text emotion model: {model_name}...")
print("This may take a few minutes...")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)
print(f"✓ Tokenizer saved to {save_path}")

# Download model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.save_pretrained(save_path)
print(f"✓ Model saved to {save_path}")

print("\n✓ Multilingual text model download complete!")
print(f"Model location: {save_path}")
