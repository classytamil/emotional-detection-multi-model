"""
Download and save audio emotion recognition model locally
Using ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
This model is trained on multiple emotion datasets and works well for English speech
"""

from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import os

# Define save path
# Define save path
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_emotion_model")

# Create directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

print("Downloading Wav2Vec2 emotion recognition model...")
print("This may take a few minutes depending on your internet connection...")

# Download feature extractor (handles audio preprocessing)
# Using FeatureExtractor instead of Processor because this model doesn't have a tokenizer
from transformers import Wav2Vec2FeatureExtractor
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
feature_extractor.save_pretrained(save_path)
print(f"[DONE] Feature Extractor saved to {save_path}")

# Download model
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
model.save_pretrained(save_path)
print(f"[DONE] Model saved to {save_path}")

print("\n[DONE] Download complete! You can now use the model offline.")
print(f"Model location: {save_path}")
