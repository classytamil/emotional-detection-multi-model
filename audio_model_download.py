"""
Download and save audio emotion recognition model locally
Using ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
This model is trained on multiple emotion datasets and works well for English speech
"""

from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import os

# Define save path
save_path = r"E:\Projects\emotion detection multi model\audio_emotion_model"

# Create directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

print("Downloading Wav2Vec2 emotion recognition model...")
print("This may take a few minutes depending on your internet connection...")

# Download processor (handles audio preprocessing)
processor = Wav2Vec2Processor.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
processor.save_pretrained(save_path)
print(f"✓ Processor saved to {save_path}")

# Download model
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
model.save_pretrained(save_path)
print(f"✓ Model saved to {save_path}")

print("\n✓ Download complete! You can now use the model offline.")
print(f"Model location: {save_path}")
