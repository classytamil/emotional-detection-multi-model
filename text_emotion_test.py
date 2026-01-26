# t_model_run.py
from transformers import pipeline
import os

# Path to the local model
model_path = "./emotion_model_local"

# Check if local model exists, else use online
if os.path.exists(model_path):
    print(f"Loading local model from {model_path}...")
    pipe = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        top_k=None,  # Return all emotions
        truncation=True
    )
else:
    print("⚠️ Local model not found. Using online model (MilaNLProc/xlm-roberta-base-emotion)...")
    pipe = pipeline(
        "text-classification", 
        model="MilaNLProc/xlm-roberta-base-emotion", 
        top_k=None
    )

def analyze_text(text):
    print(f"\nAnalyzing: '{text}'")
    results = pipe(text)[0]
    
    # Sort results by score
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    
    print(f"🎯 Dominant Emotion: {results[0]['label'].upper()}")
    print(f"📊 Confidence: {results[0]['score']*100:.2f}%")
    print("\nAll Emotions:")
    for res in results:
        bar = "█" * int(res['score'] * 30)
        print(f"  {res['label']:12s}: {bar} {res['score']*100:.1f}%")

# Test with English
analyze_text("I am very happy today!")

# Test with Tamil
analyze_text("நான் இன்று ரொம்ப கோபமாக இருக்கேன்!")
