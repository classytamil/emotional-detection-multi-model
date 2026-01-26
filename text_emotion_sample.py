from transformers import pipeline
import os

def run_sample():
    print("Loading Text Emotion Model...")
    model_path = os.path.join(os.getcwd(), "emotion_model_local")
    try:
        if os.path.exists(model_path):
            print(f"Using local model at {model_path}")
            classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, top_k=None, truncation=True)
        else:
            print("Using online model 'MilaNLProc/xlm-roberta-base-emotion'")
            classifier = pipeline("text-classification", model="MilaNLProc/xlm-roberta-base-emotion", top_k=None)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    samples = [
        "I am so happy today! This is the best day ever.",
        "I feel very sad and lonely, I don't know what to do.",
        "That movie was absolutely disgusting and terrible.",
        "I am afraid of what might happen next."
    ]

    print("\n--- Sample Predictions ---\n")

    for text in samples:
        print(f"Input: {text}")
        try:
            results = classifier(text)
            # Handle list of lists
            if isinstance(results[0], list):
                results = results[0]
            
            # Sort by score
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            top = results[0]
            print(f"Dominant: {top['label'].upper()} ({top['score']:.4f})")
            print("All scores:")
            for r in results:
                print(f"  - {r['label']}: {r['score']:.4f}")
            print("-" * 30)
        except Exception as e:
            print(f"Error processing text: {e}")

if __name__ == "__main__":
    run_sample()
