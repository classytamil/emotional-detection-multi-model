from transformers import pipeline

def test_new_model():
    print("Loading Happy/Sad Model...")
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    
    samples = [
        "I am so happy today!",
        "I feel very sad and lonely.",
        "I am angry about this!",
        "Wow, that is a surprise."
    ]
    
    for text in samples:
        results = classifier(text)
        if isinstance(results[0], list): results = results[0]
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        top = results[0]
        print(f"'{text}' -> {top['label']} ({top['score']:.4f})")

if __name__ == "__main__":
    test_new_model()
