
from transformers import pipeline

try:
    print("Loading model...")
    # using the model name directly
    pipe = pipeline("text-classification", model="AnasAlokla/multilingual_go_emotions", top_k=5)
    
    tamil_text = "நான் மிகவும் மகிழ்ச்சியாக இருக்கிறேன்" # "I am very happy"
    print(f"Testing Tamil text: {tamil_text}")
    
    results = pipe(tamil_text)
    print("Results:", results)
    
except Exception as e:
    print(f"Error: {e}")
