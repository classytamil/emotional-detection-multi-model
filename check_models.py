
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys

def check_model(model_name):
    print(f"Checking {model_name}...")
    try:
        AutoTokenizer.from_pretrained(model_name)
        print(f"Tokenier found for {model_name}")
        return True
    except Exception as e:
        print(f"Failed {model_name}: {e}")
        return False

# Candidates
models = [
    "MilaNLProc/xlm-emo", 
    "MilaNLProc/xlm-roberta-base-emotion",
    "seara/roberta-base-german-emotion", # Just a test
    "nateraw/bert-base-uncased-emotion",
    "AnasAlokla/multilingual_go_emotions",
    "MilaNLProc/xlm-emo-t"
]

for m in models:
    check_model(m)
