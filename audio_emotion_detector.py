"""
Audio Emotion Detection from Voice
Supports both real-time microphone input and audio file analysis
"""

import os
from datetime import datetime
import numpy as np
import wave
import pyaudio


try:
    import torch
    import torchaudio
    import librosa
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
    AUDIO_DEPS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Audio dependencies not available: {e}")
    AUDIO_DEPS_AVAILABLE = False
    # Dummy classes/modules to prevent NameErrors before usage
    torch = None
    torchaudio = None
    librosa = None
    Wav2Vec2FeatureExtractor = None
    Wav2Vec2ForSequenceClassification = None
except OSError as e:
    print(f"Warning: Audio library load error: {e}")
    AUDIO_DEPS_AVAILABLE = False
    torch = None
    torchaudio = None
    librosa = None
    Wav2Vec2FeatureExtractor = None
    Wav2Vec2ForSequenceClassification = None

class AudioEmotionDetector:
    def __init__(self, model_path=None):
        """
        Initialize the audio emotion detector
        
        Args:
            model_path: Path to local model. If None, downloads from HuggingFace
        """
        if not AUDIO_DEPS_AVAILABLE:
            raise ImportError("Audio dependencies (torch, torchaudio, transformers, librosa) are not available or failed to load.")

        if model_path and os.path.exists(model_path):
            # Verify it's a valid model directory
            try:
                if os.path.exists(os.path.join(model_path, "config.json")):
                    print(f"Loading model from {model_path}...")
                    self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
                    self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
                else:
                    raise ValueError("Missing config.json")
            except Exception as e:
                print(f"Warning: Failed to load local model from {model_path}: {e}")
                print("Falling back to online model...")
                try:
                    model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                    self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                    self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
                except Exception as online_error:
                    import traceback
                    traceback.print_exc()
                    raise ImportError(f"Online fallback failed: {online_error}")
        else:
            print("Loading model from HuggingFace (this may take time on first run)...")
            model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Emotion labels for this model
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
        print(f"[DONE] Model loaded successfully on {self.device}")
    
    def predict_emotion(self, audio_path):
        """
        Predict emotion from an audio file
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            
        Returns:
            dict: Emotion predictions with scores
        """
        # Load audio file using librosa (more robust than torchaudio for various formats)
        # librosa automatically handles resampling to 16kHz
        try:
            audio_array, sample_rate = librosa.load(audio_path, sr=16000)
        except Exception as e:
            raise ValueError(f"Error loading audio file: {e}")
        
        # Process audio
        inputs = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        # Convert to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]
        
        # Create results dictionary
        results = {
            emotion: float(prob) 
            for emotion, prob in zip(self.emotions, probs)
        }
        
        # Sort by probability
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        dominant_emotion = list(results.keys())[0]
        confidence = list(results.values())[0]
        
        return {
            'dominant_emotion': dominant_emotion,
            'confidence': confidence,
            'all_emotions': results
        }
    
    def record_and_predict(self, duration=5, output_file=None):
        """
        Record audio from microphone and predict emotion
        
        Args:
            duration: Recording duration in seconds
            output_file: Optional path to save the recording
            
        Returns:
            dict: Emotion predictions
        """
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        if output_file is None:
            output_file = f"temp_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        p = pyaudio.PyAudio()
        
        print(f"🎤 Recording for {duration} seconds...")
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        frames = []
        
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        print("[DONE] Recording finished")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save recording
        wf = wave.open(output_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        print(f"[DONE] Audio saved to {output_file}")
        
        # Predict emotion
        result = self.predict_emotion(output_file)
        
        return result


def main():
    """Demo usage of the audio emotion detector"""
    
    # Initialize detector (use local model if available)
    model_path = r"E:\Projects\emotion detection multi model\audio_emotion_model"
    
    if os.path.exists(model_path):
        detector = AudioEmotionDetector(model_path)
    else:
        print("Local model not found. Using online model...")
        print("Tip: Run audio_model_download.py first to download the model locally")
        detector = AudioEmotionDetector()
    
    print("\n" + "="*60)
    print("AUDIO EMOTION DETECTION DEMO")
    print("="*60)
    
    while True:
        print("\nOptions:")
        print("1. Record from microphone and detect emotion")
        print("2. Analyze an audio file")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            try:
                duration = input("Enter recording duration in seconds (default 5): ").strip()
                duration = int(duration) if duration else 5
                
                result = detector.record_and_predict(duration=duration)
                
                print("\n" + "="*60)
                print("EMOTION DETECTION RESULTS")
                print("="*60)
                print(f"🎯 Dominant Emotion: {result['dominant_emotion'].upper()}")
                print(f"📊 Confidence: {result['confidence']*100:.2f}%")
                print("\nAll Emotions:")
                for emotion, score in result['all_emotions'].items():
                    bar = "█" * int(score * 50)
                    print(f"  {emotion:12s}: {bar} {score*100:.1f}%")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "2":
            audio_path = input("Enter path to audio file: ").strip()
            
            if not os.path.exists(audio_path):
                print(f"❌ File not found: {audio_path}")
                continue
            
            try:
                result = detector.predict_emotion(audio_path)
                
                print("\n" + "="*60)
                print("EMOTION DETECTION RESULTS")
                print("="*60)
                print(f"🎯 Dominant Emotion: {result['dominant_emotion'].upper()}")
                print(f"📊 Confidence: {result['confidence']*100:.2f}%")
                print("\nAll Emotions:")
                for emotion, score in result['all_emotions'].items():
                    bar = "█" * int(score * 50)
                    print(f"  {emotion:12s}: {bar} {score*100:.1f}%")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "3":
            print("Goodbye! 👋")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
