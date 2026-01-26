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
    from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
    AUDIO_DEPS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Audio dependencies not available: {e}")
    AUDIO_DEPS_AVAILABLE = False
    # Dummy classes/modules to prevent NameErrors before usage
    torch = None
    torchaudio = None
    Wav2Vec2Processor = None
    Wav2Vec2ForSequenceClassification = None
except OSError as e:
    print(f"Warning: Audio library load error: {e}")
    AUDIO_DEPS_AVAILABLE = False
    torch = None
    torchaudio = None
    Wav2Vec2Processor = None
    Wav2Vec2ForSequenceClassification = None

class AudioEmotionDetector:
    def __init__(self, model_path=None):
        """
        Initialize the audio emotion detector
        
        Args:
            model_path: Path to local model. If None, downloads from HuggingFace
        """
        if not AUDIO_DEPS_AVAILABLE:
            raise ImportError("Audio dependencies (torch, torchaudio, transformers) are not available or failed to load.")

        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        else:
            print("Loading model from HuggingFace (this may take time on first run)...")
            model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Emotion labels for this model
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        
        print(f"✓ Model loaded successfully on {self.device}")
    
    def predict_emotion(self, audio_path):
        """
        Predict emotion from an audio file
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            
        Returns:
            dict: Emotion predictions with scores
        """
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to 16kHz if needed (model expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to numpy
        audio_array = waveform.squeeze().numpy()
        
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
        
        print("✓ Recording finished")
        
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
        
        print(f"✓ Audio saved to {output_file}")
        
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
