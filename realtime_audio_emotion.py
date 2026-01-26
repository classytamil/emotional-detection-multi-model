"""
Real-time Audio Emotion Detection
Continuously records audio in chunks and detects emotions in real-time
"""

import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import pyaudio
import wave
import os
import threading
import queue
from datetime import datetime

class RealtimeAudioEmotionDetector:
    def __init__(self, model_path=None, chunk_duration=3):
        """
        Initialize real-time audio emotion detector
        
        Args:
            model_path: Path to local model
            chunk_duration: Duration of audio chunks to analyze (seconds)
        """
        if model_path and os.path.exists(model_path):
            print(f"Loading model from {model_path}...")
            self.processor = Wav2Vec2Processor.from_pretrained(model_path)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        else:
            print("Loading model from HuggingFace...")
            model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
        self.chunk_duration = chunk_duration
        self.is_running = False
        self.audio_queue = queue.Queue()
        
        print(f"✓ Model loaded on {self.device}")
    
    def predict_from_array(self, audio_array, sample_rate=16000):
        """Predict emotion from audio numpy array"""
        
        # Process audio
        inputs = self.processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]
        
        results = {
            emotion: float(prob) 
            for emotion, prob in zip(self.emotions, probs)
        }
        
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return {
            'dominant_emotion': list(results.keys())[0],
            'confidence': list(results.values())[0],
            'all_emotions': results
        }
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def process_audio_stream(self):
        """Process audio chunks from the queue"""
        RATE = 16000
        CHUNK = 1024
        frames_per_chunk = int(RATE * self.chunk_duration)
        
        buffer = []
        
        while self.is_running:
            try:
                # Get audio data from queue
                data = self.audio_queue.get(timeout=1)
                buffer.append(np.frombuffer(data, dtype=np.int16))
                
                # Check if we have enough data
                if len(buffer) * CHUNK >= frames_per_chunk:
                    # Combine buffer
                    audio_array = np.concatenate(buffer)
                    
                    # Normalize to [-1, 1]
                    audio_array = audio_array.astype(np.float32) / 32768.0
                    
                    # Predict emotion
                    result = self.predict_from_array(audio_array, RATE)
                    
                    # Display results
                    self.display_results(result)
                    
                    # Clear buffer
                    buffer = []
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def display_results(self, result):
        """Display emotion detection results"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("="*70)
        print("REAL-TIME AUDIO EMOTION DETECTION")
        print("="*70)
        print(f"\n🎯 Dominant Emotion: {result['dominant_emotion'].upper()}")
        print(f"📊 Confidence: {result['confidence']*100:.2f}%")
        print("\nEmotion Distribution:")
        print("-"*70)
        
        for emotion, score in result['all_emotions'].items():
            bar_length = int(score * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"{emotion:12s}: {bar} {score*100:.1f}%")
        
        print("-"*70)
        print("Press Ctrl+C to stop")
        print("="*70)
    
    def start(self):
        """Start real-time emotion detection"""
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        self.is_running = True
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_audio_stream)
        process_thread.daemon = True
        process_thread.start()
        
        # Start audio stream
        p = pyaudio.PyAudio()
        
        print(f"\n🎤 Starting real-time emotion detection...")
        print(f"Analyzing audio in {self.chunk_duration}-second chunks")
        print("Speak into your microphone...")
        print("\n")
        
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self.audio_callback
        )
        
        stream.start_stream()
        
        try:
            while stream.is_active() and self.is_running:
                import time
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopping...")
        
        self.is_running = False
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        print("✓ Real-time detection stopped")


def main():
    """Run real-time audio emotion detection"""
    
    model_path = r"E:\Projects\emotion detection multi model\audio_emotion_model"
    
    if os.path.exists(model_path):
        detector = RealtimeAudioEmotionDetector(model_path, chunk_duration=3)
    else:
        print("Local model not found. Using online model...")
        print("Tip: Run audio_model_download.py first for faster loading\n")
        detector = RealtimeAudioEmotionDetector(chunk_duration=3)
    
    detector.start()


if __name__ == "__main__":
    main()
