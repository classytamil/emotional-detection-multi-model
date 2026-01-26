"""
Multimodal Emotion Detection System
Combines Video (Face), Audio (Voice), and Text emotion detection
"""

import os
import sys

def print_header():
    print("\n" + "="*70)
    print(" " * 15 + "MULTIMODAL EMOTION DETECTION SYSTEM")
    print("="*70)
    print("\nThis system can detect emotions from:")
    print("  📹 Video (Face) - Using DeepFace")
    print("  🎤 Audio (Voice) - Using Wav2Vec2")
    print("  📝 Text - Using Transformers")
    print("="*70 + "\n")

def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    
    # Check for video dependencies
    try:
        import cv2
        from deepface import DeepFace
    except ImportError:
        missing.append("Video: pip install opencv-python deepface tf-keras")
    
    # Check for audio dependencies
    try:
        import torch
        import torchaudio
        from transformers import Wav2Vec2Processor
        import pyaudio
    except ImportError:
        missing.append("Audio: pip install torch torchaudio transformers pyaudio")
    
    # Check for text dependencies
    try:
        from transformers import pipeline
    except ImportError:
        missing.append("Text: pip install transformers")
    
    if missing:
        print("⚠️  Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nPlease install missing dependencies before proceeding.\n")
        return False
    
    return True

def run_video_emotion():
    """Run video-based emotion detection"""
    print("\n🎥 Starting Video Emotion Detection...")
    print("Press 'q' to quit\n")
    
    try:
        import cv2
        from deepface import DeepFace
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("✓ Camera opened successfully")
        print("Show your face to the camera...\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                emotion = result[0]["dominant_emotion"]
                confidence = result[0]["emotion"][emotion]
                
                # Display emotion on frame
                cv2.putText(frame, f"{emotion} ({confidence:.1f}%)", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display all emotions
                y_offset = 100
                for emo, score in result[0]["emotion"].items():
                    text = f"{emo}: {score:.1f}%"
                    cv2.putText(frame, text, (50, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 30
                    
            except Exception as e:
                cv2.putText(frame, "No face detected", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Video Emotion Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Video emotion detection stopped")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def run_audio_emotion():
    """Run audio-based emotion detection"""
    print("\n🎤 Starting Audio Emotion Detection...")
    
    try:
        from audio_emotion_detector import AudioEmotionDetector
        
        model_path = r"E:\Projects\emotion detection multi model\audio_emotion_model"
        
        if os.path.exists(model_path):
            detector = AudioEmotionDetector(model_path)
        else:
            print("⚠️  Local model not found. Using online model (slower first time)...")
            detector = AudioEmotionDetector()
        
        print("\n1. Record from microphone")
        print("2. Analyze audio file")
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            duration = input("Recording duration in seconds (default 5): ").strip()
            duration = int(duration) if duration else 5
            
            result = detector.record_and_predict(duration=duration)
        elif choice == "2":
            audio_path = input("Enter path to audio file: ").strip()
            if not os.path.exists(audio_path):
                print(f"❌ File not found: {audio_path}")
                return
            result = detector.predict_emotion(audio_path)
        else:
            print("❌ Invalid choice")
            return
        
        # Display results
        print("\n" + "="*60)
        print("AUDIO EMOTION RESULTS")
        print("="*60)
        print(f"🎯 Dominant Emotion: {result['dominant_emotion'].upper()}")
        print(f"📊 Confidence: {result['confidence']*100:.2f}%")
        print("\nAll Emotions:")
        for emotion, score in result['all_emotions'].items():
            bar = "█" * int(score * 50)
            print(f"  {emotion:12s}: {bar} {score*100:.1f}%")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_text_emotion():
    """Run text-based emotion detection (English & Tamil)"""
    print("\n📝 Starting Multilingual Text Emotion Detection...")
    
    try:
        from transformers import pipeline
        
        model_path = r"E:\Projects\emotion detection multi model\emotion_model_local"
        
        if os.path.exists(model_path):
            print(f"Loading local model from {model_path}...")
            pipe = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                top_k=None,
                truncation=True
            )
        else:
            print("⚠️  Local model not found. Using online multilingual model...")
            pipe = pipeline("text-classification", model="MilaNLProc/xlm-roberta-base-emotion", top_k=None)
        
        print("\n✓ Model loaded (Supports English, Tamil, and 100+ other languages)")
        print("\nEnter text to analyze (or 'quit' to exit):")
        
        while True:
            text = input("\n> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            result = pipe(text)
            
            # Handle results (pipeline with top_k=None returns a list of results)
            if isinstance(result[0], list):
                result = result[0]
            
            # Sort by score
            result = sorted(result, key=lambda x: x['score'], reverse=True)
            
            print("\n" + "-"*60)
            print("TEXT EMOTION RESULTS")
            print("-"*60)
            print(f"🎯 Dominant Emotion: {result[0]['label'].upper()}")
            print(f"📊 Confidence: {result[0]['score']*100:.2f}%")
            print("\nAll Emotions:")
            for item in result:
                score = item['score']
                bar = "█" * int(score * 50)
                print(f"  {item['label']:12s}: {bar} {score*100:.1f}%")
            print("-"*60)
        
        print("\n✓ Text emotion detection stopped")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def run_realtime_audio():
    """Run real-time audio emotion detection"""
    print("\n🎤 Starting Real-time Audio Emotion Detection...")
    print("This will continuously analyze your voice in real-time")
    print("Press Ctrl+C to stop\n")
    
    try:
        from realtime_audio_emotion import RealtimeAudioEmotionDetector
        
        model_path = r"E:\Projects\emotion detection multi model\audio_emotion_model"
        
        if os.path.exists(model_path):
            detector = RealtimeAudioEmotionDetector(model_path, chunk_duration=3)
        else:
            print("⚠️  Local model not found. Using online model...")
            detector = RealtimeAudioEmotionDetector(chunk_duration=3)
        
        detector.start()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main menu for multimodal emotion detection"""
    
    print_header()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    while True:
        print("\n" + "="*70)
        print("SELECT EMOTION DETECTION MODE")
        print("="*70)
        print("\n1. 📹 Video Emotion Detection (Real-time from webcam)")
        print("2. 🎤 Audio Emotion Detection (Record or analyze file)")
        print("3. 🎙️  Real-time Audio Emotion Detection (Continuous)")
        print("4. 📝 Text Emotion Detection (Type and analyze)")
        print("5. ℹ️  System Information")
        print("6. 🚪 Exit")
        print("\n" + "="*70)
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            run_video_emotion()
        elif choice == "2":
            run_audio_emotion()
        elif choice == "3":
            run_realtime_audio()
        elif choice == "4":
            run_text_emotion()
        elif choice == "5":
            print("\n" + "="*70)
            print("SYSTEM INFORMATION")
            print("="*70)
            print("\n📁 Project Directory:")
            print(f"   {os.path.dirname(os.path.abspath(__file__))}")
            
            print("\n📦 Available Models:")
            
            model_dirs = [
                ("Video Model", "deepface (online)"),
                ("Audio Model", r"E:\Projects\emotion detection multi model\audio_emotion_model"),
                ("Text Model", r"E:\Projects\emotion detection multi model\emotion_model_local"),
            ]
            
            for name, path in model_dirs:
                if path.startswith("E:"):
                    status = "✓ Installed" if os.path.exists(path) else "✗ Not found (will use online)"
                    print(f"   {name}: {status}")
                else:
                    print(f"   {name}: {path}")
            
            print("\n💡 Tips:")
            print("   - Run audio_model_download.py to download audio model locally")
            print("   - Run text_model_download.py to download text model locally")
            print("   - Video model downloads automatically on first use")
            print("="*70)
            
        elif choice == "6":
            print("\n👋 Thank you for using Multimodal Emotion Detection System!")
            print("Goodbye!\n")
            break
        else:
            print("\n❌ Invalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
        sys.exit(0)
