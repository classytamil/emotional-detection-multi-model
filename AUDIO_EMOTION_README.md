# Audio Emotion Detection System

This system detects emotions from voice/audio using the **Wav2Vec2** model, a state-of-the-art open-source speech emotion recognition model.

## 🎯 Features

- **Real-time emotion detection** from microphone input
- **Audio file analysis** (supports WAV, MP3, and other formats)
- **8 emotion categories**: angry, calm, disgust, fearful, happy, neutral, sad, surprised
- **Offline capability** - download model once and use without internet
- **High accuracy** using transformer-based architecture

## 📋 Model Information

**Model**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- Based on Facebook's Wav2Vec2 architecture
- Trained on multiple emotion datasets (RAVDESS, CREMA-D, TESS, SAVEE)
- Optimized for English speech
- ~1.2GB model size

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r audio_requirements.txt
```

**Note for Windows users**: If you encounter issues installing `pyaudio`, use:
```bash
pip install pipwin
pipwin install pyaudio
```

Or download the wheel file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio)

### Step 2: Download the Model (Optional but Recommended)

Download the model locally for faster loading and offline use:

```bash
python audio_model_download.py
```

This will download the model to `audio_emotion_model/` folder (~1.2GB).

### Step 3: Run Emotion Detection

#### Option A: Interactive Demo
```bash
python audio_emotion_detector.py
```

This provides a menu with options to:
1. Record from microphone and detect emotion
2. Analyze an existing audio file
3. Exit

#### Option B: Real-time Continuous Detection
```bash
python realtime_audio_emotion.py
```

This continuously analyzes your microphone input and displays emotions in real-time.

## 📝 Usage Examples

### Example 1: Analyze an Audio File

```python
from audio_emotion_detector import AudioEmotionDetector

# Initialize detector
detector = AudioEmotionDetector(model_path="audio_emotion_model")

# Analyze audio file
result = detector.predict_emotion("my_audio.wav")

print(f"Emotion: {result['dominant_emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"All emotions: {result['all_emotions']}")
```

### Example 2: Record and Analyze

```python
from audio_emotion_detector import AudioEmotionDetector

detector = AudioEmotionDetector(model_path="audio_emotion_model")

# Record 5 seconds and analyze
result = detector.record_and_predict(duration=5)

print(f"Detected emotion: {result['dominant_emotion']}")
```

### Example 3: Real-time Detection

```python
from realtime_audio_emotion import RealtimeAudioEmotionDetector

# Initialize with 3-second chunks
detector = RealtimeAudioEmotionDetector(
    model_path="audio_emotion_model",
    chunk_duration=3
)

# Start real-time detection (press Ctrl+C to stop)
detector.start()
```

## 🎭 Detected Emotions

The model can detect 8 different emotions:

1. **Angry** - Irritation, frustration, rage
2. **Calm** - Peaceful, relaxed state
3. **Disgust** - Revulsion, distaste
4. **Fearful** - Anxiety, worry, fear
5. **Happy** - Joy, excitement, pleasure
6. **Neutral** - No strong emotion
7. **Sad** - Sorrow, grief, melancholy
8. **Surprised** - Shock, astonishment

## 🔧 Troubleshooting

### PyAudio Installation Issues (Windows)

If `pip install pyaudio` fails:

1. **Method 1**: Use pipwin
   ```bash
   pip install pipwin
   pipwin install pyaudio
   ```

2. **Method 2**: Download wheel file
   - Visit https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
   - Download the appropriate `.whl` file for your Python version
   - Install: `pip install PyAudio‑0.2.11‑cp310‑cp310‑win_amd64.whl`

### Model Download Issues

If the model download fails or is slow:
- Check your internet connection
- The model is ~1.2GB, so it may take time
- You can use the model without downloading (it will download automatically on first use)

### Low Accuracy

For best results:
- Use clear audio with minimal background noise
- Speak naturally and expressively
- Ensure microphone is working properly
- Use at least 3-5 seconds of audio for better accuracy

## 📊 Performance

- **Inference time**: ~0.5-2 seconds per 3-second audio clip (CPU)
- **GPU acceleration**: Automatically uses CUDA if available
- **Memory usage**: ~2-3GB RAM with model loaded

## 🔄 Alternative Models

If you want to try different models, here are some alternatives:

1. **HuBERT Large** (already in your t_model_save.py):
   ```python
   model_name = "superb/hubert-large-superb-er"
   ```

2. **Wav2Vec2 Base**:
   ```python
   model_name = "facebook/wav2vec2-base"
   ```

3. **Emotion2Vec**:
   ```python
   model_name = "emotion2vec-base-finetuned"
   ```

## 📁 Project Structure

```
emotion detection multi model/
├── audio_emotion_detector.py      # Main detector with file & mic support
├── realtime_audio_emotion.py      # Real-time continuous detection
├── audio_model_download.py        # Download model locally
├── audio_requirements.txt         # Python dependencies
├── audio_emotion_model/           # Downloaded model (after running download script)
└── AUDIO_EMOTION_README.md        # This file
```

## 🤝 Integration with Your Multimodal System

You can combine this with your existing models:
- **Video emotion** (v_model.py) - Face-based emotion from DeepFace
- **Text emotion** (t_model_*.py) - Text-based emotion from transformers
- **Audio emotion** (NEW!) - Voice-based emotion from Wav2Vec2

Create a multimodal system that analyzes all three modalities simultaneously!

## 📚 References

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [Model on HuggingFace](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## 📄 License

This implementation uses open-source models and libraries:
- Wav2Vec2: Apache 2.0 License
- Transformers: Apache 2.0 License
- PyTorch: BSD License

---

**Need help?** Check the troubleshooting section or open an issue!
