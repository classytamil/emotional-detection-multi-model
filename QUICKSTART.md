# 🎭 Multimodal Emotion Detection System - Quick Start Guide

## 📦 Installation

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Windows PyAudio Fix** (if needed):
```bash
pip install pipwin
pipwin install pyaudio
```

### Step 2: Download Models (Optional but Recommended)

#### Audio Model (~1.2GB)
```bash
python audio_model_download.py
```

#### Text Model (if not already downloaded)
```bash
python text_model_download.py
```

#### Video Model
- Downloads automatically on first use via DeepFace

## 🚀 Quick Start

### Option 1: Run Complete Multimodal Demo
```bash
python multimodal_emotion_demo.py
```

This gives you a menu to choose:
- Video emotion detection (webcam)
- Audio emotion detection (mic or file)
- Real-time audio detection
- Text emotion detection

### Option 2: Run Individual Modules

#### Audio Only
```bash
python audio_emotion_detector.py
```

#### Real-time Audio
```bash
python realtime_audio_emotion.py
```

#### Video Only
```bash
python video_emotion_test.py
```

#### Text Only
```bash
python text_emotion_test.py
```

## 🎯 What Each Model Does

| Modality | Input | Output | Model Used |
|----------|-------|--------|------------|
| **Video** | Webcam feed | Face emotions | DeepFace |
| **Audio** | Voice/Audio file | Voice emotions | Wav2Vec2 |
| **Text** | Typed text | Text emotions | RoBERTa/Transformers |

## 📊 Detected Emotions

### Audio Model (8 emotions)
- Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised

### Video Model (7 emotions)
- Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### Text Model (varies by model)
- Joy, Sadness, Anger, Fear, Love, Surprise, etc.

## 🔧 Troubleshooting

### PyAudio Won't Install (Windows)
1. Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
2. Install: `pip install PyAudio‑0.2.11‑cp3XX‑cp3XX‑win_amd64.whl`

### Camera Not Working
- Check if another app is using the camera
- Try changing camera index in v_model.py: `cv2.VideoCapture(1)` instead of `0`

### Model Download Slow
- Models are large (1-2GB), be patient
- You can use online models (slower but no download needed)

### Import Errors
Make sure all dependencies are installed:
```bash
pip install torch torchaudio transformers pyaudio numpy soundfile librosa
pip install opencv-python deepface tf-keras
```

## 📁 Project Structure

```
emotion detection multi model/
├── multimodal_emotion_demo.py      # 🎯 Main unified demo
├── audio_emotion_detector.py       # Audio detection (file/mic)
├── realtime_audio_emotion.py       # Real-time audio
├── audio_model_download.py         # Download audio model
├── v_model.py                      # Video detection
├── t_model_run.py                  # Text detection
├── audio_requirements.txt          # Audio dependencies
└── AUDIO_EMOTION_README.md         # Detailed audio docs
```

## 💡 Usage Tips

1. **For best audio results**: Speak clearly for 3-5 seconds
2. **For best video results**: Good lighting, face the camera
3. **For best text results**: Use complete sentences

## 🎬 Next Steps

1. Install dependencies
2. Download models (optional)
3. Run `python multimodal_emotion_demo.py`
4. Choose your preferred mode
5. Start detecting emotions!

---

**Need more help?** Check `AUDIO_EMOTION_README.md` for detailed audio documentation.
