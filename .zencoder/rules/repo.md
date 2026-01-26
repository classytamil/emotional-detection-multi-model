---
description: Repository Information Overview
alwaysApply: true
---

# Multimodal Emotion Detection System Information

## Summary
A comprehensive multimodal system for detecting emotions from video (facial expressions), audio (voice), and text. It integrates multiple state-of-the-art models including Wav2Vec2 for audio and DeepFace for video to provide a holistic emotion analysis across different input formats.

## Structure
- **Root**: Contains the main entry point and individual modality scripts.
- **emotion_model_local/**: Storage for local text-classification models.
- **.venv/**: Python virtual environment containing project-specific packages and configuration.

## Language & Runtime
**Language**: Python  
**Version**: 3.10.9  
**Build System**: Pip  
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- **transformers**: For text classification and speech emotion recognition (Wav2Vec2).
- **torch & torchaudio**: Deep learning framework for model inference.
- **deepface**: For video-based facial emotion detection.
- **opencv-python**: For webcam feed handling and image processing.
- **tf-keras & tensorflow**: Backend for DeepFace and other models.
- **pyaudio & soundfile & librosa**: For audio recording and processing.
- **numpy**: For numerical operations.

## Build & Installation
```bash
# Install audio dependencies
pip install -r audio_requirements.txt

# Install video and general dependencies
pip install tf-keras deepface opencv-python

# Download audio model (optional but recommended)
python audio_model_download.py

# Save/Download text model
python t_model_save.py
```

## Main Files & Resources
- **multimodal_emotion_demo.py**: The central entry point providing a unified interface for all detection modules.
- **audio_emotion_detector.py**: Script for detecting emotions from audio files or microphone input.
- **realtime_audio_emotion.py**: Dedicated module for continuous real-time audio emotion analysis.
- **v_model.py**: Video-based emotion detection using the webcam.
- **t_model_run.py**: Text-based emotion classification script.
- **QUICKSTART.md**: Comprehensive guide for installation and usage.

## Usage & Operations
**Run Multimodal Demo**:
```bash
python multimodal_emotion_demo.py
```

**Run Individual Modules**:
```bash
# Audio Emotion Detection
python audio_emotion_detector.py

# Real-time Audio Detection
python realtime_audio_emotion.py

# Video Emotion Detection
python v_model.py

# Text Emotion Detection
python t_model_run.py
```
