# Multimodal Emotion Recognition System
## Final Year Project Report

---

### 1. Abstract

Evaluation of human emotion is a complex task that relies on multiple cues. Traditional emotion recognition systems often focus on a single modality, such as facial expressions or speech, which can lead to ambiguity and lower accuracy. This project presents a **Multimodal Emotion Recognition System** that integrates three distinct modalities: **Audio (Speech)**, **Video (Facial Expressions)**, and **Text (Linguistic Content)**. By combining these inputs, the system provides a more robust and holistic determination of the user's emotional state. The application is built using **Streamlit** for the user interface, leveraging state-of-the-art Deep Learning models including **Wav2Vec2** for audio, **DeepFace** for video, and **XLM-RoBERTa** for multilingual text analysis.

---

### 2. Introduction

#### 2.1 Problem Statement
Human communication is multimodal, involving words, tone of voice, and facial expressions. Detecting emotion from a single source often fails in real-world scenarios. For instance, a sarcastic "I'm fine" (text) might be spoken with a sad tone (audio) and a distressed face (video). Single-modality models fail to capture this nuance.

#### 2.2 Objective
The primary objective of this project is to design and implement a comprehensive system that can:
1.  Detect emotions from live video streams or webcams in real-time.
2.  Analyze audio recordings or live microphone input for vocal emotional cues.
3.  Process text input in multiple languages to classify emotional sentiment.
4.  Provide a unified, user-friendly interface for interacting with all three modalities.

#### 2.3 Scope
The project is implemented as a web-based application utilizing Python's rich ecosystem of AI libraries. It is designed to be extensible, allowing for future integration of additional models or fusion techniques.

---

### 3. Literature Survey / Theoretical Background

The system utilizes pre-trained transformer and deep learning models to achieve high accuracy without the need for extensive training from scratch.

*   **Audio Emotion Recognition**: utilizes **Wav2Vec2** (specifically `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`), a transformer-based model pretrained on 53 languages and fine-tuned on emotion datasets like RAVDESS and CREMA-D.
*   **Facial Emotion Recognition**: employs **DeepFace**, a lightweight hybrid framework wrapping state-of-the-art face recognition models (VGG-Face, Google FaceNet, etc.) to analyze facial attributes including emotion.
*   **Text Emotion Recognition**: leverages **XLM-RoBERTa**, a multilingual transformer model. To enhance performance on colloquial and code-mixed text, the model was fine-tuned using the **TamilEmo** dataset (comprising YouTube comments) and the **WA Dataset** (WhatsApp conversational data), enabling robust sentiment analysis in both English and Tamil.

---

### 4. System Architecture

The system follows a modular architecture where each modality is handled by a dedicated processing pipeline. The central controller is the Streamlit application, which routes user input to the appropriate model and aggregates the results.

#### 4.1 Modules
1.  **Video Module**:
    *   **Input**: Real-time webcam feed via WebRTC.
    *   **Process**: Frame extraction -> Face Detection -> Emotion Classification (DeepFace).
    *   **Output**: Live video feed with emotion labels and confidence scores overlayed.

2.  **Audio Module**:
    *   **Input**: WAV/MP3 file upload or live microphone recording.
    *   **Process**: Resampling (16kHz) -> Feature Extraction (Wav2Vec2 Processor) -> Inference.
    *   **Output**: Dominant emotion (e.g., Angry, Happy, Neutral) and probability distribution.

3.  **Text Module**:
    *   **Input**: User-typed text (multilingual support).
    *   **Process**: Tokenization -> Transformer Pipeline (Fine-tuned on TamilEmo & WA Dataset) -> Classification.
    *   **Output**: Predicted sentiment with confidence score.

---

### 5. Implementation Details

#### 5.1 Technology Stack
*   **Language**: Python 3.8+
*   **Frontend Framework**: Streamlit
*   **Computer Vision**: OpenCV, Streamlit-WebRTC
*   **Deep Learning Frameworks**: PyTorch, TensorFlow/Keras
*   **Audio Processing**: Librosa, PyAudio, SoundFile
*   **Model Libraries**: HuggingFace Transformers, DeepFace

#### 5.2 Key Algorithms & Models
*   **Wav2Vec 2.0**: Used for mapping raw audio waveforms to speech representations.
*   **Convolutional Neural Networks (CNNs)**: Implicitly used within DeepFace for feature extraction from facial images.
*   **Transformers**: Utilized for both audio (Wav2Vec2) and text (XLM-R) understanding, utilizing self-attention mechanisms to capture context.

#### 5.3 Code Structure
*   `app.py`: Main entry point and UI logic.
*   `audio_emotion_detector.py`: Class for handling audio model loading and inference.
*   `realtime_audio_emotion.py`: Helper for continuous audio stream processing.
*   `requirements.txt`: Dependency management.

---

### 6. Results and Discussion

The system successfully demonstrates the capability to identify emotions across all three modalities.
*   **Video**: Real-time performance is optimized by processing every Nth frame, achieving smooth playback while maintaining detection accuracy.
*   **Audio**: The Wav2Vec2 model shows high robustness to background noise and successfully identifies subtle emotional changes in tone.
*   **Text**: The multilingual model, fine-tuned on **TamilEmo** and **WA Dataset**, demonstrated superior performance in classifying emotions in colloquial Tamil and code-mixed social media text compared to the base model.

---

### 7. Conclusion & Future Scope

This project successfully integrates Audio, Video, and Text modalities into a single analytical platform. It highlights the power of modern Transfer Learning to build complex AI systems with limited computational resources.

**Future Scope**:
*   **Multimodal Fusion**: Implement a "Decision Level Fusion" or "Feature Level Fusion" algorithm to give a single weighted emotion score based on all three inputs simultaneously.
*   **Temporal Analysis**: Analyze emotions over time to detect mood swings or long-term behavioral patterns.
*   **Edge Deployment**: Optimize models for mobile or edge devices.
