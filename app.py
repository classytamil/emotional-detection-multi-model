import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
import av
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
except ImportError:
    st.error("streamlit-webrtc is not installed. Please install it with `pip install streamlit-webrtc`")

# Append current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deepface import DeepFace
# Safely import AudioEmotionDetector handling potential missing libs
AudioEmotionDetector = None
try:
    from audio_emotion_detector import AudioEmotionDetector
except ImportError as e:
    st.error(f"Error importing audio module: {e}")

from transformers import pipeline

# --- Page Config ---
st.set_page_config(
    page_title="Multimodal Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 100%;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
    .css-1d391kg {
        padding-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/clouds/100/000000/emotion.png", width=100)
st.sidebar.title("Navigation")
st.sidebar.info("Select an input modality below:")

mode = st.sidebar.radio(
    "Choose Mode:",
    ("📹 Video (WebRTC)", "🎤 Audio Analysis", "📝 Text Analysis")
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Info")
st.sidebar.text("Multimodal One-shot Agent")
st.sidebar.text("v1.0.0")

# --- Helper Functions (Cached) ---

@st.cache_resource
def load_audio_model():
    if AudioEmotionDetector is None:
        return None
    model_path = os.path.join(os.getcwd(), "audio_emotion_model")
    if os.path.exists(model_path):
        return AudioEmotionDetector(model_path)
    return AudioEmotionDetector() 

@st.cache_resource
def load_text_pipeline():
    # Use a specific model that gives happy/sad/etc emotions
    # j-hartmann/emotion-english-distilroberta-base supports: anger, disgust, fear, joy, neutral, sadness, surprise
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)

# --- Video Processor ---

class EmotionVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_emotions = None

    def recv(self, frame_param):
        frame = frame_param.to_ndarray(format="bgr24")
        
        # Analyze every 10th frame to improve performance
        self.frame_count += 1
        if self.frame_count % 10 == 0 or self.last_emotions is None:
            try:
                # DeepFace.analyze is heavy, usually better to run in a separate thread or use a lighter model for real-time
                # For this demo, we try to run it directly but catch exceptions
                # To optimize: use a smaller face detector or skip frames rigorously
                
                # Using enforces_detection=False to avoid crash if no face
                result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False, detector_backend='opencv')
                
                if result:
                     # Taking the first face found
                    face_data = result[0]
                    self.last_emotions = face_data
            except Exception as e:
                # Pass if analysis fails (e.g. no face detected significantly)
                pass

        
        # Draw results if available
        if self.last_emotions:
            dom_emotion = self.last_emotions['dominant_emotion']
            confidence = self.last_emotions['emotion'][dom_emotion]
            
            # Draw text
            cv2.putText(frame, f"{dom_emotion.upper()} ({confidence:.1f}%)", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw bars for other emotions
            y_offset = 90
            for emo, score in self.last_emotions['emotion'].items():
                if score > 1.0: # Only show significant ones
                    text = f"{emo}: {score:.1f}%"
                    cv2.putText(frame, text, (30, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_offset += 25

        return av.VideoFrame.from_ndarray(frame, format="bgr24")

# --- Main App Logic ---

st.title("🎭 Multimodal Emotion Detection")
st.markdown("### Detect emotions from Video, Audio, and Text")

if mode == "📹 Video (WebRTC)":
    st.header("Real-time Video Emotion Detection")
    st.write("Use your webcam to detect emotions in real-time. This uses WebRTC and DeepFace.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # WebRTC Streamer
        try:
            webrtc_streamer(
                key="emotion-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                video_processor_factory=EmotionVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        except Exception as e:
            st.error(f"Error initializing WebRTC: {e}")
            st.info("Ensure you are running this in a compatible browser and environment (https usually required for remote access).")

    with col2:
        st.info("💡 **Instructions**")
        st.markdown("""
        1. Click 'START' to access camera.
        2. Allow browser camera permissions.
        3. Look at the camera.
        4. See your detected emotion!
        """)

elif mode == "🎤 Audio Analysis":
    st.header("Audio Emotion Detection")
    st.write("Upload an audio file or record voice to detect emotions.")

    # Load audio model
    with st.spinner("Loading Audio Model..."):
        try:
            audio_detector = load_audio_model()
            st.success("Audio Model Loaded")
        except Exception as e:
            st.error(f"Failed to load audio model: {e}")
            audio_detector = None

    if audio_detector:
        tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])
        
        with tab1:
            uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])
            if uploaded_file is not None:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                st.audio(uploaded_file, format='audio/wav')
                
                if st.button("Analyze Uploaded Audio"):
                    with st.spinner("Analyzing audio..."):
                        try:
                            results = audio_detector.predict_emotion(tmp_path)
                            
                            # Display results
                            col_res1, col_res2 = st.columns(2)
                            with col_res1:
                                st.metric("Dominant Emotion", results['dominant_emotion'].upper(), f"{results['confidence']*100:.1f}%")
                            
                            with col_res2:
                                st.write("All Emotions:")
                                for emo, score in results['all_emotions'].items():
                                    st.progress(score, text=f"{emo} ({score*100:.1f}%)")
                                    
                        except Exception as e:
                            st.error(f"Error processing audio: {e}")
                        finally:
                            # Cleanup
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)

        with tab2:
            st.write("Experimental: This requires browser support for media recording.")
            # Simple alternative: "For recording, please use system recorder and upload in the previous tab for best stability."
            # OR integration with st_audiorec if installed.
            # Using st.audio_input if streamlit version supports it (new feature) or leaving as placeholder since user asked "use tem to store".
            
            st.info("Please record audio using your device functionalities, save it, and upload it in the 'Upload Audio' tab.")
    else:
        st.warning("⚠️ Audio emotion detection is currently unavailable.")
        st.info("This is likely due to missing dependencies (torch, torchaudio) or a library loading error. Please check the terminal logs.")
        st.write("You can try restarting the application to resolve library loading issues.")


elif mode == "📝 Text Analysis":
    st.header("Text Emotion Detection")
    st.write("Type existing text to analyze its sentiment/emotion.")
    
    # Load Text Model
    with st.spinner("Loading Text Model..."):
        try:
            text_pipe = load_text_pipeline()
        except Exception as e:
            st.error(f"Failed to load text pipeline: {e}")
            text_pipe = None

    if text_pipe:
        user_text = st.text_area("Enter text here:", height=150, placeholder="I am feeling great today!")
        
        if st.button("Analyze Text"):
            if user_text.strip():
                with st.spinner("Analyzing..."):
                    try:
                        results = text_pipe(user_text)
                        # Handle list of lists or list of dicts depending on pipeline version
                        if isinstance(results[0], list):
                            results = results[0]
                        
                        # Sort
                        results = sorted(results, key=lambda x: x['score'], reverse=True)
                        
                        top_emotion = results[0]['label']
                        top_score = results[0]['score']
                        
                        st.subheader(f"Result: {top_emotion.upper()}")
                        
                        # Charts
                        chart_data = {item['label']: item['score'] for item in results}
                        st.bar_chart(chart_data)
                        
                    except Exception as e:
                        st.error(f"Error processing text: {e}")
            else:
                st.warning("Please enter some text first.")
