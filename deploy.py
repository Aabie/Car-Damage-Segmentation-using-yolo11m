import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import time
import av

# Load YOLO model
model = YOLO(r"best.pt")

# Enhanced Modern CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    }
    
    /* Main container enhancements */
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    /* Modern header design */
    .header {
        background: linear-gradient(135deg, #4158D0, #C850C0);
        border-radius: 20px;
        padding: 1rem 1rem;
        text-align: center;
        margin-bottom: 0rem;
        box-shadow: 0 10px 30px rgba(65, 88, 208, 0.2);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    /* Enhanced card design */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 0rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }

    .glass-card h3 {
        color: #4158D0;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .glass-card p {
        color: #6c757d;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4158D0, #C850C0);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        border: none;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(65, 88, 208, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(65, 88, 208, 0.3);
    }
    
    /* Image container enhancements */
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
    }
    
    /* Status messages */
    .status {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
        text-align: center;
    }
    
    .success {
        background: linear-gradient(135deg, #84fab0, #8fd3f4);
        color: #05445E;
    }
    
    .error {
        background: linear-gradient(135deg, #ff9a9e, #fad0c4);
        color: #c92a2a;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(65, 88, 208, 0.1);
    }
    
    .sidebar-card {
        background: linear-gradient(135deg, #f6f9ff, #f8f9fa);
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid rgba(65, 88, 208, 0.1);
    }
    
    /* Footer design */
    .footer {
        background: linear-gradient(135deg, #4158D0, #C850C0);
        padding: 0.2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-top: 0.5rem;
        box-shadow: 0 10px 30px rgba(65, 88, 208, 0.2);
    }
    
    /* Upload area enhancement */
    .uploadfile {
        border: 2px dashed #4158D0;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: rgba(65, 88, 208, 0.05);
    }
    
    /* Progress bar */
    .progress-bar {
        width: 100%;
        height: 6px;
        background: #e9ecef;
        border-radius: 3px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-value {
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, #4158D0, #C850C0);
        animation: progress 1s ease infinite;
    }
    
    @keyframes progress {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .header-subtitle {
            font-size: 1rem;
        }
        
        .glass-card {
            padding: 1.5rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Enhanced Header
st.markdown("""
    <div class="header">
        <h1 class="header-title">🚗 Car Damage Detective</h1>
        <p class="header-subtitle">Advanced AI-Powered Vehicle Damage Analysis System</p>
    </div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='color: #4158D0; font-weight: 600;'>⚙️ Control Center</h2>
        </div>
    """, unsafe_allow_html=True)
    
    option = st.selectbox(
        "Select Input Method",
        ("Upload Image", "Live Camera", "Image Link"),
    )
    
    confidence_threshold = st.slider(
        "Detection Sensitivity",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='sidebar-card'>
            <h4 style='color: #4158D0; margin-bottom: 1rem;'>💡 Pro Tips</h4>
            <ul style='color: #6c757d;'>
                <li>Higher sensitivity = More precise detection (but harder to detect) </li>
                <li>Ensure good lighting conditions</li>
                <li>Keep camera steady for best results</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Processing functions remain the same
def process_frame(frame):
    results = model.predict(source=frame, task="segment", stream=True, conf=confidence_threshold, show_boxes=False)
    for result in results:
        annotated_frame = result.plot()
    return annotated_frame

def save_image(image, file_name):
    output_dir = 'segmented_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, file_name)
    Image.fromarray(image).save(file_path)
    return file_path

# Upload Image Option
if option == "Upload Image":
    st.markdown("""
        <div class="glass-card">
            <h3>📤 Upload Vehicle Image</h3>
            <p>Drop your image here or click to browse</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        with st.spinner('🔍 Analyzing damage patterns...'):
            img = Image.open(uploaded_file)
            img = np.array(img)
            
            st.markdown("""
                <div class="progress-bar">
                    <div class="progress-value"></div>
                </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(img, caption="Original Vehicle Image", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            annotated_img = process_frame(img)
            
            with col2:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(annotated_img, caption="Damage Detection Result", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if st.button("💾 Save Analysis Report"):
                file_path = save_image(annotated_img, "damage_report.png")
                with open(file_path, "rb") as f:
                    st.download_button("📥 Download Report", f, "damage_report.png", "image/png")
                st.markdown("""
                    <div class="status success">
                        ✨ Analysis completed successfully! Your report is ready for download.
                    </div>
                """, unsafe_allow_html=True)

# Image Link Option
elif option == "Image Link":
    st.markdown("""
        <div class="glass-card">
            <h3>🔗 Analyze from URL</h3>
            <p>Enter the direct link to your vehicle image</p>
        </div>
    """, unsafe_allow_html=True)
    
    image_url = st.text_input("", placeholder="https://example.com/car-image.jpg")
    
    if image_url:
        try:
            with st.spinner('🌐 Fetching and analyzing image...'):
                response = requests.get(image_url)
                img = Image.open(BytesIO(response.content))
                img = np.array(img)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(img, caption="Original Vehicle Image", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                annotated_img = process_frame(img)
                
                with col2:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(annotated_img, caption="Damage Detection Result", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button("💾 Generate Report"):
                    file_path = save_image(annotated_img, "damage_report.png")
                    with open(file_path, "rb") as f:
                        st.download_button("📥 Download Report", f, "damage_report.png", "image/png")
                    st.markdown("""
                        <div class="status success">
                            ✨ Analysis completed! Your report is ready for download.
                        </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown("""
                <div class="status error">
                    ❌ Unable to process image. Please check the URL and try again.
                </div>
            """, unsafe_allow_html=True)

# Live Camera Option with streamlit-webrtc
elif option == "Live Camera":
    st.markdown("""
        <div class="glass-card">
            <h3>📷 Real-Time Detection</h3>
            <p>Instant damage detection using your camera</p>
        </div>
    """, unsafe_allow_html=True)
    
    run = st.checkbox('▶️ Activate Camera')
    
    # Define a video processor class to handle YOLO inference on each frame
    class YOLOProcessor(VideoProcessorBase):
        def __init__(self):
            self.confidence_threshold = confidence_threshold
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            annotated_img = process_frame(img_rgb)
            return av.VideoFrame.from_ndarray(annotated_img, format="rgb24")

    # Set up webrtc_streamer with the custom YOLO processor
    if run:
        webrtc_streamer(
            key="yolo-realtime",
            video_processor_factory=YOLOProcessor,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            media_stream_constraints={"video": True, "audio": False},
        )
    else:
        st.markdown("""
            <div class="status">
                ℹ️ Click the checkbox above to begin real-time detection
            </div>
        """, unsafe_allow_html=True)

# Enhanced Footer
st.markdown("""
    <div class="footer">
        <h3 style='margin-bottom: 1rem;'>🚀 Car Damage Detective</h3>
        <p style='margin-bottom: 0.5rem;'>Powered by YOLO V11m & Streamlit</p>
        <p style='opacity: 0.8;'>Created by Abie Nugraha | © 2024</p>
    </div>
""", unsafe_allow_html=True)
