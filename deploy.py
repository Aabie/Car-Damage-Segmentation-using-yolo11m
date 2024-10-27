import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
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

# Enhanced Modern CSS with tab styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: linear-gradient(135deg, #1a1f2d, #2d1a2a);
        color: #ffffff;
    }
    
    /* Main container enhancements */
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    /* Modern header design */
    .header {
        background: linear-gradient(135deg, #2a3346, #462a44);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 0rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .header-title {
        background: linear-gradient(135deg, #c684fc, #8684fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    /* Enhanced tab design */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(42, 51, 70, 0.95);
        padding: 0.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        width: 250px;
        background-color: transparent;
        border: 1px solid rgba(198, 132, 252, 0.1);
        border-radius: 10px;
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(198, 132, 252, 0.1);
        border-color: rgba(198, 132, 252, 0.3);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #c684fc, #8684fc) !important;
        border: none !important;
    }

    /* Enhanced card design */
    .glass-card {
        background: rgba(42, 51, 70, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(198, 132, 252, 0.1);
        text-align: center;
    }

    /* Rest of your existing CSS styles... */
    </style>
""", unsafe_allow_html=True)

# Enhanced Header
st.markdown("""
    <style>
        .header {
            text-align: center;
            padding: 20px;
            background-color: #e0f7fa; /* Soft aqua background color */
            border-radius: 15px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
            animation: pulse 2s infinite; /* Add a subtle pulse animation */
        }
        .header-title {
            font-size: 3em;
            color: #ff5722; /* Bright orange color for title */
            font-weight: bold;
            margin: 0;
            transition: color 0.5s; /* Smooth color transition on hover */
        }
        .header-title:hover {
            color: #f44336; /* Change color on hover */
        }
        .header-subtitle {
            font-size: 1.8em;
            color: white; /* Dark teal for subtitle */
            margin: 5px 0 0 0;
            font-style: italic; /* Italicize subtitle */
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
    </style>
    <div class="header">
        <p class="header-title">🚀 Car Damage Detective</p>
        <p class="header-subtitle">🔍 Unraveling Vehicle Mysteries with AI Magic!</p>
    </div>
""", unsafe_allow_html=True)



# Custom CSS for styling
st.markdown("""
    <style>
        .glass-card {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            padding: 5px;
            margin-bottom: 20px;
            backdrop-filter: blur(15px);
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }

        h4 {
            font-family: 'Arial', sans-serif;
            font-size: 1.5rem;
            color: #ffffff;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.5);
        }

        ul {
            list-style-type: disc;
            padding-left: 20px;
            color: #f0f0f0;
        }

        li {
            margin: 5px 0;
            font-size: 1.1rem;
        }
    </style>
""", unsafe_allow_html=True)


with st.expander("⚙️ Detection Settings", expanded=True):
    confidence_threshold = st.slider(
        "Detection Confidnece Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        format="%.2f"
    )
    
    st.markdown("""
        <div style='border: 1px solid #cccccc; border-radius: 8px; padding: 10px;'>
            <h4 style='color: white; margin-bottom: 1rem;'>💡 Pro Tips</h4>
            <ul style='color: white;'>
                <li>Higher Confidnece Threshold, = More precise detection (but harder to detect)</li>
                <li>Ensure good lighting conditions</li>
                <li>Keep camera steady for best results</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)


# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📷 Camera", "🔗 Image Link"])

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

# Upload Image Tab
with tab1:
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
            
            if st.button("💾 Save Analysis Report", key="save_upload"):
                file_path = save_image(annotated_img, "damage_report.png")
                with open(file_path, "rb") as f:
                    st.download_button("📥 Download Report", f, "damage_report.png", "image/png")
                st.markdown("""
                    <div class="status success">
                        ✨ Analysis completed successfully! Your report is ready for download.
                    </div>
                """, unsafe_allow_html=True)

# Camera Tab
with tab2:
    st.markdown("""
        <div class="glass-card">
            <h3>📷 Camera Detection</h3>
            <p>Take a photo of the vehicle damage using your camera</p>
        </div>
    """, unsafe_allow_html=True)
    
    camera_image = st.camera_input("Point camera at vehicle damage")
    
    if camera_image is not None:
        with st.spinner('🔍 Analyzing damage patterns...'):
            img = Image.open(camera_image)
            img_array = np.array(img)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(img_array, caption="Captured Image", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            try:
                annotated_img = process_frame(img_array)
                
                with col2:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(annotated_img, caption="Damage Detection Result", use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                if st.button("💾 Save Analysis Report", key="save_camera"):
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    file_name = f"damage_report_{timestamp}.png"
                    file_path = save_image(annotated_img, file_name)
                    
                    with open(file_path, "rb") as f:
                        st.download_button("📥 Download Report", f, file_name, "image/png")
                    st.markdown("""
                        <div class="status success">
                            ✨ Analysis completed successfully! Your report is ready for download.
                        </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.markdown("""
                    <div class="status error">
                        ❌ Unable to process image. Please try again.
                    </div>
                """, unsafe_allow_html=True)

# Image Link Tab
with tab3:
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
                
                if st.button("💾 Generate Report", key="save_url"):
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

# Footer
st.markdown("""
    <style>
    .footer {
        background: linear-gradient(135deg, rgba(42, 51, 70, 0.95), rgba(28, 35, 50, 0.95));
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-top: 2rem;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        box-shadow: 0 -4px 10px rgba(0, 0, 0, 0.3);
        transition: background 0.3s ease; /* Add transition for background change */
    }

    .footer:hover {
        background: linear-gradient(135deg, rgba(28, 35, 50, 0.95), rgba(42, 51, 70, 0.95));
    }

    .footer p {
        margin: 0;
        transition: color 0.3s ease, transform 0.3s ease; /* Add transform transition */
    }

    .footer p:hover {
        color: #c684fc; /* Change color on hover */
        transform: scale(1.05); /* Slightly enlarge on hover */
    }
    </style>

    <div class="footer">
        <p>&copy; 2024 Car Damage Detective | All rights reserved | by Abie Nugraha</p>
    </div>
""", unsafe_allow_html=True)
