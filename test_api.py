import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison
import requests
import io

# API Configuration
API_URL = "http://localhost:8000"

# Page Config
st.set_page_config(
    page_title="AquaEnhance - AI Image Enhancer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# [Your CSS styles here - keep them as they are]

# Helper Functions
@st.cache_data
def check_api_health():
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def enhance_image_api(image_file):
    """Send image to FastAPI for enhancement"""
    try:
        img_byte_arr = io.BytesIO()
        image_file.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(f"{API_URL}/enhance", files=files, timeout=30)
        
        if response.status_code == 200:
            enhanced = Image.open(io.BytesIO(response.content))
            return enhanced, None
        else:
            return None, f"API Error: {response.status_code}"
    
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Ensure FastAPI server is running."
    except Exception as e:
        return None, f"Error: {str(e)}"

# Hero Section
st.markdown('<h1 class="hero-title">🌊 AquaEnhance</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Transform your underwater memories with AI-powered color restoration</p>', unsafe_allow_html=True)

# Check API Status
api_status = check_api_health()
if api_status:
    st.success(f"✓ Connected to API | Model: {api_status.get('model_architecture', 'N/A')}")
else:
    st.error("⚠️ Cannot connect to FastAPI. Run: `python app.py`")

# Image Upload & Processing
st.markdown('<h2 class="section-header">✨ Experience the Magic</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="image_upload")

if uploaded_file:
    original = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner("🔄 Enhancing your image..."):
        enhanced, error = enhance_image_api(original)
    
    if error:
        st.error(f"❌ {error}")
        enhanced = original
    else:
        st.success("✓ Enhancement complete!")
    
    # Display comparison
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image_comparison(img1=original, img2=enhanced, label1="Original", label2="Enhanced", width=700)
    
    # Download button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        buf = io.BytesIO()
        enhanced.save(buf, format='PNG')
        st.download_button(
            label="📥 Download Enhanced Image",
            data=buf.getvalue(),
            file_name="enhanced_aqua_image.png",
            mime="image/png"
        )
else:
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-text">
            📷 Drop your underwater photo here<br>
            <small>Supports JPG, JPEG, PNG</small>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Features Section - FIXED HERE
st.markdown('<h2 class="section-header">🎯 Why Choose AquaEnhance?</h2>', unsafe_allow_html=True)

features_html = """
<div class="feature-grid">
    <div class="feature-card">
        <span class="feature-icon">🎨</span>
        <h3 class="feature-title">AI-Powered Restoration</h3>
        <p class="feature-desc">Advanced CLCC neural network trained specifically on underwater imagery</p>
    </div>
    <div class="feature-card">
        <span class="feature-icon">⚡</span>
        <h3 class="feature-title">Lightning Fast</h3>
        <p class="feature-desc">Process images in seconds with GPU acceleration</p>
    </div>
    <div class="feature-card">
        <span class="feature-icon">🔒</span>
        <h3 class="feature-title">Privacy First</h3>
        <p class="feature-desc">Images never stored - complete data privacy guaranteed</p>
    </div>
    <div class="feature-card">
        <span class="feature-icon">🌊</span>
        <h3 class="feature-title">Underwater Specialist</h3>
        <p class="feature-desc">Purpose-built for marine photography</p>
    </div>
</div>
"""
st.markdown(features_html, unsafe_allow_html=True)

# [Rest of your code: Stats, Team, Contact, Footer sections]