import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image

# Page config
st.set_page_config(
    page_title="🐱🐶 Pet Classifier",
    page_icon="🎯",
    layout="centered"
)

# Simple styling
st.markdown("""
<style>
    .big-font {
        font-size: 50px !important;
        text-align: center;
        margin: 20px 0;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
        font-size: 24px;
        font-weight: bold;
    }
    .cat-result {
        background-color: #ff6b6b;
        color: white;
    }
    .dog-result {
        background-color: #4ecdc4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_components():
    """Load model components"""
    try:
        with open('svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('pca.pkl', 'rb') as f:
            pca = pickle.load(f)
        with open('model_info.pkl', 'rb') as f:
            model_info = pickle.load(f)
        return model, scaler, pca, model_info
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None, None

def preprocess_image(image, img_size):
    """Preprocess image for prediction"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    elif len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    img_resized = cv2.resize(img_array, (img_size, img_size))
    
    # Feature extraction
    img_normalized = img_resized.astype('float32') / 255.0
    raw_features = img_normalized.flatten()[::4]
    
    hist_r = cv2.calcHist([img_resized], [0], None, [32], [0, 256]).flatten()
    hist_g = cv2.calcHist([img_resized], [1], None, [32], [0, 256]).flatten()
    hist_b = cv2.calcHist([img_resized], [2], None, [32], [0, 256]).flatten()
    color_features = np.concatenate([hist_r, hist_g, hist_b]) / (img_size * img_size)
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_features = edges.flatten().astype('float32')[::8] / 255.0
    
    texture_features = []
    window_size = 8
    for i in range(0, img_size - window_size, window_size):
        for j in range(0, img_size - window_size, window_size):
            window = gray[i:i+window_size, j:j+window_size]
            texture_features.append(np.std(window))
    
    combined_features = np.concatenate([
        raw_features,
        color_features,
        edge_features,
        np.array(texture_features)
    ])
    
    # Ensure exactly 12288 features
    target_size = 12288
    if len(combined_features) > target_size:
        combined_features = combined_features[:target_size]
    elif len(combined_features) < target_size:
        padding = np.zeros(target_size - len(combined_features))
        combined_features = np.concatenate([combined_features, padding])
    
    return combined_features

def main():
    # Header
    st.markdown('<p class="big-font">🐱 vs 🐶</p>', unsafe_allow_html=True)
    st.markdown("### Upload a photo and I'll tell you if it's a cat or dog!")
    
    # Load model
    model, scaler, pca, model_info = load_model_components()
    
    if model is None:
        st.error("❌ Could not load the AI model!")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📸 Choose your pet photo",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear photo of a cat or dog for best results!"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Your pet photo", use_container_width=True)
        
        # Predict button
        if st.button("🔮 Analyze Pet!", use_container_width=True):
            with st.spinner('🤔 Thinking...'):
                try:
                    # Make prediction
                    processed_img = preprocess_image(image, model_info['img_size'])
                    processed_img = processed_img.reshape(1, -1)
                    scaled_img = scaler.transform(processed_img)
                    pca_img = pca.transform(scaled_img)
                    
                    prediction = model.predict(pca_img)[0]
                    probabilities = model.predict_proba(pca_img)[0]
                    confidence = max(probabilities) * 100
                    
                    # Show result with style
                    if prediction == 1:
                        st.markdown(
                            f'<div class="result-box dog-result">🐶 WOOF! It\'s a DOG!<br>Confidence: {confidence:.0f}%</div>',
                            unsafe_allow_html=True
                        )
                        st.balloons()
                    else:
                        st.markdown(
                            f'<div class="result-box cat-result">🐱 MEOW! It\'s a CAT!<br>Confidence: {confidence:.0f}%</div>',
                            unsafe_allow_html=True
                        )
                        st.balloons()
                    
                except Exception as e:
                    st.error(f"😵 Oops! Something went wrong: {e}")
                    st.info("Try uploading a different image!")
    
    else:
        # Instructions
        st.info("👆 Upload a photo above to get started!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🐱 Cats")
            st.markdown("- Pointy ears\n- Whiskers\n- Usually smaller")
        with col2:
            st.markdown("### 🐶 Dogs") 
            st.markdown("- Floppy or perky ears\n- Wet nose\n- Usually bigger")

if __name__ == "__main__":
    main()