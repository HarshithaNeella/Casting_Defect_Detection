import streamlit as st
import numpy as np
import cv2
import pickle
import base64
from skimage.feature import hog
# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Casting Defect Detection",
    layout="centered"
)

# --------------------------------------------------
# BACKGROUND IMAGE (BASE64 SAFE)
# --------------------------------------------------
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("Casting_img.jpg")

# --------------------------------------------------
# GLOBAL STYLES
# --------------------------------------------------
st.markdown("""
<style>
h1 {
    color: #FFD700;
    text-align: center;
    font-weight: 800;
}

h3 {
    color: #EAEAEA;
    text-align: center;
}

label, p {
    color: #F2F2F2;
}

.prediction-box {
    padding: 20px;
    border-radius: 12px;
    font-size: 26px;
    font-weight: 800;
    text-align: center;
    margin-top: 20px;
}

.ok {
    background-color: rgba(0, 160, 0, 0.85);
    color: white;
}

.defective {
    background-color: rgba(200, 0, 0, 0.85);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL (PIPELINE WITH SCALER INSIDE)
# --------------------------------------------------
with open("casting_model.pkl", "rb") as f:
    model = pickle.load(f)

# --------------------------------------------------
# IMAGE PREPROCESSING (MATCHES TRAINING EXACTLY)
# --------------------------------------------------
def preprocess_image(image):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image=cv2.resize(image,(200,200),interpolation=cv2.INTER_LINEAR)
    hog_features=hog(image,
                orientations=9,
                pixels_per_cell=(4,4),
                cells_per_block=(2,2),
                block_norm='L2-Hys')
    
    return hog_features.reshape(1,-1)
    

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title(" Casting Defect Detection")
st.subheader("ML-based Visual Quality Inspection System")

uploaded_file = st.file_uploader(
    "Upload a casting image",
    type=["jpg", "png", "jpeg"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, width=400)

    if st.button("🔍 Prediction"):
        processed_image = preprocess_image(image)
        prediction = model.predict_proba(processed_image)[0][0]

        if prediction > 0.3:
            st.markdown(
                '<div class="prediction-box defective">❌ DEFECTIVE CASTING</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="prediction-box ok">✅ NON-DEFECTIVE CASTING</div>',
                unsafe_allow_html=True
            )
