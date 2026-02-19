
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
# BACKGROUND
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
# STYLES
# --------------------------------------------------
st.markdown("""
<style>

/* FORCE Title Color */
h1, h2, h3 {
    color: white !important;
    text-align: center !important;
    font-weight: 800 !important;
    text-shadow: 3px 3px 8px black !important;
}

/* Streamlit Title */
div[data-testid="stTitle"] * {
    color: white !important;
}

/* Streamlit Subheader */
div[data-testid="stSubheader"] * {
    color: #F2F2F2 !important;
}

/* Prediction Box */
.prediction-box {
    padding: 25px;
    border-radius: 12px;
    font-size: 26px;
    font-weight: 800;
    text-align: center;
    margin-top: 25px;
}

.ok {
    background-color: rgba(0,160,0,0.9);
    color: white;
}

.defective {
    background-color: rgba(200,0,0,0.9);
    color: white;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
with open("Casting_hog_model.pkl", "rb") as f:
    model = pickle.load(f)


# --------------------------------------------------
# PREPROCESS (HOG - SAME AS TRAINING)
# --------------------------------------------------
def preprocess_image(image):

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize (MUST match training)
    gray = cv2.resize(gray, (200, 200))

    # Extract HOG
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1)


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Casting Defect Detection")
st.subheader("ML-Based Visual Quality Inspection System")

uploaded_file = st.file_uploader(
    "Upload a Casting Image",
    type=["jpg", "png", "jpeg"]
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, width=400)


    if st.button("🔍 Predict"):

        # Preprocess using HOG
        X = preprocess_image(image)

        # Predict probability
        prob = model.predict_proba(X)[0]

        non_defective = prob[1]
        defective = prob[0]

        


        # Result
        if defective >= 0.4:

            st.markdown(
                f"""
                <div class="prediction-box defective">
                ❌ DEFECTIVE CASTING<br>
                
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="prediction-box ok">
                ✅ NON-DEFECTIVE CASTING<br>
                
                </div>
                """,
                unsafe_allow_html=True
            )

