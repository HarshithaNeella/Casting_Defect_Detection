# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
import streamlit as st
import numpy as np
import cv2
import pickle
import base64


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Casting Defect Detection",
    layout="centered"
)


# --------------------------------------------------
# BACKGROUND FUNCTION
# --------------------------------------------------
def set_background(image_path):

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(0,0,0,0.7),
                    rgba(0,0,0,0.7)
                ),
                url("data:image/jpg;base64,{encoded}");

            background-size: cover;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Set background
set_background("Casting_img.jpg")


# --------------------------------------------------
# STYLES
# --------------------------------------------------
st.markdown("""
<style>

h1, h2, h3 {
    color: white !important;
    text-align: center !important;
    font-weight: 800 !important;
    text-shadow: 3px 3px 8px black !important;
}

div[data-testid="stTitle"] * {
    color: white !important;
}

div[data-testid="stSubheader"] * {
    color: #F2F2F2 !important;
}

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
# LOAD MODEL (.pkl CNN)
# --------------------------------------------------
with open("CNN_Casting_model.pkl", "rb") as f:
    model = pickle.load(f)


# --------------------------------------------------
# PREPROCESS FUNCTION (FIXED)
# --------------------------------------------------
def preprocess_image(uploaded_file):

    if uploaded_file is None:
        return None, None

    # Read file bytes
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    # Decode image
    image = cv2.imdecode(file_bytes, 1)

    if image is None:
        st.error("Image decoding failed")
        return None, None

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Resize
    gray = cv2.resize(
        gray,
        (200, 200)
    )

    # Normalize
    gray = gray / 255.0

    # Add channel dimension
    gray = np.expand_dims(
        gray,
        axis=-1
    )

    # Add batch dimension
    gray = np.expand_dims(
        gray,
        axis=0
    )

    return gray, image


# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Casting Defect Detection")

st.subheader(
    "CNN-Based Visual Quality Inspection System"
)

uploaded_file = st.file_uploader(
    "Upload a Casting Image",
    type=["jpg", "png", "jpeg"]
)


# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if uploaded_file is not None:

    # Preprocess once
    X, original_img = preprocess_image(uploaded_file)

    if X is not None:

        st.subheader("Original Image")

        st.image(
            original_img,
            channels="BGR",
            width=400
        )

        if st.button("🔍 Predict"):

            # CNN Prediction
            prob = model.predict(X)[0][0]

            # Show probability
            

            st.write(
                f"Prediction Probability: {prob:.5f}"
            )

            
            # Classification
            if prob <= 0.5:

                st.markdown(
                    f"""
                    <div class="prediction-box defective">
                    ❌ DEFECTIVE CASTING
                    <br>
                
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="prediction-box ok">
                    ✅ NON-DEFECTIVE CASTING
                    <br>
                    
                    </div>
                    """,
                    unsafe_allow_html=True
                )