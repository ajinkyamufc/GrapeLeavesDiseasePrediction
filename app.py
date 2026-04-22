import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Grape Leaf Disease Classifier",
    page_icon="🍇",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #1a0a2e 0%, #16213e 50%, #0f3460 100%);
    min-height: 100vh;
}

/* Title */
h1 {
    font-family: 'Playfair Display', serif !important;
    color: #c8a2e8 !important;
    text-align: center;
    font-size: 2.6rem !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.2rem !important;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #8899aa;
    font-size: 0.95rem;
    margin-bottom: 2.5rem;
    letter-spacing: 0.5px;
}

/* Upload area */
[data-testid="stFileUploadDropzone"] {
    background: rgba(200, 162, 232, 0.06) !important;
    border: 2px dashed rgba(200, 162, 232, 0.35) !important;
    border-radius: 16px !important;
    color: #c8a2e8 !important;
    padding: 2rem !important;
    transition: border-color 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: rgba(200, 162, 232, 0.7) !important;
}

/* Result card */
.result-card {
    background: rgba(200, 162, 232, 0.1);
    border: 1px solid rgba(200, 162, 232, 0.25);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1.5rem;
    text-align: center;
    backdrop-filter: blur(8px);
}
.result-label {
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8899aa;
    margin-bottom: 0.4rem;
}
.result-disease {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #e2c4ff;
    margin-bottom: 0.6rem;
}
.result-confidence {
    font-size: 0.9rem;
    color: #9ab;
}
.healthy { color: #7ee8a2; }
.diseased { color: #f9a8d4; }

/* Progress bar override */
[data-testid="stProgress"] > div > div {
    background-color: #c8a2e8 !important;
}

/* Image caption */
.stImage > div > div > p {
    color: #667788 !important;
    font-size: 0.8rem !important;
}

/* Divider */
hr { border-color: rgba(200,162,232,0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants — edit these to match your model ────────────────────────────────
MODEL_PATH = "grape_disease_classification_model.tflite"          # path to your .tflite file
IMAGE_SIZE = (224, 224)              # input size expected by the model

# Update this list to match the exact class order your model was trained on
CLASS_NAMES = [
    "Black Rot",
    "Esca (Black Measles)",
    "Healthy",
    "Leaf Blight (Isariopsis Leaf Spot)",
]


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(interpreter, img_array: np.ndarray):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    idx = int(np.argmax(output))
    return CLASS_NAMES[idx], float(output[idx]), output


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("<h1>🍇 Grape Leaf Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Upload a grape leaf image · Get instant disease diagnosis</p>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Drop a leaf image here, or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded leaf", use_column_width=True)

    st.markdown("---")

    with st.spinner("Analysing the leaf…"):
        try:
            interpreter = load_model(MODEL_PATH)
            arr = preprocess(image)
            disease, confidence, all_scores = predict(interpreter, arr)

            is_healthy = "healthy" in disease.lower()
            status_cls = "healthy" if is_healthy else "diseased"
            icon = "✅" if is_healthy else "🔬"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Diagnosis</div>
                <div class="result-disease {status_cls}">{icon} {disease}</div>
                <div class="result-confidence">Confidence: {confidence * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar chart for all classes
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Confidence breakdown**")
            for name, score in zip(CLASS_NAMES, all_scores):
                st.progress(float(score), text=f"{name}  —  {score*100:.1f}%")

        except FileNotFoundError:
            st.error(
                f"⚠️ Model file `{MODEL_PATH}` not found. "
                "Place your `.tflite` file in the same directory as `app.py` "
                "and update `MODEL_PATH` if needed."
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.info("👆 Upload a grape leaf image to get started.")
