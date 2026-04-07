import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gc

from lime import lime_image
from skimage.segmentation import mark_boundaries

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# =========================
# CONFIG
# =========================
WEIGHTS_PATH = "best_model_cataract1.h5"
IMG_SIZE = 224
CLASS_NAMES = ["Cataract", "Normal"]

# Reduced for memory safety
LIME_NUM_SAMPLES = 120
LIME_NUM_FEATURES = 4

# =========================
# BUILD MODEL
# =========================
def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None,
        alpha=0.35
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation="relu")(x)
    output = Dense(2, activation="softmax", use_bias=False)(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model

@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights(WEIGHTS_PATH)
    return model

model = load_model()

# =========================
# PREPROCESS
# =========================
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def preprocess(img):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    return arr

def predict(image_np):
    image_np = np.expand_dims(image_np, axis=0)
    preds = model.predict(image_np, verbose=0)[0]
    return preds

# =========================
# LIME
# =========================
@st.cache_resource
def get_explainer():
    return lime_image.LimeImageExplainer()

def explain(image_np):
    explainer = get_explainer()
lime_img = np.clip(lime_img, 0, 1)
    explanation = explainer.explain_instance(
        image_np.astype("double"),
        classifier_fn=lambda x: model.predict(x),
        top_labels=2,
        hide_color=0,
        num_samples=LIME_NUM_SAMPLES
    )

    preds = predict(image_np)
    pred_idx = int(np.argmax(preds))

    temp, mask = explanation.get_image_and_mask(
        label=pred_idx,
        positive_only=True,
        num_features=LIME_NUM_FEATURES,
        hide_rest=False
    )

    if temp.max() > 1:
        temp = temp / 255.0

    return mark_boundaries(temp, mask), preds, pred_idx

# =========================
# UI
# =========================
st.set_page_config(page_title="Cataract XAI App", layout="wide")
st.title("Cataract Detection with Explainable AI")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = preprocess(image)

    st.image(lime_img, caption="LIME Explanation", use_container_width=True, clamp=True)

    # Prediction button
    if st.button("🔍 Run Prediction"):
        preds = predict(image_np)
        pred_idx = int(np.argmax(preds))

        st.success(f"Prediction: {CLASS_NAMES[pred_idx]}")
        st.write("Confidence:", float(preds[pred_idx]))

    # LIME button (separate!)
    if st.button("🧠 Generate Explanation (LIME)"):
        with st.spinner("Generating explanation..."):
            lime_img, preds, pred_idx = explain(image_np)

        st.image(lime_img, caption="LIME Explanation", use_container_width=True)

        # Free memory
        del lime_img
        gc.collect()

# Footer
st.markdown("---")
st.markdown("Developed for Explainable AI in Medical Diagnosis")
