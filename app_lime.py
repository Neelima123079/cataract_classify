import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gc

from lime import lime_image
from skimage.segmentation import mark_boundaries

# IMPORTANT:
# use the SAME preprocessing used during training
from keras.applications.resnet import preprocess_input

# =========================
# CONFIG
# =========================
WEIGHTS_PATH = "best_model_cataract1.h5"
IMG_SIZE = 224

# Change this ONLY after checking train_generator.class_indices
CLASS_NAMES = ["Cataract", "Normal"]

LIME_NUM_SAMPLES = 120
LIME_NUM_FEATURES = 4

# =========================
# LOAD FULL MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(WEIGHTS_PATH, compile=False)

model = load_model()

# =========================
# PREPROCESS
# =========================
def preprocess_for_model(img):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)   # same as training
    return arr

def prepare_image_for_lime(img):
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    return arr

def classifier_fn(images):
    images = np.array(images).astype("float32")
    images_pp = np.array([preprocess_input(img.copy()) for img in images])
    return model.predict(images_pp, verbose=0)

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

def explain(image_for_lime, image_for_model):
    explainer = get_explainer()

    explanation = explainer.explain_instance(
        image_for_lime.astype("double"),
        classifier_fn=classifier_fn,
        top_labels=2,
        hide_color=0,
        num_samples=LIME_NUM_SAMPLES
    )

    preds = predict(image_for_model)
    pred_idx = int(np.argmax(preds))

    temp, mask = explanation.get_image_and_mask(
        label=pred_idx,
        positive_only=True,
        num_features=LIME_NUM_FEATURES,
        hide_rest=False
    )

    if temp.max() > 1:
        temp = temp / 255.0

    lime_img = mark_boundaries(temp, mask)
    lime_img = np.clip(lime_img, 0, 1)

    return lime_img, preds, pred_idx

# =========================
# UI
# =========================
st.set_page_config(page_title="Cataract XAI App", layout="wide")
st.title("Cataract Detection with Explainable AI")

uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    image_for_model = preprocess_for_model(image)
    image_for_lime = prepare_image_for_lime(image)

    st.image(image, caption="Uploaded Eye Image", use_container_width=True)

    if st.button("🔍 Run Prediction"):
        preds = predict(image_for_model)
        pred_idx = int(np.argmax(preds))

        st.write("Raw prediction vector:", preds)
        st.write(f"{CLASS_NAMES[0]} probability:", float(preds[0]))
        st.write(f"{CLASS_NAMES[1]} probability:", float(preds[1]))

        st.success(f"Prediction: {CLASS_NAMES[pred_idx]}")
        st.write("Confidence:", float(preds[pred_idx]))

    if st.button("🧠 Generate Explanation (LIME)"):
        with st.spinner("Generating explanation..."):
            lime_img, preds, pred_idx = explain(image_for_lime, image_for_model)

        st.write("Raw prediction vector:", preds)
        st.write(f"{CLASS_NAMES[0]} probability:", float(preds[0]))
        st.write(f"{CLASS_NAMES[1]} probability:", float(preds[1]))

        st.success(f"Prediction: {CLASS_NAMES[pred_idx]}")
        st.write("Confidence:", float(preds[pred_idx]))
        st.image(lime_img, caption="LIME Explanation", use_container_width=True)

        del lime_img
        gc.collect()

st.markdown("---")
st.markdown("Developed for Explainable AI in Medical Diagnosis")
