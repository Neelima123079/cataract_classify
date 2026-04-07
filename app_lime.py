import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from streamlit_pdf_viewer import pdf_viewer

from lime import lime_image
from skimage.segmentation import mark_boundaries


IMG_SIZE = 224
CLASS_NAMES = ["Normal", "Cataract"]
BINARY_SIGMOID_OUTPUT = True
LIME_NUM_SAMPLES = 1000
LIME_NUM_FEATURES = 5

import os
import streamlit as st
import tensorflow as tf

MODEL_PATH = "cataract_resaved.h5"


def load_model():
    st.write("Current working directory:", os.getcwd())
    st.write("Files in app folder:", os.listdir("."))
    st.write("Trying to load model from:", MODEL_PATH)
    st.write("File exists:", os.path.exists(MODEL_PATH))
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

def preprocess_pil(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return arr

def predict_fn(images):
    batch = []
    for img in images:
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img.astype("uint8"))
        else:
            pil_img = img
        batch.append(preprocess_pil(pil_img))

    batch = np.array(batch, dtype=np.float32)
    preds = model.predict(batch, verbose=0)
    preds = np.array(preds)

    if BINARY_SIGMOID_OUTPUT:
        preds = preds.reshape(-1, 1)
        probs = np.hstack([1 - preds, preds])
        return probs

    if preds.ndim == 2 and preds.shape[1] == 2:
        return preds

    raise ValueError(f"Unexpected model output shape: {preds.shape}")

def predict_single(image_np):
    probs = predict_fn([image_np])[0]
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs

@st.cache_resource
def get_explainer():
    return lime_image.LimeImageExplainer()

def explain_image(image_np):
    explainer = get_explainer()

    explanation = explainer.explain_instance(
        image_np.astype("double"),
        classifier_fn=predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=LIME_NUM_SAMPLES
    )

    pred_idx, probs = predict_single(image_np)

    temp, mask = explanation.get_image_and_mask(
        label=pred_idx,
        positive_only=True,
        num_features=LIME_NUM_FEATURES,
        hide_rest=False
    )

    if temp.max() > 1:
        temp = temp / 255.0

    lime_vis = mark_boundaries(temp, mask)
    return pred_idx, probs, lime_vis

st.set_page_config(page_title="Cataract Classifier with LIME", layout="wide")
st.title("Cataract Image Classifier Web App with Explainability")

tab1, tab2 = st.tabs(["Make Prediction", "View Report"])

with tab1:
    st.header("Cataract and Normal Eye Image Classifier")
    st.subheader("Upload an image for prediction and explanation")

    uploaded_file = st.file_uploader(
        label="Upload an eye image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        file_details = {
            "file name": uploaded_file.name,
            "file type": uploaded_file.type,
            "file size": uploaded_file.size
        }

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Uploaded Image")
            st.write(file_details)
            st.image(image, use_container_width=True)

        with st.spinner("Running prediction and generating LIME explanation..."):
            pred_idx, probs, lime_vis = explain_image(image_np)

        with col2:
            st.write("### Prediction Result")
            st.metric("Prediction Label", CLASS_NAMES[pred_idx])
            st.metric("Confidence Score", f"{float(probs[pred_idx]):.4f}")

            st.write("### Class Probabilities")
            st.write({
                "Normal": float(probs[0]),
                "Cataract": float(probs[1])
            })

        st.write("### LIME Explanation")
        st.image(
            lime_vis,
            caption="Highlighted regions contributed positively to the predicted class.",
            use_container_width=True
        )

        st.info(
            "The highlighted image regions show which parts of the eye image "
            "most influenced the model's prediction."
        )

with tab2:
    st.header("Project Report")

    st.link_button(
        "My GitHub Repository",
        "https://github.com/Neelima123079/cataract_classify"
    )

    pdf_url = "https://raw.githubusercontent.com/Neelima123079/cataract_classify/main/Cataract Classifier project report.pdf"
    response = requests.get(pdf_url)

    if response.status_code == 200:
        st.download_button(
            label="Download Report",
            data=response.content,
            file_name="Cataract_Classifier_Report.pdf",
            mime="application/pdf"
        )

        if st.button("Show Report"):
            with st.sidebar:
                pdf_viewer(response.content)
    else:
        st.error("Failed to fetch PDF file. Please check the URL.")
