import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gc
import cv2

from lime import lime_image
from skimage.segmentation import mark_boundaries
import shap

# SAME preprocessing used during training
from keras.applications.resnet import preprocess_input

# =========================
# CONFIG
# =========================
MODEL_PATH = "best_model_cataract1.keras"
IMG_SIZE = 224
CLASS_NAMES = ["Cataract", "Normal"]   # update if class order differs
LIME_NUM_SAMPLES = 120
LIME_NUM_FEATURES = 4
LAST_CONV_LAYER_NAME = "Conv_1"  # MobileNetV2 usually uses Conv_1
GOV_CONF_THRESHOLD = 0.75
GOV_HMEG_THRESHOLD = 0.60
GOV_TRUST_THRESHOLD = 3.0

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Cataract Detection with HME-G", layout="wide")
st.title("Cataract Detection with Explainable AI and HME-G")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# =========================
# PREPROCESSING
# =========================
def preprocess_for_model(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    return arr

def prepare_image_for_display(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img).astype("uint8")

def predict(image_np: np.ndarray) -> np.ndarray:
    image_np = np.expand_dims(image_np, axis=0)
    preds = model.predict(image_np, verbose=0)[0]
    return preds

def classifier_fn(images: np.ndarray) -> np.ndarray:
    images = np.array(images).astype("float32")
    processed = np.array([preprocess_input(img.copy()) for img in images])
    return model.predict(processed, verbose=0)

# =========================
# LIME
# =========================
@st.cache_resource
def get_lime_explainer():
    return lime_image.LimeImageExplainer()

def explain_lime(image_for_lime: np.ndarray, image_for_model: np.ndarray):
    explainer = get_lime_explainer()

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

    # simple sparsity proxy from selected superpixels
    sparsity_score = 1.0 / max(np.unique(mask).size, 1)

    return lime_img, preds, pred_idx, mask, sparsity_score

# =========================
# GRAD-CAM
# =========================
def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)

        pred_index = tf.argmax(predictions[0])
        class_channel = tf.gather(predictions[0], pred_index)

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if float(max_val) == 0.0:
        return np.zeros((heatmap.shape[0], heatmap.shape[1]), dtype=np.float32)

    heatmap = heatmap / max_val
    return heatmap.numpy()

# =========================
# SHAP
# =========================
def explain_shap(image_for_display: np.ndarray, image_for_model: np.ndarray):
    """
    Uses SHAP Image masker with a tiny background set for efficiency.
    """
    background = np.expand_dims(image_for_display.astype("float32"), axis=0)
    masker = shap.maskers.Image("inpaint_telea", image_for_display.shape)

    def shap_predict(x):
        x = np.array(x).astype("float32")
        x_pp = np.array([preprocess_input(img.copy()) for img in x])
        return model.predict(x_pp, verbose=0)

    explainer = shap.Explainer(shap_predict, masker)
    shap_values = explainer(
        np.expand_dims(image_for_display.astype("float32"), axis=0),
        max_evals=100,
        batch_size=10
    )

    preds = predict(image_for_model)
    pred_idx = int(np.argmax(preds))

    # shap_values.values shape typically: (1, H, W, C, classes)
    values = shap_values.values[0, :, :, :, pred_idx]
    heatmap = np.mean(np.abs(values), axis=-1)

    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)

    shap_overlay = overlay_gradcam(image_for_display, heatmap)
    return shap_overlay, heatmap

# =========================
# HME-G METRICS
# =========================
def compute_fidelity(confidence: float) -> float:
    # simple deployable proxy: use confidence as local fidelity proxy
    return float(np.clip(confidence, 0.0, 1.0))

def compute_stability(image_for_model: np.ndarray, num_runs: int = 3, noise_std: float = 0.02) -> float:
    base_pred = predict(image_for_model)
    base_idx = int(np.argmax(base_pred))
    probs = []

    for _ in range(num_runs):
        noisy = image_for_model + np.random.normal(0, noise_std, image_for_model.shape).astype("float32")
        noisy = np.clip(noisy, -255, 255)
        p = predict(noisy)
        probs.append(float(p[base_idx]))

    variability = np.std(probs) if len(probs) > 1 else 0.0
    stability = 1.0 - variability
    return float(np.clip(stability, 0.0, 1.0))

def compute_bias_proxy(preds: np.ndarray) -> float:
    # placeholder governance-oriented proxy for single-instance demo
    # lower is better
    return float(np.clip(abs(float(preds[0]) - float(preds[1])) * 0.1, 0.0, 1.0))

def normalize_human_score(score_1_to_5: float) -> float:
    return float(np.clip((score_1_to_5 - 1.0) / 4.0, 0.0, 1.0))

def normalize_clarity(score_1_to_5: float) -> float:
    return float(np.clip((score_1_to_5 - 1.0) / 4.0, 0.0, 1.0))

def governance_audit(confidence, fidelity, stability, bias_proxy, human_trust):
    flags = []
    audit_score = 1.0

    if confidence < GOV_CONF_THRESHOLD:
        flags.append("Low confidence")
        audit_score -= 0.20

    if fidelity < 0.70:
        flags.append("Low fidelity")
        audit_score -= 0.15

    if stability < 0.70:
        flags.append("Low stability")
        audit_score -= 0.15

    if bias_proxy > 0.20:
        flags.append("Potential bias/disparity")
        audit_score -= 0.20

    if human_trust < GOV_TRUST_THRESHOLD:
        flags.append("Low user trust")
        audit_score -= 0.20

    audit_score = float(np.clip(audit_score, 0.0, 1.0))
    return audit_score, flags

def compute_hmeg_score(metrics_dict, human_trust_score, human_clarity_score, audit_score):
    """
    HME-G weighted aggregation:
    metrics = fidelity, stability, sparsity, interpretability, governance
    """
    interpretability = (normalize_human_score(human_trust_score) + normalize_clarity(human_clarity_score)) / 2.0

    metric_values = {
        "fidelity": metrics_dict["fidelity"],
        "stability": metrics_dict["stability"],
        "sparsity": metrics_dict["sparsity"],
        "interpretability": interpretability,
        "governance": audit_score
    }

    # adaptive weights from human + governance
    base_weights = {
        "fidelity": 0.22,
        "stability": 0.20,
        "sparsity": 0.12,
        "interpretability": 0.23,
        "governance": 0.23
    }

    trust_norm = normalize_human_score(human_trust_score)
    clarity_norm = normalize_clarity(human_clarity_score)

    adjusted_weights = {
        "fidelity": base_weights["fidelity"],
        "stability": base_weights["stability"],
        "sparsity": base_weights["sparsity"],
        "interpretability": base_weights["interpretability"] + 0.10 * ((trust_norm + clarity_norm) / 2.0),
        "governance": base_weights["governance"] + 0.10 * audit_score
    }

    total_w = sum(adjusted_weights.values())
    adjusted_weights = {k: v / total_w for k, v in adjusted_weights.items()}

    hmeg = sum(adjusted_weights[k] * metric_values[k] for k in metric_values)
    return float(np.clip(hmeg, 0.0, 1.0)), metric_values, adjusted_weights

def governance_decision(confidence, hmeg_score, human_trust, flags):
    if confidence >= GOV_CONF_THRESHOLD and hmeg_score >= GOV_HMEG_THRESHOLD and human_trust >= GOV_TRUST_THRESHOLD and len(flags) == 0:
        return "Approved"
    return "Review Required"
def overlay_gradcam(original_img, heatmap):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    return overlay    
# =========================
# UI
# =========================
uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_for_model = preprocess_for_model(image)
    image_for_display = prepare_image_for_display(image)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Eye Image", use_container_width=True)

        if st.button("🔍 Run Prediction"):
            preds = predict(image_for_model)
            pred_idx = int(np.argmax(preds))
            confidence = float(preds[pred_idx])

            if pred_idx == 0:
                st.error(f"Prediction: {CLASS_NAMES[pred_idx]} ({confidence*100:.2f}%)")
            else:
                st.success(f"Prediction: {CLASS_NAMES[pred_idx]} ({confidence*100:.2f}%)")

            st.write("Raw prediction vector:", preds)
            st.progress(confidence)

    with col2:
        st.subheader("Human Feedback for HME-G")
        human_trust = st.slider("Trust in explanation", 1, 5, 4)
        human_clarity = st.slider("Clarity of explanation", 1, 5, 4)
        clinical_alignment = st.selectbox(
            "Does explanation align with clinical expectation?",
            ["Yes", "Partially", "No"]
        )
        reviewer_comment = st.text_area("Reviewer comment", "")

    st.markdown("---")
    st.subheader("Explainability")

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    with exp_col1:
        if st.button("🧠 Generate LIME"):
            with st.spinner("Generating LIME explanation..."):
                lime_img, preds, pred_idx, mask, sparsity_score = explain_lime(image_for_display, image_for_model)

            st.image(lime_img, caption="LIME Explanation", use_container_width=True)

            st.session_state["lime_ready"] = True
            st.session_state["preds"] = preds
            st.session_state["pred_idx"] = pred_idx
            st.session_state["confidence"] = float(preds[pred_idx])
            st.session_state["sparsity_score"] = float(np.clip(sparsity_score, 0.0, 1.0))

            del lime_img
            gc.collect()

    with exp_col2:
        if st.button("🔥 Generate Grad-CAM"):
            with st.spinner("Generating Grad-CAM explanation..."):
                img_batch = np.expand_dims(image_for_model, axis=0)
                heatmap = get_gradcam_heatmap(model, img_batch, LAST_CONV_LAYER_NAME)
                gradcam_img = overlay_gradcam(image_for_display, heatmap)

            st.image(gradcam_img, caption="Grad-CAM Explanation", use_container_width=True)
            st.session_state["gradcam_ready"] = True

            del gradcam_img
            gc.collect()

    with exp_col3:
        if st.button("🌐 Generate SHAP"):
            with st.spinner("Generating SHAP explanation..."):
                shap_img, shap_heatmap = explain_shap(image_for_display, image_for_model)
            st.image(shap_img, caption="SHAP Explanation", use_container_width=True)
            st.session_state["shap_ready"] = True

            del shap_img, shap_heatmap
            gc.collect()

    st.markdown("---")
    st.subheader("HME-G Evaluation")

    if st.button("⚖️ Run HME-G Evaluation"):
        preds = predict(image_for_model)
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])

        stability = compute_stability(image_for_model)
        fidelity = compute_fidelity(confidence)
        bias_proxy = compute_bias_proxy(preds)

        # if LIME wasn't run, assign a neutral sparsity value
        sparsity_score = float(st.session_state.get("sparsity_score", 0.50))

        metrics_dict = {
            "fidelity": fidelity,
            "stability": stability,
            "sparsity": sparsity_score
        }

        audit_score, flags = governance_audit(
            confidence=confidence,
            fidelity=fidelity,
            stability=stability,
            bias_proxy=bias_proxy,
            human_trust=human_trust
        )

        hmeg_score, metric_values, adjusted_weights = compute_hmeg_score(
            metrics_dict=metrics_dict,
            human_trust_score=human_trust,
            human_clarity_score=human_clarity,
            audit_score=audit_score
        )

        decision = governance_decision(
            confidence=confidence,
            hmeg_score=hmeg_score,
            human_trust=human_trust,
            flags=flags if clinical_alignment != "Yes" else flags
        )

        if clinical_alignment == "No" and "Clinical mismatch" not in flags:
            flags.append("Clinical mismatch")
            decision = "Review Required"

        met1, met2, met3 = st.columns(3)
        with met1:
            st.metric("Prediction", CLASS_NAMES[pred_idx])
            st.metric("Confidence", f"{confidence:.3f}")
            st.metric("Fidelity", f"{metric_values['fidelity']:.3f}")
        with met2:
            st.metric("Stability", f"{metric_values['stability']:.3f}")
            st.metric("Sparsity", f"{metric_values['sparsity']:.3f}")
            st.metric("Interpretability", f"{metric_values['interpretability']:.3f}")
        with met3:
            st.metric("Governance Score", f"{metric_values['governance']:.3f}")
            st.metric("HME-G Score", f"{hmeg_score:.3f}")
            st.metric("Decision", decision)

        st.subheader("Adaptive Weights")
        st.json(adjusted_weights)

        st.subheader("Governance Audit")
        if len(flags) == 0:
            st.success("No governance flags raised.")
        else:
            for flag in flags:
                st.warning(flag)

        st.subheader("Audit Trail")
        st.write({
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "human_trust": human_trust,
            "human_clarity": human_clarity,
            "clinical_alignment": clinical_alignment,
            "reviewer_comment": reviewer_comment,
            "governance_flags": flags,
            "decision": decision
        })

st.markdown("---")
st.markdown("Developed for Explainable AI in Medical Diagnosis with HME-G")
