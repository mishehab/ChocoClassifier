import io
import time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Chocolate Classifier", page_icon="🍫", layout="centered")

IMG_SIZE = (224, 224)
CONF_WARN = 0.55  

@st.cache_resource
def load_model(path="chocolate_classifier.keras"):
    return tf.keras.models.load_model(path)

@st.cache_data
def load_label_and_meta(csv_path="label.csv"):
    """
    Supports two formats:
      A) Mapping CSV: two columns [index,label] (exported from training)
      B) Per-image CSV: columns [filename,label,price,manufacturer,calories]
    Returns:
      idx_to_label: dict[int -> str]
      meta_by_label: dict[label -> {price, manufacturer, calories}]
    """
    df = pd.read_csv(csv_path)

    
    if set(df.columns[:2]) >= {"index", "label"} and len(df.columns) <= 3:
        df = df.dropna(subset=["index", "label"])
        df["index"] = df["index"].astype(int)
        idx_to_label = dict(zip(df["index"], df["label"].astype(str)))
        meta_by_label = {}
    else:
        needed = {"filename", "label", "price", "manufacturer", "calories"}
        if not needed.issubset(set(df.columns)):
            raise ValueError(
                "CSV must be either (index,label) mapping or per-image with "
                "columns: filename,label,price,manufacturer,calories"
            )
        meta_by_label = (
            df.drop_duplicates(subset=["label"])
              .set_index("label")[["price", "manufacturer", "calories"]]
              .to_dict(orient="index")
        )
        labels = sorted(meta_by_label.keys())
        idx_to_label = {i: lbl for i, lbl in enumerate(labels)}

    return idx_to_label, meta_by_label


def preprocess(img_pil: Image.Image) -> np.ndarray:
    img = img_pil.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = (np.array(img) / 255.0).astype("float32")
    return arr

def predict_one(model, img_pil: Image.Image, idx_to_label: dict):
    x = preprocess(img_pil)[None, ...]
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = idx_to_label.get(pred_idx, f"Class_{pred_idx}")
    pred_conf = float(probs[pred_idx])

    
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(idx_to_label.get(int(i), f"Class_{int(i)}"), float(probs[i])) for i in top3_idx]
    return pred_label, pred_conf, top3


st.title("🍫 Live Chocolate Classifier")
st.write("Capture a photo of a chocolate bar (phone or webcam). The app predicts the category and shows **price (BDT)**, **manufacturer**, and **calories**.")

with st.sidebar:
    st.subheader("Settings")
    show_top3 = st.toggle("Show top-3 predictions", value=True)
    st.caption("Tip: ensure good lighting, hold the bar steady, and fill the frame.")


model = load_model()
idx_to_label, meta_by_label = load_label_and_meta("class_index_to_label.csv")

tabs = st.tabs(["📷 Camera", "📁 Upload"])
image = None

with tabs[0]:
    cap = st.camera_input("Take a photo")
    if cap is not None:
        image = Image.open(cap)

with tabs[1]:
    up = st.file_uploader("Or upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if up is not None:
        image = Image.open(up)

if image is None:
    st.info("Use **Camera** to capture or **Upload** an image to get started.")
    st.stop()

st.image(image, caption="Input image", use_container_width=True)

with st.spinner("Predicting…"):
    t0 = time.time()
    label, conf, top3 = predict_one(model, image, idx_to_label)
    latency = (time.time() - t0) * 1000

st.success(f"**Prediction:** {label}  \n**Confidence:** {conf*100:.2f}%  \n_latency: {latency:.1f} ms_")

info = meta_by_label.get(label)
col1, col2, col3 = st.columns(3)
if info:
    col1.metric("Price (BDT)", info.get("price", "N/A"))
    col2.metric("Manufacturer", info.get("manufacturer", "N/A"))
    col3.metric("Calories", info.get("calories", "N/A"))
else:
    col1.metric("Price (BDT)", "N/A")
    col2.metric("Manufacturer", "N/A")
    col3.metric("Calories", "N/A")
    st.caption("No per-label metadata found in CSV. If you used an index→label mapping file, add a per-image CSV or a separate labels.csv and adapt loader.")

if conf < CONF_WARN:
    st.warning("Low confidence. Try better lighting, move closer, or reduce glare.")

if show_top3:
    st.subheader("Top-3 predictions")
    for i, (lbl, p) in enumerate(top3, start=1):
        st.write(f"{i}. **{lbl}** — {p*100:.2f}%")

st.caption("Model: MobileNetV2 • Input 224×224 • Streamlit camera • TensorFlow `.keras`")
