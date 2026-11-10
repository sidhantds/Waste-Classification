import os

os.environ.setdefault("KERAS_BACKEND", "jax")  # must be set before importing keras

import streamlit as st
import numpy as np
import keras
from PIL import Image

CLASS_NAMES = ["glass", "metal", "organic", "paper", "plastic"]
CLASS_ICONS = {
    "glass": ":material/wine_bar:",
    "metal": ":material/hardware:",
    "organic": ":material/compost:",
    "paper": ":material/description:",
    "plastic": ":material/water_bottle:",
}

st.set_page_config(page_title="Waste classifier", page_icon="♻️", layout="wide")


@st.cache_resource
def load_model():
    return keras.models.load_model("waste_mobilenetv2.keras")


def predict(image: Image.Image) -> np.ndarray:
    resized = image.resize((224, 224))
    img_batch = np.expand_dims(np.array(resized, dtype="float32"), 0)
    return load_model().predict(img_batch, verbose=0)[0]


st.title("♻️ Waste classifier")
st.caption("Upload a photo of waste and the AI will tell you what type it is.")

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader(":material/upload: Upload image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        image = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Your image", width="stretch")

with right:
    with st.container(border=True):
        st.subheader(":material/analytics: Results")
        if image is None:
            st.info("Upload an image on the left to see the prediction.", icon=":material/arrow_back:")
        else:
            with st.spinner("Classifying..."):
                prediction = predict(image)

            best = int(np.argmax(prediction))
            predicted_class = CLASS_NAMES[best]
            confidence = float(prediction[best]) * 100

            st.success(
                f"**Prediction: {predicted_class.upper()}**",
                icon=CLASS_ICONS[predicted_class],
            )
            st.metric("Confidence", f"{confidence:.1f}%")

            st.markdown("**All class probabilities**")
            for idx in np.argsort(prediction)[::-1]:
                name, prob = CLASS_NAMES[idx], float(prediction[idx])
                st.progress(prob, text=f"{CLASS_ICONS[name]} {name.title()} — {prob * 100:.1f}%")
