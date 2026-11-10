import os

os.environ.setdefault("KERAS_BACKEND", "jax")  # must be set before importing keras

import streamlit as st
import numpy as np
import keras
from PIL import Image

class_names = ["glass", "metal", "organic", "paper", "plastic"]


@st.cache_resource
def load_model():
    return keras.models.load_model("waste_mobilenetv2.keras")


model = load_model()

st.title("♻️ Waste Classifier")
st.write("Upload a photo of waste and the AI will tell you what type it is.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Your image", width=300)

    resized = image.resize((224, 224))
    img_array = np.array(resized, dtype="float32")
    img_batch = np.expand_dims(img_array, 0)

    prediction = model.predict(img_batch)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class.upper()} ({confidence:.1f}% confident)")

    st.subheader("All class probabilities")
    for name, prob in zip(class_names, prediction):
        st.write(f"{name}: {prob * 100:.1f}%")
        st.progress(float(prob))