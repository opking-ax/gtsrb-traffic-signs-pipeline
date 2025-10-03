import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from src.data.preprocess import prepare_datasets
import pandas as pd
import os


def preprocess_uploaded_image(uploaded_file, target_size=(32, 32)):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, image, class_names):
    probs = model.predict(image)
    class_idx = np.argmax(probs, axis=1)
    pred_class = class_names[class_idx]
    return pred_class, probs


def display_prediction_results(pred_class, probs, class_names, top_k=5):
    st.write("Predicted Class: ", pred_class)

    top_indices = np.argsort(probs)[-top_k:][::-1]
    top_probs = probs[top_indices]
    top_labels = [class_names[i] for i in top_indices]

    df = pd.DataFrame({"Class": top_labels, "Probability": top_probs})
    st.bar_chart(df.set_index("Class"))


def display_saved_plots():
    st.subheader("Evaluation Reports")
    cm_path = os.path.join("reports", "confusion_matrix.png")
    dist_path = os.path.join("reports", "dataset_distribution.png")


def main():
    st.title("GTSRB Traffic Sign Classifier")

    uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jgp", "png"])

    model = load_model("../models/cnn_grapevine.h5")
    class_names = sorted(os.listdir("../data"))

    if uploaded_file:
        image = preprocess_uploaded_image(uploaded_file)
        pred_class, probs = predict_image(model, image, class_names)
        display_prediction_results(pred_class, probs, class_names)

    st.subheader("Model Evaluation")
    display_saved_plots()


if __name__ == "__main__":
    main()