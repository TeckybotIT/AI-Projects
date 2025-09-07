# Import necessary libraries
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import json
from PIL import Image

# Load disease information from JSON file
@st.cache_data
def load_disease_info():
    """Load disease descriptions and solutions from JSON file."""
    with open('disease_info.json', 'r') as file:
        disease_info = json.load(file)
    return disease_info

# Load class indices from JSON
@st.cache_data
def load_class_indices():
    """Load class indices from JSON and reverse the dictionary for prediction."""
    with open('class_indices.json', 'r') as file:
        class_indices = json.load(file)
    return {v: k for k, v in class_indices.items()}  # Reverse to get class names

# Load the trained model
@st.cache_resource
def load_model():
    """Load the pre-trained crop disease detection model."""
    model = tf.keras.models.load_model('crop_disease_model.h5')
    return model

# Preprocess the image for prediction
def preprocess_image(image):
    """Resize, normalize, and preprocess the image for model prediction."""
    image = np.array(image)
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Main Streamlit app
def main():
    st.set_page_config(page_title="🌱 Crop Disease Detection", layout="wide")

    st.title("🌾 Crop Disease Detection & Recommendation System")
    st.write("Upload a leaf image to identify the disease and get a solution.")

    # Load disease information, class indices, and model
    disease_info = load_disease_info()
    class_names = load_class_indices()
    model = load_model()

    # File uploader for image
    uploaded_file = st.file_uploader("📸 Upload an image of the leaf", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image (resized for smaller display)
        image = Image.open(uploaded_file)
        resized_image = image.resize((300, 300))  # Resize image for display
        st.image(resized_image, caption="Uploaded Image (Resized)", use_column_width=False)

        # Preprocess and predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        class_index = np.argmax(prediction)  # Get the index of highest confidence

        # Get predicted class name dynamically
        predicted_class = class_names[class_index]
        st.success(f"✅ Prediction: {predicted_class.replace('_', ' ')}")

        # Display disease information
        if predicted_class in disease_info:
            st.subheader(f"🦠 Disease: {predicted_class.replace('_', ' ')}")
            st.write(f"**📝 Description:** {disease_info[predicted_class]['description']}")

            # Display solution as bullet points if multiple steps
            solution = disease_info[predicted_class]['solution']
            st.write("✅ **Solution:**")
            if isinstance(solution, list):
                for step in solution:
                    st.markdown(f"- {step}")
            else:
                st.markdown(f"- {solution}")
        else:
            st.warning("⚠️ No information available for this disease.")

    # Add a sidebar for additional information
    st.sidebar.header("ℹ️ About")
    st.sidebar.info(
        "This app uses a deep learning model to identify common crop diseases from images "
        "and provides a description and solution based on the predicted disease."
    )

# Run the app
if __name__ == "__main__":
    main()
