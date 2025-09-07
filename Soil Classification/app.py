from flask import Flask, render_template, request
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from tensorflow.keras.preprocessing import image # type: ignore
import pickle
import os

app = Flask(__name__)

# Load trained model
model = load_model("soil_model.h5")

# Load class labels from pickle file
with open("class_indices.pkl", "rb") as f:
    class_indices = pickle.load(f)

# Invert dictionary to get index -> label mapping
labels = {v: k for k, v in class_indices.items()}

# Crop suggestions for each soil type
crop_suggestions = {
    "Alluvial_Soil": ["Wheat", "Rice", "Sugarcane", "Cotton", "Jute"],
    "Arid_Soil": ["Barley", "Millets", "Pulses", "Maize"],
    "Black_Soil": ["Cotton", "Soybean", "Sunflower", "Wheat"],
    "Laterite_Soil": ["Tea", "Coffee", "Cashew", "Rubber"],
    "Mountain_Soil": ["Tea", "Coffee", "Spices", "Apples"],
    "Red_Soil": ["Groundnut", "Millets", "Pulses", "Rice"],
    "Yellow_Soil": ["Paddy", "Sugarcane", "Groundnut"]
}

# Regions for each soil type
soil_regions = {
    "Alluvial_Soil": "Ganges plain, Punjab, Haryana",
    "Arid_Soil": "Rajasthan, Gujarat",
    "Black_Soil": "Maharashtra, Madhya Pradesh, Andhra Pradesh",
    "Laterite_Soil": "Kerala, Karnataka, Tamil Nadu",
    "Mountain_Soil": "Himalayas, North-Eastern states",
    "Red_Soil": "Tamil Nadu, Karnataka, Andhra Pradesh",
    "Yellow_Soil": "Coastal areas of Odisha and Andhra Pradesh"
}

# About soil info
soil_info_dict = {
    "Alluvial_Soil": "Alluvial soil is fertile, suitable for crops like wheat, rice, and sugarcane.",
    "Arid_Soil": "Arid soil is sandy, low in moisture, suitable for drought-resistant crops like millets and barley.",
    "Black_Soil": "Black soil is rich in clay and retains moisture, ideal for cotton and soybean.",
    "Laterite_Soil": "Laterite soil is acidic and rich in iron, suitable for tea, coffee, and cashew.",
    "Mountain_Soil": "Mountain soil is found in hilly areas, good for tea, coffee, and spices.",
    "Red_Soil": "Red soil is rich in iron, supports crops like groundnut, millets, and pulses.",
    "Yellow_Soil": "Yellow soil is rich in nutrients, suitable for paddy, sugarcane, and groundnut."
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # Save uploaded file
    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128,128))  # adjust if your input size is different
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_label = labels[predicted_class_index]

    # Suggest crops
    # Suggest crops, regions, and soil info
    crops = crop_suggestions.get(predicted_label, ["No crop suggestion available"])
    regions = soil_regions.get(predicted_label, "Region info not available")
    soil_info = soil_info_dict.get(predicted_label, "Soil information not available")

    return render_template("result.html",
                           soil_type=predicted_label,
                           crops=crops,
                           soil_regions=regions,
                            soil_info=soil_info,
                           image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)
