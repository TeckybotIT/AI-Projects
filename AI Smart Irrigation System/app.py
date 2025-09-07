from flask import Flask, render_template, request, redirect
import os
import pandas as pd
import numpy as np
import joblib
import json
from src.irrigation_calculator import irrigation_pipeline, predict_crop_from_features

app = Flask(__name__)

# Directory where model files are stored
MODELS_DIR = "models"

feature_columns = None
crop_model = None
crop_scaler = None
model = None
label_encoders = None
scaler = None

try:
    feature_columns_path = os.path.join(MODELS_DIR, "soil_feature_columns.json")
    with open(feature_columns_path, "r") as f:
        feature_columns = json.load(f)

    crop_model = joblib.load(os.path.join(MODELS_DIR, "crop_recommender.pkl"))
    crop_scaler = joblib.load(os.path.join(MODELS_DIR, "crop_scaler.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "soil_recommendation_model.pkl"))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, "soil_label_encoders.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "soil_scaler.pkl"))
    
    print("✅ All models and features loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models or features: {e}")
    # Optionally stop app startup if models or features not loaded:
    raise


@app.route('/')
def index():
    return render_template('index.html')

# Redirect /crop to /predict_crop to avoid 404 error
@app.route('/crop')
def crop_redirect():
    return redirect('/predict_crop')

@app.route('/predict_crop', methods=['GET', 'POST'])
def predict_crop():
    if request.method == 'POST':
        try:
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])
            predicted_crop = predict_crop_from_features(N, P, K, temperature, humidity, ph, rainfall)
            return render_template('crop.html', result=predicted_crop)
        except Exception as e:
            return render_template('crop.html', result=f"Error: {str(e)}")
    return render_template('crop.html', result=None)


# PREDICTION FUNCTION
def predict_best_soil(input_data: dict):
    # Convert to dataframe
    df_input = pd.DataFrame([input_data], columns=feature_columns)

    # Encode categorical features
    if "Crop_Type" in df_input.columns:
        le = label_encoders["Crop_Type"]
        df_input["Crop_Type"] = le.transform(df_input["Crop_Type"])

    # Scale features
    X_scaled = scaler.transform(df_input)

    # Predict soil
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]

    soil_label = label_encoders["Soil_Type"].inverse_transform([prediction])[0]

    return {
        "Best_Soil": soil_label,
        "Probabilities": {
            soil: float(p)
            for soil, p in zip(label_encoders["Soil_Type"].classes_, prob)
        }
    }

@app.route('/soil', methods=['GET', 'POST'])
def soil():
    if request.method == 'POST':
        try:
            # Collect form inputs
            input_data = {
                "Crop_Type": request.form["Crop_Type"],
                "Farm_Size_Acres": float(request.form["Farm_Size_Acres"]),
                "Irrigation_Available": int(request.form["Irrigation_Available"]),
                "Soil_pH": float(request.form["Soil_pH"]),
                "Soil_Nitrogen": float(request.form["Soil_Nitrogen"]),
                "Soil_Organic_Matter": float(request.form["Soil_Organic_Matter"]),
                "Temperature": float(request.form["Temperature"]),
                "Rainfall": float(request.form["Rainfall"]),
                "Humidity": float(request.form["Humidity"]),
                "Compatible": int(request.form.get("Compatible", 1))
            }

            # Call the prediction function
            result = predict_best_soil(input_data)

            return render_template('soil.html', result=result)

        except Exception as e:
            result = {
                "Best_Soil": None,
                "Probabilities": {},
                "Error": str(e)
            }
            return render_template('soil.html', result=result)

    return render_template('soil.html', result=None)

@app.route('/irrigation', methods=['GET', 'POST'])
def irrigation():
    if request.method == 'POST':
        try:
            soil_type = request.form['soil_type']
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall_mm_week = float(request.form['rainfall'])
            area = float(request.form['area'])
            
            # Get optional inputs with defaults
            row_spacing_m = float(request.form.get('row_spacing_m', 0) or 0)
            plant_spacing_m = float(request.form.get('plant_spacing_m', 0) or 0)
            area_unit = request.form.get('area_unit', 'ha')

            result = irrigation_pipeline(
                soil_type=soil_type,
                N=N, P=P, K=K,
                temperature=temperature,
                humidity=humidity,
                ph=ph,
                rainfall_mm_week=rainfall_mm_week,
                land_area=area,
                area_unit=area_unit,
                row_spacing_m=row_spacing_m,
                plant_spacing_m=plant_spacing_m,
            )
            return render_template('irrigation.html', result=result)
        except Exception as e:
            return render_template('irrigation.html', result=f"Error: {str(e)}")
    return render_template('irrigation.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)
