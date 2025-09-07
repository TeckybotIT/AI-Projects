import os
import joblib
import numpy as np
from typing import Dict, Any, List

HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.normpath(os.path.join(HERE, "..", "models"))

CROP_MODEL_PATH = os.path.join(MODELS_DIR, "crop_recommender.pkl")
CROP_SCALER_PATH = os.path.join(MODELS_DIR, "crop_scaler.pkl")

if not os.path.exists(CROP_MODEL_PATH) or not os.path.exists(CROP_SCALER_PATH):
    raise FileNotFoundError(f"Crop model or scaler missing in {MODELS_DIR}")

crop_model = joblib.load(CROP_MODEL_PATH)
crop_scaler = joblib.load(CROP_SCALER_PATH)

SOIL_TYPES: List[str] = [
    "Red and Yellow soils",
    "Alluvial soils",
    "Laterite soils",
    "Black soils",
]

CROP_BASE_MM_PER_WEEK: Dict[str, float] = {
    "apple": 35, "banana": 55, "grapes": 40, "mango": 45, "orange": 40,
    "papaya": 55, "pomegranate": 45, "muskmelon": 35, "watermelon": 40,
    "coconut": 60, "coffee": 55, "cotton": 45, "jute": 50,
    "rice": 65, "maize": 45,
    "blackgram": 30, "chickpea": 30, "kidneybeans": 32, "lentil": 28,
    "mothbeans": 26, "mungbean": 26, "pigeonpeas": 35,
}

SOIL_MULTIPLIER: Dict[str, float] = {
    "Red and Yellow soils": 1.10,
    "Alluvial soils": 1.00,
    "Laterite soils": 1.15,
    "Black soils": 0.85,
}

TEMP_INCREASE_PER_C_ABOVE_30 = 0.03
HUMIDITY_REDUCTION_PER_PC_ABOVE_60 = 0.01
MAX_HUMIDITY_REDUCTION = 0.25  # max 25% reduction


def predict_crop_from_features(
    N: float, P: float, K: float,
    temperature: float, humidity: float,
    ph: float, rainfall: float
) -> str:
    arr = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
    arr_scaled = crop_scaler.transform(arr)
    pred = crop_model.predict(arr_scaled)[0]
    return str(pred)


def calculate_irrigation_mm_per_week(
    crop_label: str,
    soil_type: str,
    temperature_c: float,
    humidity_pc: float,
    rainfall_mm_week: float
) -> Dict[str, Any]:
    crop = str(crop_label).lower()
    base = CROP_BASE_MM_PER_WEEK.get(crop, 40.0)
    soil_factor = SOIL_MULTIPLIER.get(soil_type, 1.0)

    temp_factor = 1.0 + max(temperature_c - 30.0, 0) * TEMP_INCREASE_PER_C_ABOVE_30

    if humidity_pc > 60.0:
        reduction = min(
            (humidity_pc - 60.0) * HUMIDITY_REDUCTION_PER_PC_ABOVE_60,
            MAX_HUMIDITY_REDUCTION
        )
        humidity_factor = 1.0 - reduction
    else:
        humidity_factor = 1.0

    gross_need = base * soil_factor * temp_factor * humidity_factor
    effective_rain = min(rainfall_mm_week * 0.8, gross_need)
    irrigation = max(gross_need - effective_rain, 0.0)

    if irrigation < 20:
        category = "Low"
    elif irrigation < 40:
        category = "Medium"
    elif irrigation < 60:
        category = "High"
    else:
        category = "Very High"

    notes = []
    if soil_type == "Laterite soils":
        notes.append("Porous soil – consider more frequent, lighter irrigations.")
    if soil_type == "Black soils":
        notes.append("Cracking clay – use deeper but less frequent irrigations.")
    if temperature_c > 35.0:
        notes.append("High evapotranspiration risk (>35°C). Monitor more often.")
    if humidity_pc > 80.0:
        notes.append("High humidity – reduce irrigation slightly to avoid disease.")

    return {
        "crop": crop,
        "soil_type": soil_type,
        "base_mm_week": round(base, 2),
        "soil_factor": round(soil_factor, 2),
        "temp_factor": round(temp_factor, 2),
        "humidity_factor": round(humidity_factor, 2),
        "effective_rain_mm": round(effective_rain, 2),
        "irrigation_mm_week": round(irrigation, 2),
        "category": category,
        "notes": notes,
    }


def irrigation_pipeline(
    soil_type: str,
    N: float,
    P: float,
    K: float,
    temperature: float,
    humidity: float,
    ph: float,
    rainfall_mm_week: float,
) -> Dict[str, Any]:
    if soil_type not in SOIL_TYPES:
        raise ValueError(f"Soil type '{soil_type}' not in valid types: {SOIL_TYPES}")

    predicted_crop = predict_crop_from_features(N, P, K, temperature, humidity, ph, rainfall_mm_week)
    irri = calculate_irrigation_mm_per_week(
        crop_label=predicted_crop,
        soil_type=soil_type,
        temperature_c=temperature,
        humidity_pc=humidity,
        rainfall_mm_week=rainfall_mm_week
    )
    return {
        "Predicted Crop": predicted_crop,
        "Soil Type": soil_type,
        "Irrigation (mm/week)": irri["irrigation_mm_week"],
        "Category": irri["category"],
        "Breakdown": {
            "Base crop need (mm/wk)": irri["base_mm_week"],
            "Soil factor": irri["soil_factor"],
            "Temp factor": irri["temp_factor"],
            "Humidity factor": irri["humidity_factor"],
            "Effective rainfall (mm/wk)": irri["effective_rain_mm"],
        },
        "Notes": irri["notes"],
    }


# Optional CLI testing mechanism
if __name__ == "__main__":
    example = irrigation_pipeline(
        soil_type="Alluvial soils",
        N=90, P=42, K=43,
        temperature=32, humidity=65,
        ph=6.5, rainfall_mm_week=30
    )
    print("Example pipeline output:\n", example)
