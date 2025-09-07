# irrigation.py
# ------------------------------------------------------------
# Rules-based irrigation calculator integrated with your crop
# recommender (joblib). Adds water-need conversions and options.
# ------------------------------------------------------------

import os
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np

# ================================
# Paths / Model Loading
# ================================
HERE = os.path.dirname(__file__)
MODELS_DIR = os.path.normpath(os.path.join(HERE, "..", "models"))

CROP_MODEL_PATH = os.path.join(MODELS_DIR, "crop_recommender.pkl")
CROP_SCALER_PATH = os.path.join(MODELS_DIR, "crop_scaler.pkl")

if not os.path.exists(CROP_MODEL_PATH) or not os.path.exists(CROP_SCALER_PATH):
    raise FileNotFoundError(f"Crop model or scaler missing in {MODELS_DIR}")

crop_model = joblib.load(CROP_MODEL_PATH)
crop_scaler = joblib.load(CROP_SCALER_PATH)

# ================================
# Constants
# ================================
SOIL_TYPES: List[str] = [
    "Red and Yellow soils",
    "Alluvial soils",
    "Laterite soils",
    "Black soils",
]

# Base weekly crop water requirement in mm/week (before adjustments)
CROP_BASE_MM_PER_WEEK: Dict[str, float] = {
    "apple": 35, "banana": 55, "grapes": 40, "mango": 45, "orange": 40,
    "papaya": 55, "pomegranate": 45, "muskmelon": 35, "watermelon": 40,
    "coconut": 60, "coffee": 55, "cotton": 45, "jute": 50,
    "rice": 65, "maize": 45,
    "blackgram": 30, "chickpea": 30, "kidneybeans": 32, "lentil": 28,
    "mothbeans": 26, "mungbean": 26, "pigeonpeas": 35,
}

# Soil water retention / percolation multipliers
SOIL_MULTIPLIER: Dict[str, float] = {
    "Red and Yellow soils": 1.10,
    "Alluvial soils": 1.00,
    "Laterite soils": 1.15,
    "Black soils": 0.85,
}

# Climate adjustment factors
TEMP_INCREASE_PER_C_ABOVE_30 = 0.03     # +3% per °C above 30
HUMIDITY_REDUCTION_PER_PC_ABOVE_60 = 0.01  # -1% per %RH above 60
MAX_HUMIDITY_REDUCTION = 0.25           # cap at 25% reduction

# Conversions
LITERS_PER_M3 = 1000.0
M3_PER_HA_PER_MM = 10.0        # 1 mm over 1 ha = 10 m³
HECTARES_PER_ACRE = 0.40468564224

# ================================
# Utility helpers
# ================================
def _area_to_hectares(area: float, unit: str) -> float:
    """Convert area to hectares."""
    unit = (unit or "ha").strip().lower()
    if unit in ("ha", "hectare", "hectares"):
        return float(area)
    if unit in ("acre", "acres"):
        return float(area) * HECTARES_PER_ACRE
    raise ValueError("area_unit must be 'ha' or 'acre'")

def _mm_to_m3_per_ha(mm: float) -> float:
    return mm * M3_PER_HA_PER_MM

def _schedule_suggestion(category: str, soil_type: str) -> Tuple[str, str]:
    """
    Return (frequency, method) suggestion based on category & soil.
    """
    # Base frequency by category
    if category == "Low":
        freq = "every 7-10 days"
    elif category == "Medium":
        freq = "every 4-6 days"
    elif category == "High":
        freq = "every 2-3 days"
    else:  # Very High
        freq = "daily or every 1-2 days"

    # Method hint by soil
    if soil_type == "Laterite soils":
        method = "frequent, lighter irrigations (sprinkler/drip preferred)."
    elif soil_type == "Black soils":
        method = "deeper but less frequent irrigations (avoid surface cracking)."
    else:
        method = "balanced schedule; drip or sprinkler recommended for efficiency."
    return freq, method

def _plants_per_hectare(row_spacing_m: float, plant_spacing_m: float) -> Optional[float]:
    """
    Rough estimate plants per hectare using spacing (row x plant).
    1 ha = 10,000 m²
    """
    if row_spacing_m and plant_spacing_m and row_spacing_m > 0 and plant_spacing_m > 0:
        area_per_plant = row_spacing_m * plant_spacing_m
        if area_per_plant > 0:
            return 10000.0 / area_per_plant
    return None

# ================================
# Core functions
# ================================
def predict_crop_from_features(
    N: float, P: float, K: float,
    temperature: float, humidity: float,
    ph: float, rainfall: float
) -> str:
    """
    Predict most suitable crop from soil & climate features using saved model.
    Input order must be: [N, P, K, temperature, humidity, pH, rainfall]
    """
    arr = np.array([[N, P, K, temperature, humidity, ph, rainfall]], dtype=float)
    arr_scaled = crop_scaler.transform(arr)
    pred = crop_model.predict(arr_scaled)[0]
    return str(pred)


def calculate_irrigation_mm_per_week(
    crop_label: str,
    soil_type: str,
    temperature_c: float,
    humidity_pc: float,
    rainfall_mm_week: float,
) -> Dict[str, Any]:
    """
    Rule-based weekly irrigation requirement (mm/week), adjusted for:
    - Soil water behavior
    - Temperature above 30°C
    - Humidity above 60% (reduces need)
    - Effective rainfall (assumed 80% effective)
    """
    crop = str(crop_label).lower()
    base = CROP_BASE_MM_PER_WEEK.get(crop, 40.0)  # safe default
    soil_factor = SOIL_MULTIPLIER.get(soil_type, 1.0)

    # Temperature adjustment (only above 30°C)
    temp_factor = 1.0 + max(temperature_c - 30.0, 0.0) * TEMP_INCREASE_PER_C_ABOVE_30

    # Humidity adjustment (only above 60% RH)
    if humidity_pc > 60.0:
        reduction = min((humidity_pc - 60.0) * HUMIDITY_REDUCTION_PER_PC_ABOVE_60, MAX_HUMIDITY_REDUCTION)
        humidity_factor = 1.0 - reduction
    else:
        humidity_factor = 1.0

    # Compute gross need, effective rainfall, and net irrigation
    gross_need = base * soil_factor * temp_factor * humidity_factor
    effective_rain = min(rainfall_mm_week * 0.8, gross_need)  # assume 80% efficiency
    irrigation = max(gross_need - effective_rain, 0.0)

    # Category bands
    if irrigation < 20:
        category = "Low"
    elif irrigation < 40:
        category = "Medium"
    elif irrigation < 60:
        category = "High"
    else:
        category = "Very High"

    # Notes
    notes: List[str] = []
    if soil_type == "Laterite soils":
        notes.append("Porous soil- consider more frequent, lighter irrigations.")
    if soil_type == "Black soils":
        notes.append("Cracking clay- use deeper but less frequent irrigations.")
    if temperature_c > 35.0:
        notes.append("High evapotranspiration risk (>35°C). Monitor more often.")
    if humidity_pc > 80.0:
        notes.append("High humidity - reduce irrigation slightly to avoid disease.")

    freq, method = _schedule_suggestion(category, soil_type)
    notes.append(f"Suggested schedule: {freq}; {method}")

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
    land_area: Optional[float] = None,
    area_unit: str = "ha",
    row_spacing_m: Optional[float] = None,
    plant_spacing_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Full pipeline:
      1) Predict crop from NPK + climate
      2) Compute irrigation in mm/week
      3) Convert to m³/ha, liters/ha, and total for given area (ha or acre)
      4) Optional: liters per plant (requires spacing)

    Args:
        soil_type: one of SOIL_TYPES
        land_area: optional numeric area
        area_unit: 'ha' or 'acre'
        row_spacing_m, plant_spacing_m: optional spacing (meters) to estimate per-plant water

    Returns:
        Dict with crop, soil, irrigation in mm/week, water need in m³/ha, liters/ha,
        total water for the given area (m³ and liters), and notes.
    """
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

    # Per-hectare needs
    water_m3_per_ha = _mm_to_m3_per_ha(irri["irrigation_mm_week"])
    water_liters_per_ha = water_m3_per_ha * LITERS_PER_M3

    # Totals for given area (if provided)
    total_water_m3 = None
    total_water_liters = None
    plants_per_ha = None
    liters_per_plant_per_week = None

    if land_area is not None:
        area_ha = _area_to_hectares(land_area, area_unit)
        total_water_m3 = round(water_m3_per_ha * area_ha, 2)
        total_water_liters = round(total_water_m3 * LITERS_PER_M3, 2)

        # Optional per-plant estimate (if spacing provided)
        plants_per_ha = _plants_per_hectare(row_spacing_m or 0, plant_spacing_m or 0)
        if plants_per_ha:
            liters_per_plant_per_week = round(water_liters_per_ha / plants_per_ha, 2)

    output: Dict[str, Any] = {
        "Predicted Crop": predicted_crop,
        "Soil Type": soil_type,
        "Irrigation (mm/week)": irri["irrigation_mm_week"],
        "Water Need (m³/ha)": round(water_m3_per_ha, 2),
        "Water Need (liters/ha)": round(water_liters_per_ha, 2),
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

    if land_area is not None:
        output["Area"] = {"value": land_area, "unit": area_unit}
        output["Total Water Need (m³)"] = total_water_m3
        output["Total Water Need (liters)"] = total_water_liters

    if plants_per_ha:
        output["Per-plant estimate"] = {
            "Plants per hectare": int(plants_per_ha),
            "Liters per plant per week": liters_per_plant_per_week
        }

    return output


# ================================
# CLI example
# ================================
if __name__ == "__main__":
    example = irrigation_pipeline(
        soil_type="Alluvial soils",
        N=90, P=42, K=43,
        temperature=32, humidity=65,
        ph=6.5, rainfall_mm_week=30,
        land_area=1.5, area_unit="ha",
        row_spacing_m=2.0, plant_spacing_m=1.5
    )
    import json
    print("Example pipeline output:\n", json.dumps(example, indent=2))
