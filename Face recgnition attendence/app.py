import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
import os
from datetime import datetime

# Paths and Constants
MODEL_PATH = "static/models/face_recognition_model.h5"
IMG_SIZE = 50
DATA_PATH = "static/faces"
ATTENDANCE_FILE = "attendance.csv"

# Load CNN model
@st.cache_resource
def load_face_model():
    return load_model(MODEL_PATH)


# Load user names from folder names
def load_user_names(data_path):
    return os.listdir(data_path)


# Predict face using CNN
def predict_face(face, model, user_names):
    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face_reshaped = face_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0

    pred = model.predict(face_reshaped)
    pred_index = np.argmax(pred)
    return user_names[pred_index]


# Save attendance to CSV
def mark_attendance(user_name):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")

    # Check if file exists
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["User Name", "Date & Time"])
        df.to_csv(ATTENDANCE_FILE, index=False)

    # Read existing data
    df = pd.read_csv(ATTENDANCE_FILE)

    # Add new record
    new_record = pd.DataFrame([[user_name, current_time]], columns=["User Name", "Date & Time"])
    df = pd.concat([df, new_record], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)


# Streamlit UI
st.title("📸 Face Recognition Attendance System")
st.sidebar.title("⚙️ Options")

# Load model and user names
face_model = load_face_model()
user_names = load_user_names(DATA_PATH)

# Option to use camera
use_camera = st.sidebar.checkbox("🎥 Use Camera for Face Detection")

if use_camera:
    # Start webcam
    st.write("📷 Starting Webcam...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Unable to access camera!")
    else:
        st.write("✅ Camera Connected")
        start_button = st.button("Capture Image")

        # Capture image when button is clicked
        if start_button:
            ret, frame = cap.read()
            if ret:
                gray_face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                st.image(frame, caption="Captured Image", use_column_width=True)

                # Predict face
                user_name = predict_face(gray_face, face_model, user_names)
                st.success(f"🎯 Recognized User: {user_name}")

                # Mark attendance
                mark_attendance(user_name)
                st.success("✅ Attendance Marked Successfully!")
            else:
                st.error("❗ Failed to capture image. Try again.")
        cap.release()
else:
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        face = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if face is not None:
            st.image(face, caption="Uploaded Image", use_column_width=True)

            # Predict face
            user_name = predict_face(face, face_model, user_names)
            st.success(f"🎯 Recognized User: {user_name}")

            # Mark attendance
            mark_attendance(user_name)
            st.success("✅ Attendance Marked Successfully!")
        else:
            st.error("❗ Invalid image format. Try again!")

# Download attendance as CSV
st.sidebar.title("📥 Download Attendance")
if st.sidebar.button("Download Attendance CSV"):
    with open(ATTENDANCE_FILE, "rb") as file:
        st.sidebar.download_button(
            label="📥 Download Attendance",
            data=file,
            file_name="attendance.csv",
            mime="text/csv",
        )
