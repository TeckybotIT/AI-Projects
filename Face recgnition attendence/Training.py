import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Paths and Constants
IMG_SIZE = 50  # Image size for CNN
DATA_PATH = "static/augmented_faces"
MODEL_PATH = "static/models/face_recognition_model.h5"

# Load and augment face data
def load_and_augment_images(data_path):
    faces = []
    labels = []
    userlist = os.listdir(data_path)

    for user in userlist:
        user_folder = os.path.join(data_path, user)
        if os.path.isdir(user_folder):
            for imgname in os.listdir(user_folder):
                if imgname.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(user_folder, imgname)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        print(f"⚠️ Skipping invalid image: {img_path}")
                        continue

                    # Resize and normalize
                    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img_normalized = img_resized / 255.0

                    # Add original image
                    faces.append(img_normalized)
                    labels.append(user)

                    # Data Augmentation (Zoom and Rotate)
                    zoomed_img = cv2.resize(img_resized[5:-5, 5:-5], (IMG_SIZE, IMG_SIZE))
                    faces.append(zoomed_img / 255.0)
                    labels.append(user)

                    rotated_img = cv2.rotate(img_resized, cv2.ROTATE_90_CLOCKWISE)
                    faces.append(rotated_img / 255.0)
                    labels.append(user)

    faces = np.array(faces).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    labels = np.array(labels)
    return faces, labels, userlist


# Create CNN model
def create_cnn_model(num_classes):
    model = Sequential()
    
    # Convolutional and Pooling Layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Reduce overfitting
    model.add(Dense(num_classes, activation='softmax'))

    # Compile Model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Main training function
def train_model():
    faces, labels, userlist = load_and_augment_images(DATA_PATH)
    
    # Encode labels and convert to categorical
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    y_categorical = to_categorical(labels_encoded)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(faces, y_categorical, test_size=0.2, random_state=42)

    # Create and train the model
    model = create_cnn_model(num_classes=len(userlist))
    model.fit(X_train, y_train, epochs=20, validation_split=0.2, batch_size=32)

    # Evaluate model accuracy
    train_acc = model.evaluate(X_train, y_train)[1] * 100
    test_acc = model.evaluate(X_test, y_test)[1] * 100
    print(f"✅ Model trained successfully!")
    print(f"🎯 Training Accuracy: {train_acc:.2f}%")
    print(f"🧪 Testing Accuracy: {test_acc:.2f}%")

    # Save the trained model
    model.save(MODEL_PATH)
    print(f"📚 Model saved at {MODEL_PATH}")


# Run training
if __name__ == "__main__":
    train_model()
