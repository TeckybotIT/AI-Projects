import cv2
import os
import numpy as np
import random

# Base folder where images are stored
base_path = "static/faces"

# Output folder for augmented images
augmented_path = "static/augmented_faces"

# Create augmented folder if it doesn't exist
if not os.path.exists(augmented_path):
    os.makedirs(augmented_path)

# Function to apply augmentation and save images
def augment_and_save(img, output_path, user, imgname, count):
    rows, cols = img.shape[:2]
    
    # ✅ Rotation (Random -5 to +5 degrees)
    angle = random.randint(-5, 5)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(img, M, (cols, rows))
    cv2.imwrite(f"{output_path}/{user}/{imgname}_rotated_{count}.jpeg", rotated)
    
    # 🔍 Zooming (Random 5% zoom)
    zoom_factor = random.uniform(1.1, 1.2)
    zoomed = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)
    h, w = img.shape[:2]
    crop_h, crop_w = int(h * 0.9), int(w * 0.9)
    zoomed_cropped = zoomed[:crop_h, :crop_w]
    zoomed_resized = cv2.resize(zoomed_cropped, (cols, rows))
    cv2.imwrite(f"{output_path}/{user}/{imgname}_zoomed_{count}.jpeg", zoomed_resized)
    
    # ➡️ Translation (Shift by 5 pixels)
    tx, ty = random.randint(-5, 5), random.randint(-5, 5)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, M_trans, (cols, rows))
    cv2.imwrite(f"{output_path}/{user}/{imgname}_translated_{count}.jpeg", translated)
    
    # 🔁 Flipping (Horizontal flip)
    flipped = cv2.flip(img, 1)
    cv2.imwrite(f"{output_path}/{user}/{imgname}_flipped_{count}.jpeg", flipped)
    
    # 💡 Brightness Adjustment
    bright_factor = random.uniform(0.8, 1.2)
    brightened = np.clip(img * bright_factor, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{output_path}/{user}/{imgname}_bright_{count}.jpeg", brightened)

# Main function to apply augmentation for all users
def augment_faces(base_path, augmented_path):
    userlist = os.listdir(base_path)
    
    for user in userlist:
        user_path = f"{base_path}/{user}"
        aug_user_path = f"{augmented_path}/{user}"
        
        # Create user folder in augmented_path if not exists
        if not os.path.exists(aug_user_path):
            os.makedirs(aug_user_path)
        
        for imgname in os.listdir(user_path):
            img_path = f"{user_path}/{imgname}"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Skip invalid images
            if img is None or img.size == 0:
                continue
            
            # Apply 5 augmented versions for each image
            for i in range(5):
                augment_and_save(img, augmented_path, user, imgname.split('.')[0], i)

# Run augmentation
augment_faces(base_path, augmented_path)

print("✅ Augmentation completed! New images saved in 'static/augmented_faces'")
