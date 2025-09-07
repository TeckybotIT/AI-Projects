import os
import cv2

def remove_corrupted_images(base_path):
    userlist = os.listdir(base_path)
    
    for user in userlist:
        user_path = f"{base_path}/{user}"
        for imgname in os.listdir(user_path):
            img_path = f"{user_path}/{imgname}"
            
            # Read the image to check if it's valid
            img = cv2.imread(img_path)
            if img is None or img.size == 0:
                print(f"⚠️ Removing corrupted image: {img_path}")
                os.remove(img_path)

# Run the function
remove_corrupted_images("static/faces")
