# ===============================================================
# ADVANCED MULTI-CORE FUNDUS IMAGE PREPROCESSING PIPELINE
# FOR DIABETIC RETINOPATHY DETECTION (MAX QUALITY + SPEED)
# ===============================================================

import cv2
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# ===============================================================
# 1. ADVANCED PREPROCESSING FUNCTIONS (No Quality Loss)
# ===============================================================

def crop_black(img):
    """Remove unnecessary black borders."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def circular_mask(img):
    """Apply circular crop to isolate retina region."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w//2, h//2), min(h, w)//2, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)

def clahe_enhance(img):
    """Improve contrast and vessel visibility."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def gamma_correction(img, gamma=1.2):
    """Brightness correction to normalize illumination."""
    invGamma = 1.0 / gamma
    table = np.array([(i/255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def preprocess_fundus(image_path, output_folder, input_folder, target_size=512):
    """Main preprocessing pipeline for a single image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return

    try:
        # Step 1: Crop black regions
        img = crop_black(img)

        # Step 2: Resize to target dimension
        img = cv2.resize(img, (target_size, target_size))

        # Step 3: Circular crop
        img = circular_mask(img)

        # Step 4: CLAHE enhancement
        img = clahe_enhance(img)

        # Step 5: Gamma correction
        img = gamma_correction(img, 1.2)

        # Step 6: Gaussian blur subtraction (for vessel sharpening)
        blur = cv2.GaussianBlur(img, (0, 0), 40)
        img = cv2.addWeighted(img, 4, blur, -4, 128)

        # Save processed image preserving folder structure
        relative_path = os.path.relpath(image_path, input_folder)
        save_path = os.path.join(output_folder, relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)

    except Exception as e:
        print(f"⚠️ Error processing {image_path}: {e}")


# ===============================================================
# 2. PARALLEL MULTI-CORE PROCESSING FUNCTION
# ===============================================================

def process_all_images(input_folder, output_folder, num_workers=None):
    """Process all images in dataset using multiple CPU cores."""
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # leave 1 core free

    # Gather all image paths
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))

    print(f"🧠 Found {len(image_paths)} images. Using {num_workers} CPU cores...")
    os.makedirs(output_folder, exist_ok=True)

    # Run preprocessing in parallel
    with Pool(processes=num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(
                partial(preprocess_fundus, output_folder=output_folder, input_folder=input_folder),
                image_paths
            ),
            total=len(image_paths),
            desc="🚀 Preprocessing Images"
        ))

    print(f"✅ All {len(image_paths)} images preprocessed successfully and saved to: {output_folder}")


# ===============================================================
# 3. RUN PIPELINE – for your Kaggle DR dataset
# ===============================================================

if __name__ == "__main__":
    # TRAIN SET
    input_folder_train = "C:/Users/kondk/Downloads/archive (2)/split_dataset/train"
    output_folder_train = "C:/Users/kondk/Downloads/archive (2)/split_dataset_processed/train"
    process_all_images(input_folder_train, output_folder_train)

    # VALIDATION SET
    input_folder_val = "C:/Users/kondk/Downloads/archive (2)/split_dataset/val"
    output_folder_val = "C:/Users/kondk/Downloads/archive (2)/split_dataset_processed/val"
    process_all_images(input_folder_val, output_folder_val)

    # TEST SET
    input_folder_test = "C:/Users/kondk/Downloads/archive (2)/split_dataset/test"
    output_folder_test = "C:/Users/kondk/Downloads/archive (2)/split_dataset_processed/test"
    process_all_images(input_folder_test, output_folder_test)