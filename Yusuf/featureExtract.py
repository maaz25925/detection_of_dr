# ===============================================================
# ADVANCED FEATURE EXTRACTION FOR FUNDUS IMAGES (CSV OUTPUT)
# ===============================================================

import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ---------------------------------------------------------------
# Feature extraction function
# ---------------------------------------------------------------

def extract_features(image_path, label):
    """Extract texture, color and edge features from a preprocessed fundus image."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1️⃣ Basic Intensity Stats
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    # 2️⃣ Color Stats (in LAB space)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    mean_l, mean_a, mean_b = np.mean(l), np.mean(a), np.mean(b)
    std_l, std_a, std_b = np.std(l), np.std(a), np.std(b)

    # 3️⃣ Texture Features using Gray Level Co-occurrence Matrix (GLCM)
    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # 4️⃣ Local Binary Pattern (LBP)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
    lbp_hist = lbp_hist / np.sum(lbp_hist)

    # 5️⃣ Edge Density (Canny)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Combine all features
    feature_dict = {
        'image': os.path.basename(image_path),
        'label': label,
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'mean_l': mean_l, 'mean_a': mean_a, 'mean_b': mean_b,
        'std_l': std_l, 'std_a': std_a, 'std_b': std_b,
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation,
        'edge_density': edge_density
    }

    # Add LBP features (texture pattern)
    for i, val in enumerate(lbp_hist):
        feature_dict[f'lbp_{i}'] = val

    return feature_dict


# ---------------------------------------------------------------
# Run over your dataset
# ---------------------------------------------------------------

def extract_features_from_folder(base_folder, output_csv):
    rows = []
    for stage in range(5):  # stages 0-4 in your dataset
        stage_folder = os.path.join(base_folder, str(stage))
        if not os.path.exists(stage_folder):
            continue

        for file in tqdm(os.listdir(stage_folder), desc=f"Stage {stage}"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(stage_folder, file)
                feat = extract_features(img_path, stage)
                if feat is not None:
                    rows.append(feat)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Features saved to {output_csv} with shape {df.shape}")


# ===============================================================
# Example run
# ===============================================================
if __name__ == "__main__":
    base_folder = "C:/Users/kondk/Downloads/archive (2)/split_dataset_processed/train"
    output_csv = "C:/Users/kondk/Downloads/archive (2)/fundus_features_train.csv"
    extract_features_from_folder(base_folder, output_csv)