import os
import numpy as np
import cv2
from skimage.feature import hog

# Get the main project directory
base_path = os.path.abspath(os.getcwd())
dataset_path = os.path.join(base_path, "processed_dataset")
output_path = os.path.join(base_path, "model")

# Ensure model folder exists
os.makedirs(output_path, exist_ok=True)

X = []  # Features
y = []  # Labels
label_map = {"bengin": 0, "malignant": 1, "normal": 2}

# Process dataset
for category, label in label_map.items():
    folder_path = os.path.join(dataset_path, category)

    if not os.path.exists(folder_path):
        print(f"❌ ERROR: Folder '{folder_path}' not found!")
        continue

    for file in os.listdir(folder_path):
        if file.endswith((".jpg", ".png")):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Extract features
            features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True)

            X.append(features)
            y.append(label)

# Convert to NumPy arrays and save
X = np.array(X)
y = np.array(y)
np.save(os.path.join(output_path, "X_features.npy"), X)
np.save(os.path.join(output_path, "y_labels.npy"), y)

print("✅ Features Extracted and Saved!")
