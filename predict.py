import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from tkinter import Tk, filedialog

# Get the main project directory
base_path = os.path.abspath(os.getcwd())
model_path = os.path.join(base_path, "model")

# Load trained model
svm = joblib.load(os.path.join(model_path, "svm_lung_cancer.pkl"))

# Open file dialog to upload an image
def upload_image():
    Tk().withdraw()  # Hide the root Tkinter window
    image_path = filedialog.askopenfilename(title="Select a Lung CT Scan",
                                            filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    return image_path

# Function to preprocess and extract features
def predict_lung_cancer(image_path):
    if not image_path:
        print("❌ No image selected. Please upload an image.")
        return

    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image '{image_path}' not found! Please check the file path.")
        return

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Ensure image is read properly
    if img is None:
        print(f"❌ ERROR: Unable to read image '{image_path}'. Check if the file is corrupted.")
        return

    img = cv2.resize(img, (128, 128))
    img = cv2.equalizeHist(img)

    # Extract HOG features
    features, _ = hog(img, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)

    features = np.array(features).reshape(1, -1)  # Reshape for prediction

    # Predict
    prediction = svm.predict(features)

    # Map prediction to class
    class_map = {0: "Benign (Non-Cancerous)", 1: "Malignant (Cancerous)", 2: "Normal"}
    return class_map[prediction[0]]

# Upload image and predict
test_image_path = upload_image()
if test_image_path:
    result = predict_lung_cancer(test_image_path)
    if result:
        print(f"\n✅ Prediction for '{test_image_path}': {result}")
