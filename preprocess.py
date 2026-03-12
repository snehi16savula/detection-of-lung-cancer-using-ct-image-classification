import cv2
import os

# Get the main project directory
base_path = os.path.abspath(os.getcwd())
dataset_path = os.path.join(base_path, "dataset")

# Automatically find "Lung cancer dataset" inside "dataset/"
lung_dataset_path = None
for subfolder in os.listdir(dataset_path):
    if "lung cancer dataset" in subfolder.lower():
        lung_dataset_path = os.path.join(dataset_path, subfolder)
        break

# If "Lung cancer dataset" is not found, show an error
if not lung_dataset_path:
    print("❌ ERROR: 'Lung cancer dataset' folder not found inside 'dataset/'!")
    exit(1)

output_path = os.path.join(base_path, "processed_dataset")

# Ensure output folder exists
os.makedirs(output_path, exist_ok=True)

# Process each category
for category in ["bengin", "malignant", "normal"]:
    input_folder = os.path.join(lung_dataset_path, category)
    output_folder = os.path.join(output_path, category)

    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"❌ ERROR: Folder '{input_folder}' not found! Please check dataset structure.")
        continue  # Skip if the folder is missing

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith((".jpg", ".png")):
            img_path = os.path.join(input_folder, file)
            processed_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            processed_img = cv2.resize(processed_img, (128, 128))  # Resize
            processed_img = cv2.equalizeHist(processed_img)  # Enhance contrast

            # Save the processed image
            cv2.imwrite(os.path.join(output_folder, file), processed_img)

print(f"✅ Processed images saved in '{output_path}'")
