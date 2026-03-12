import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Get the main project directory
base_path = os.path.abspath(os.getcwd())
model_path = os.path.join(base_path, "model")

# Load extracted features
X = np.load(os.path.join(model_path, "X_features.npy"))
y = np.load(os.path.join(model_path, "y_labels.npy"))

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)

# Save the trained model
joblib.dump(svm, os.path.join(model_path, "svm_lung_cancer.pkl"))

# Evaluate model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Trained! Accuracy: {accuracy * 100:.2f}%")
