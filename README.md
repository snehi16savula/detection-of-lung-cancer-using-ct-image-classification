

 Lung Cancer Detection using CT Image Classification with SVM

Overview

This project focuses on detecting lung cancer by analyzing CT scan images using a Support Vector Machine (SVM) classifier. The goal is to provide an automated method to assist radiologists in identifying malignant lung nodules, improving diagnostic accuracy, and potentially enabling early detection.

Objective

The main objective of the project is to build a machine learning model that can classify CT scan images into two categories:

* Cancerous (malignant)
* Non-cancerous (benign)

By analyzing the visual patterns in CT scans, the model helps in reducing manual effort and supports healthcare professionals in decision-making.

Dataset

The dataset used in this project contains CT scan images of lungs, labeled as either cancerous or non-cancerous. Each image is preprocessed to ensure uniformity in size and quality for effective training. Key details include:

* Image type: CT scans
* Labels: Cancerous or Non-cancerous
* Dataset size: Dependent on the source )

Tools and Technologies

* Python for data preprocessing and model development
* OpenCV and PIL for image processing
* scikit-learn for implementing SVM classifier
* NumPy and pandas for data manipulation
* Matplotlib and Seaborn for visualization

Methodology

1. Data Preprocessing

   * Images are resized and normalized for uniformity.
   * Noise removal and contrast enhancement are applied to improve feature extraction.
2. Feature Extraction

   * Features like pixel intensity, texture, and shape descriptors are extracted.
   * These features help the SVM model differentiate between cancerous and non-cancerous regions.
3. Model Training

   * SVM classifier is trained using the labeled feature dataset.
   * Hyperparameter tuning is performed to optimize accuracy.
4. Evaluation

   * Model performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.
   * Confusion matrix is used to visualize classification results.

 Results and Insights

* The SVM model demonstrates good classification performance for lung cancer detection.
* It highlights the potential of machine learning in supporting medical diagnostics.
* Misclassifications indicate areas where additional features or data augmentation could improve results.

 Learning Outcomes

Through this project, I gained experience in:

 Processing and handling medical image data
 Extracting meaningful features from CT scans for machine learning
 Implementing and tuning SVM classifiers for image classification
 Evaluating model performance using standard metrics
 Understanding the challenges of applying AI in healthcare

 Conclusion

The project successfully demonstrates that SVM can be applied for lung cancer detection using CT scan images. While it may not replace medical expertise, it provides an effective tool for assisting radiologists in early diagnosis, potentially improving patient outcomes.



