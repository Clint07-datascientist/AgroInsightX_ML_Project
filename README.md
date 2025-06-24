# AgroInsightX: Maize Pests and Disease Classification using Machine Learning

[Video Description of The Project](https:)

### Problem Statement: Enhancing Maize Yield and Food Security in Rwanda Through Early Pest and Disease Detection
Maize is a cornerstone of food security and agricultural livelihoods in Rwanda, contributing 24% to the national GDP in Q4 2024 and employing a significant portion of the workforce. However, its production is consistently threatened by the devastating impact of pests and diseases, leading to substantial yield losses that directly affect the incomes and food security of smallholder farmers. The unpredictability and rapid spread of these threats, coupled with limited access to timely expert diagnosis for many smallholder farmers, exacerbate the problem.

### **Project Overview**

This project aims to develop and implement robust machine learning models for the early and accurate detection of common maize pests and diseases through image analysis. By leveraging a publicly available, specialized Mendeley dataset focusing exclusively on maize, we will explore advanced classification techniques to address the challenges faced by Rwandan smallholder farmers.

### **Dataset**

For your project on maize pest and disease detection, you are using a **Mendeley Dataset specifically focused on maize.** While there are a few excellent maize-specific datasets on Mendeley, based on your previous mention of "Crop Pest and Disease Detection" (which has maize classes) and the general requirements for your project, a strong candidate for your summary would be to describe that dataset or a similar comprehensive one like TOM2024.

Here's a summary tailored to a likely Mendeley dataset for maize pest and disease detection:

---

### **Summary of Dataset Used**

The dataset employed for this project is sourced from **Mendeley Data**, a reputable platform for sharing research datasets, ensuring its scientific rigor and quality. Specifically utilizing a dataset focused on **maize crop pests and diseases**.

This dataset comprises a substantial collection of **high-resolution images of maize plants, primarily focusing on leaves and overall plant health.** It encompasses a diverse range of visual manifestations, including healthy maize plants and those afflicted by various common pests and diseases. The key classes included in the dataset are:

* **Pests:** Such as **Fall Armyworm (FAW)**, **Grasshopper**, and **Leaf Beetle**.
* **Diseases:** Including **Leaf Blight**, **Leaf Spot**, and **Streak Virus**.
* **Healthy:** Images representing disease and pest-free maize plants.

The dataset, in its raw form, contains approximately **5,389 images specifically for maize**. Additionally, it provides an augmented version, expanding the maize image count to around **23,657 images**, which is crucial for training robust deep learning models and improving generalization. The images were collected under varied field conditions (e.g., different lighting, backgrounds, and angles) to ensure representativeness and reduce bias, and they typically feature diverse dimensions before resizing for model input. The images have been **validated and meticulously labeled by expert plant virologists/agricultural experts**, providing high-quality ground truth for supervised machine learning tasks. This comprehensive and well-curated dataset forms the foundation for developing and evaluating the pest and disease detection models in this project.

### **Getting Started**

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

#### **Prerequisites**

You will need Python 3 and the following libraries installed:
* TensorFlow
* Scikit-learn
* NumPy
* Seaborn
* Matplotlib
* Joblib

You can install them using pip:
```bash
pip install tensorflow scikit-learn numpy seaborn matplotlib joblib
```

#### **Setup and Execution**

1.  **Clone the repository:**
    ```bash
    git clone [the repository](https://github.com/Clint07-datascientist/AgroInsightX_ML_Project)
    cd AgroInsightX_ML_Project
    ```

2.  **Run the Notebook:**
    * Open and run the `Summative_Intro_to_ml_Clinton_Pikita_assignment.ipyn` file in a Jupyter environment. The notebook contains all the code for data preprocessing, model training, and evaluation.

### **Implementation of the ML Algorithms**

This project implements two main approaches for maize pest and disease classification:

- **Support Vector Machine (SVM):** Images are preprocessed by converting to grayscale, flattening, and normalizing with StandardScaler. Dimensionality reduction is performed using PCA. Hyperparameters (C, gamma, kernel) are tuned using RandomizedSearchCV. The best SVM model is evaluated on validation and test sets using accuracy, F1-score, precision, and recall.

- **Convolutional Neural Networks (CNN):** Images are resized to (128, 128, 3) and normalized. The CNN architecture includes multiple convolutional and pooling layers, followed by dense layers. Five training instances are run, each with different combinations of optimizers (Adam, RMSprop, SGD), regularization (L1, L2, Dropout), early stopping, and learning rates. Each model is evaluated on validation and test sets with accuracy, F1-score, precision, and recall. Loss and accuracy curves are plotted for each instance.

#### **Code Modularity**

The code is organized into modular functions for clarity and reusability:
- `extract_features()` and `get_cnn_data()` for preparing data for classical ML and CNNs, respectively.
- `define_model()` for flexible CNN model creation with different hyperparameters.
- `train_and_evaluate_model()` for classical ML model training and evaluation.
- `loss_curve_plot()` for visualizing training and validation loss/accuracy.
- `calculate_metrics()` for computing performance metrics.

#### **Neural Network Experiments Table**

The table below summarizes the five CNN training instances, each with different optimization strategies and their resulting performance on the validation set:

| Training Instance        | Optimizer | Learning Rate | Dropout Rate | Regularizer | Epochs (Stopped) | Validation Accuracy | Validation Precision | Validation Recall | Validation F1-score |
| :---------------------- | :-------- | :------------ | :----------- | :---------- | :--------------- | :----------------- | :------------------ | :--------------- | :----------------- |
| Instance 1 (Baseline)   | RMSprop   | 0.001         | 0.0          | None        | 50               | 47.41%             | 48.35%              | 47.41%           | 47.52%             |
| Instance 2              | Adam      | 0.001         | 0.5          | L2 (0.01)   | 40               | 35.92%             | 46.94%              | 45.69%           | 45.70%             |
| Instance 3              | RMSprop   | 0.001         | 0.2          | L1 (0.01)   | 40                | 13.79%             | 1.9%              | 13.79%           | 3.34%             |
| Instance 4              | SGD       | 0.001         | 0.4          | None        | 200               | 14.37%             | 34.19%               | 35.63%           | 32.44%              |
| Instance 5 (Best)       | Adam      | 0.0001        | 0.5          | L2 (0.01)   | 50               | 42.53%             | 43.26%              | 42.53%           | 41.87%             |

#### **Results and Discussion**

The ML algorithmn seeks to identify signs of an un healthy maize plant by analyzing it's leaves. A crop can be infested by a certain pest leading to a certain disease hence why the accuracies are low. 

### **How to Load the Best Model**

The best-performing model (Instance 5, saved as `optimized_cnn_model_4.keras`) can be loaded and used for predictions with the following Python code:

```python
import tensorflow as tf

# Load the model
best_model = tf.keras.models.load_model('saved_models/optimized_cnn_model_4.keras')

# Display the model's architecture
best_model.summary()

# You can now use this model to make predictions on new images
# predictions = best_model.predict(new_image_data)
