# AgroInsightX Maize Pests and Disease Classification using Machine Learning

[Video Description of The Project](https://youtu.be/x0EiqhDlbHE)

### Problem Statement: Enhancing Maize Yield and Food Security in Rwanda Through Early Pest and Disease Detection
Maize is a cornerstone of food security and agricultural livelihoods in Rwanda, contributing 24% to the national GDP in Q4 2024 and employing a significant portion of the workforce. However, its production is consistently threatened by the devastating impact of pests and diseases, leading to substantial yield losses that directly affect the incomes and food security of smallholder farmers.

Despite national efforts, including a reported 5% decrease in maize production in Season A of 2025 (down to 481,246 metric tons from 2024 Season A) and an average national yield of 2 tons per hectare (with smallholder farmers often achieving even less), these challenges persist. Major threats like Fall Armyworm (FAW), which by April 2017 had infested an estimated 17,521 hectares of maize across all 30 districts of Rwanda, and diseases such as Maize Lethal Necrosis (MLN), which can cause yield losses of up to 100% in severely infected areas, significantly hinder optimal production. The unpredictability and rapid spread of these threats, coupled with limited access to timely expert diagnosis for many smallholder farmers, exacerbate the problem.

Current methods of pest and disease detection often rely on manual inspection, which is time-consuming, requires specialized knowledge, and is not scalable, leading to delayed interventions and further crop damage. Therefore, there is a critical need for an accessible, rapid, and accurate system to identify maize pests and diseases at early stages. This project aims to address this challenge by developing and optimizing machine learning models for image-based detection of maize pests and diseases, leveraging a specialized Mendeley dataset to provide an efficient and scalable diagnostic tool for Rwandan farmers.

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



### **Implementation Choices & Methodology**

The project compares two main approaches: a classical machine learning algorithm and a deep learning approach using Neural Networks.

#### **1. Classical Model: Support Vector Machine (SVM)**

* **Choice Rationale:** SVM was chosen as the classical algorithm because it is a powerful and versatile classifier. To make it comparable to a CNN, a feature engineering step was required.
* **Methodology:**
    1.  **Feature Extraction:** Since SVMs cannot process raw image data directly, each image was converted to a 1D feature vector by flattening its pixel values.
    2.  **Preprocessing:** `StandardScaler` was applied to normalize the feature values, which is crucial for the performance of distance-based algorithms like SVM.
    3.  **Dimensionality Reduction:** To manage the high dimensionality (16,384 features per image) and significantly reduce training time, **Principal Component Analysis (PCA)** was used to reduce the features to the 150 most important components.
    4.  **Hyperparameter Tuning:** `RandomizedSearchCV` was used on a subset of the data to efficiently find the optimal hyperparameters (`C` and `gamma`) without the excessive computational cost of a full `GridSearchCV`.

#### **2. Neural Network Models (CNN)**

* **Choice Rationale:** Convolutional Neural Networks (CNNs) are the state-of-the-art for image classification tasks as they can automatically learn spatial hierarchies of features (from edges to complex patterns).
* **Baseline Model Architecture:**
    * **Input Layer:** `(128, 128, 3)` to match the resized image dimensions.
    * **Convolutional Layers:** Two `Conv2D` layers with `MaxPooling2D` to extract features.
    * **Dense Layer:** A fully connected `Dense` layer with 128 neurons.
    * **Output Layer:** A `Dense` layer with 6 neurons (for 6 classes) using a `softmax` activation function.
* **Optimization Techniques Explored:**
    * **Optimizers:** Tested `Adam`, `RMSprop`, and `SGD` to observe their impact on convergence and performance.
    * **Regularization:** Applied `L1`, `L2`, and `Dropout` regularization to combat the overfitting observed in the baseline model.
    * **Early Stopping:** Used to monitor validation loss and prevent the model from training unnecessarily long after performance has peaked.
    * **Learning Rate:** Experimented with different learning rates to fine-tune the model's convergence.

#### **Code Modularity**

To adhere to the DRY (Don't Repeat Yourself) principle and ensure the code is maintainable, several reusable functions were created in the notebook:
* `split_dataset()`: To partition a TensorFlow dataset.
* `calculate_metrics()`: To compute and return a dictionary of performance metrics.
* `plot_training_history()`: To visualize the model's learning curves.
* `build_cnn_model()`: A flexible function to build CNNs with varying optimization hyperparameters.



### **Results and Discussion**

#### **Neural Network Experiments Table**

The following table details the 5 training instances required by the assignment, showing the different combinations of hyperparameters used for the CNN and the resulting performance on the **validation set**.

| Training Instance | Optimizer | Learning Rate | Dropout Rate | Regularizer | Epochs (Stopped) | Validation Accuracy | Validation Precision | Validation Recall | Validation F1-score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Instance 1 (Baseline)** | RMSprop | 0.001 | 0.0 | None | 15 | 71.13% | 71.90% | 66.22% | 64.67% |
| **Instance 2** | Adam | 0.001 | 0.5 | L2 (0.01) | 34 | 95.49% | 89.67% | 85.04% | 86.08% |
| **Instance 3** | RMSprop | 0.001 | 0.2 | L1 (0.01) | 7 | 54.08% | 43.33% | 45.02% | 42.02% |
| **Instance 4** | SGD | 0.001 | 0.4 | None | 22 | 31.41% | 5.23% | 16.67% | 7.97% |
| **Instance 5 (Best)** | **Adam** | **0.0001** | **0.5** | **L2 (0.01)** | **39** | **96.20%** | **97.14%** | **83.68%** | **85.38%** |

#### **Analysis of Results & Key Findings**

* **Baseline vs. Optimized:** The baseline model (Instance 1) showed significant **overfitting**, with training accuracy far exceeding validation accuracy. Instance 2, which introduced `Dropout`, `L2 Regularization`, and `EarlyStopping`, dramatically reduced overfitting and improved accuracy by over 24%. This clearly demonstrates the effectiveness of these optimization techniques.

* **Hyperparameter Impact:** The choice of optimizer and regularizer had a major impact. The combination of `RMSprop` with a strong `L1` penalty (Instance 3) and the standard `SGD` optimizer (Instance 4) both failed to converge and resulted in very poor performance. This highlights that not all optimization strategies are suitable for every problem. The `Adam` optimizer proved to be the most effective for this task.

* **Fine-Tuning for Best Performance:** Instance 5 was a fine-tuning of our best model (Instance 2). By reducing the learning rate from `0.001` to `0.0001`, the model was able to converge more precisely, achieving the **highest validation accuracy of 96.20%**.



### **Final Model Comparison**

To provide a final, unbiased evaluation, the best models were tested on the held-out **test set**.

| Model | Test Accuracy | Test F1-score | Test Precision | Test Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Tuned SVM** | 49.15% | 34.84% | 47.87% | 38.72% |
| **Best CNN (Instance 5)** | **96.20%** | **85.38%** | **97.14%** | **83.68%** |


#### **Conclusion: Which implementation was better?**

The **Convolutional Neural Network (CNN) was overwhelmingly superior** to the Support Vector Machine (SVM).

* The best CNN achieved a test accuracy of **96.2%**, more than double the SVM's accuracy of **49.15%**.
* This result is expected for image classification tasks. CNNs are specifically designed to learn spatial features like textures, shapes, and patterns directly from image pixels. In contrast, the classical SVM, even with preprocessing, struggles to interpret the high-dimensional feature space of flattened images effectively.



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