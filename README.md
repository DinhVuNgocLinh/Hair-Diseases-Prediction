# Comparative Analysis of Deep Learning Models for Hair Disease Classification

* Full Report: [Link to Report](https://drive.google.com/file/d/1BK94x7O0K20E_iojgQNX3UT7UwvX2g9w/view?usp=sharing)
* Presentation Slides: [Link to Slides](https://drive.google.com/file/d/1pU2sXmkEnnnqHlxIZO-8QfwigfW_FP2L/view?usp=sharing)


## Team Members

* Phạm Vũ Tuyết Anh – ITDSIU21073
* Đào Ngọc Lan Hồng – ITDSIU21088
* Đinh Vũ Ngọc Linh – ITDSIU21095
* Trần Triệu Như – ITDSIU21029

---

## Table of Contents

1. [Introduction](#1-introduction)  
2. [Dataset](#2-dataset)  
3. [Methodology](#3-methodology)  
   - [3.1 Libraries and Frameworks](#31-libraries-and-frameworks)  
   - [3.2 Data Preprocessing](#32-data-preprocessing)  
   - [3.3 Model Architectures](#33-model-architectures)  
   - [3.4 Training Strategy](#34-training-strategy)  
4. [Results and Discussion](#4-results-and-discussion)  
5. [Conclusion](#5-conclusion)  
6. [References](#6-references)  

---

## 1. Introduction

Hair and scalp diseases, such as alopecia areata, seborrheic dermatitis, and tinea capitis, are increasingly common and may significantly affect patients’ physical and psychological well-being. Diagnosis often relies on dermatologists’ experience, which can lead to subjectivity and variability in results.

This project investigates the application of **deep learning methods**, specifically **Convolutional Neural Networks (CNNs)**, for automated classification of hair and scalp diseases. The study compares multiple state-of-the-art CNN architectures to evaluate their performance and suitability for clinical decision support systems.

---

## 2. Dataset

* **Source:** [Kaggle Hair Diseases Dataset](https://www.kaggle.com/datasets/sundarannamalai/hair-diseases/data)
* **Data type:** Image dataset with multiple classes of scalp and hair conditions.
* **Preprocessing steps:**

  * Training images: normalized and augmented (rotation, zoom, shifting, horizontal flipping).
  * Validation/Test images: normalized only, without augmentation, to ensure objective evaluation.

---

## 3. Methodology

### 3.1 Libraries and Frameworks

The implementation was carried out using **TensorFlow** and **Keras** for deep learning, combined with the following supporting libraries:

* **NumPy** for numerical operations
* **Matplotlib** and **Seaborn** for visualization
* **scikit-learn** for evaluation metrics and confusion matrices

### 3.2 Data Preprocessing

* All images were resized to a fixed dimension compatible with pre-trained CNN architectures.
* Training data underwent augmentation to improve generalization, including random rotation, zoom, shifting, and horizontal flipping.
* Validation and test data were only normalized, ensuring unbiased evaluation of the models.
* Images were batched and loaded efficiently for training.

### 3.3 Model Architectures

The following CNN models were developed and compared:

1. **Simple CNN** – a lightweight custom architecture serving as a baseline.
2. **VGG16 & VGG19** – deep convolutional networks pre-trained on ImageNet, known for strong transfer learning.
3. **Xception** – utilizes depthwise separable convolutions for efficient feature extraction.
4. **ResNet50** – residual learning framework designed to mitigate vanishing gradient problems.
5. **MobileNetV2** – optimized for mobile and edge devices using depthwise separable convolutions.

### 3.4 Training Strategy

* **Optimizer:** Adam with learning rate of 0.001.
* **Loss Function:** categorical cross-entropy for multi-class classification.
* **Callbacks:**

  * *ReduceLROnPlateau*: reduced learning rate when validation accuracy plateaued.
  * *ModelCheckpoint*: saved the best-performing model after each epoch.
* **Epochs:** 20 for all experiments to ensure comparability.
* **Validation Monitoring:** validation accuracy and loss were tracked to detect overfitting.
---
## 4. Results and Discussion

| Model       | Test Accuracy | Parameters | Inference Time |
| ----------- | ------------- | ---------- | -------------- | 
| MobileNetV2 | 99.3 %        | \~3.5 M    | \~20 ms        | 
| Xception    | 98.8 %        | \~23.5 M   | \~35 ms        | 
| VGG16       | 99.0 %        | \~138 M    | \~46 ms        | 
| VGG19       | 99.0 %        | \~138 M    | \~46 ms        | 
| Simple CNN  | 97.0 %        | \~3 M      | \~15 ms        | 
| ResNet50    | \~57.0 %      | \~25 M     | \~40 ms        |

**Strengths and Weaknesses of Each Model**

* **MobileNetV2:** Achieved the best balance of accuracy and efficiency, making it ideal for real-time mobile or embedded applications.
* **Xception:** Delivered high accuracy with stable convergence, but required more computational resources.
* **VGG16 & VGG19:** Both showed strong transfer learning performance with near-perfect accuracy, but were computationally expensive.
* **Simple CNN:** Provided a fast, efficient baseline but underperformed on complex lesion patterns.
* **ResNet50:** Underfit the dataset, highlighting the importance of fine-tuning rather than freezing pre-trained layers.

**Key Findings**

* MobileNetV2 achieved the highest test accuracy while maintaining efficiency, making it the most practical choice for real-time or edge deployment.
* Xception and VGG models also demonstrated strong performance but with significantly higher computational requirements.
* ResNet50 underperformed due to underfitting and would require further fine-tuning.

---

## 5. Conclusion

Among the evaluated models, **MobileNetV2** emerged as the most suitable for automated hair disease classification. It balances accuracy, computational efficiency, and deployment feasibility on mobile and edge devices.

---

## 6. References

* Dataset: [Kaggle Hair Diseases Dataset](https://www.kaggle.com/datasets/sundarannamalai/hair-diseases/data)

