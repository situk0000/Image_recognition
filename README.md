<img width="925" height="621" alt="image" src="https://github.com/user-attachments/assets/076bc236-f330-4831-825a-a0965ca1e6c8" /># Image Recognition Bootcamp Project

## Project Overview
This project is part of a 5-day bootcamp aimed at transforming beginners into skilled AI practitioners. The goal is to build an **image recognition system** using Python and deep learning, applying convolutional neural networks (CNNs) and transfer learning with MobileNetV2. 

Participants gain hands-on experience in **image preprocessing, model training, evaluation, and deployment**, culminating in a portfolio-ready project.

---

## Dataset
For this project, I used the **CIFAR-10 dataset**, which contains 60,000 32x32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).  

**Key steps performed:**
- Image normalization and resizing  
- Data augmentation (rotation, flipping)  
- Train-validation-test split  

---

## Project Structure

├── preprocess.py 
# Data loading, preprocessing, and augmentation
├── cnn_model.py 
# Custom CNN model training and evaluation
├── transfer_learning.py
# MobileNetV2 transfer learning model
├── cnn_model.h5 
# Trained CNN model
├── mobilenet_model.h5
# Trained MobileNetV2 model
├── visualizations/
# Loss/accuracy curves, confusion matrices
├── README.md
# Project summary and instructions
└── notebook.ipynb 
# Colab notebook uploaded to GitHub

## Models Implemented

### 1. Custom CNN
- Architecture: 3 convolutional layers + max pooling + dense layers  
- Activation: ReLU with softmax output  
- Optimizer: Adam  
- Metrics: Accuracy, loss, precision, recall, F1-score  
- Performance: ~XX% test accuracy  

### 2. Transfer Learning with MobileNetV2
- Pre-trained on ImageNet  
- Fine-tuned on CIFAR-10 dataset  
- Improved accuracy and faster convergence  
- Performance: ~XX% test accuracy  

---

## Training & Evaluation
- GPU-enabled training using Google Colab  
- Data augmentation applied to improve generalization  
- Plotted **loss & accuracy curves** and **confusion matrices**  
- Evaluated using metrics: Accuracy, Precision, Recall, F1-score  

---

## Sample Predictions
<img width="840" height="533" alt="image" src="https://github.com/user-attachments/assets/75ad770a-715c-4575-badd-885345b1a8ea" />
<img width="773" height="582" alt="image" src="https://github.com/user-attachments/assets/abd3c212-1cc0-494a-a2c1-5c58df6fe6fb" />



## How to Run
1. Clone the repository:  
   ```bash
   git clone https://github.com/situk0000/Image_recognition.git
Open notebook.ipynb in Google Colab

Enable GPU runtime (Runtime > Change runtime type > GPU)

Run preprocessing, CNN training, and transfer learning scripts sequentially

Trained models will be saved as .h5 files and can be used for predictions

# Key Learnings

1) Building and training a CNN from scratch

2) Applying data augmentation and preprocessing techniques

3) Transfer learning to improve model performance

4) Visualizing training metrics and evaluating models

5) Preparing portfolio-ready projects for recruiters
