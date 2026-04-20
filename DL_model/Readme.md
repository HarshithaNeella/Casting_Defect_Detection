# Deep Learning Model (CNN)

## 📌 Overview
This module implements casting defect detection using a Convolutional Neural Network (CNN).

---

## ⚙️ Model Architecture
- Conv2D → ReLU
- MaxPooling
- Conv2D → ReLU
- MaxPooling
- Flatten
- Dense layers
- Output: Sigmoid (Binary Classification)

---

## 🧠 Approach
- End-to-end learning from raw images
- Automatic feature extraction
- No manual feature engineering required

---

## 📥 Input
- Grayscale images resized to 200x200
- Normalized pixel values

---

## 📤 Output
- Binary classification:
  - 0 → Defective
  - 1 → OK

---

## ▶️ Run

python cnn_app.py

---

## 📦 Model File
The trained model (`CNN_Casting_model.pkl`) is stored using Git LFS due to large size.

---

## 🚀 Advantages over ML
- Better feature representation
- Higher scalability
- Improved performance on complex patterns
