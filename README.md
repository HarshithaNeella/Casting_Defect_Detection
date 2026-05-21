
# Casting Defect Detection (ML vs Deep Learning)

## 📌 Overview

This project focuses on automated visual inspection of industrial casting components using both traditional Machine Learning and Deep Learning techniques.

It demonstrates the transition from feature-based ML models (HOG + SVM) to data-driven Deep Learning models (CNN).

---

## 📂 Dataset

The dataset used for this project is publicly available.

- Dataset Link: Kaggle Casting Product Image Dataset]](https://drive.google.com/drive/folders/16oZ0GaeloYBt0YG5K7Q7tKDpY55kcBfZ?usp=sharing
- Industrial casting defect dataset
- Binary classification: Defective vs OK
- Images are preprocessed (resized, normalized) before training

---

## ⚙️ Approaches

### 🔹 Machine Learning (HOG + SVM)

- Feature extraction using Histogram of Oriented Gradients (HOG)
- Classification using Support Vector Machine (SVM)
- Requires manual feature engineering

### 🔹 Deep Learning (CNN)

- End-to-end feature learning using Convolutional Neural Networks
- Learns spatial features directly from images
- Eliminates manual feature extraction

---

## 📊 Key Insights

- CNN model improves performance by learning complex image patterns
- Deep Learning approach is more scalable compared to traditional ML
- Demonstrates practical transition from ML → DL

---

## 📁 Project Structure

```bash
Casting_Defect_Detection/

│
├── ML_model/        # HOG + SVM implementation
├── DL_model/        # CNN implementation
├── README.md
└── requirements.txt
```

---

## ▶️ How to Run

### ML Model

```bash
python ML_model/ml_app.py
```

### DL Model

```bash
python DL_model/cnn_app.py
```

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib

---

## 🚀 Future Enhancements

- Real-time defect detection
- Streamlit/Flask deployment
- Transfer Learning implementation
- Model optimization for production use
