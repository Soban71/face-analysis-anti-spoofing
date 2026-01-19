# 👁️ Face Analysis System (Age • Gender • AI Anti-Spoofing)
Real-time face analysis from webcam using a multi-task deep learning model:
- **Age Estimation** (regression)
- **Gender Classification** (binary)
- **Anti-Spoofing / Liveness** (Real vs AI-generated Fake faces)

This project focuses on a practical, lightweight pipeline that can run on **CPU**.

---

## ✨ Demo Features
✅ Real-time webcam inference  
✅ Face detection + overlay UI  
✅ **Stable age output** (smoothing + optional locking)  
✅ **Spoof stability** (probability smoothing)  
✅ Auto reset when face changes  
✅ Works on Windows (OpenCV CAP_DSHOW)

---

## 🧠 Model Overview
A **multi-task CNN** with a shared backbone and three heads:

- **Backbone:** ResNet-18 (transfer learning)
- **Heads:**
  - `Age Head` → regression
  - `Gender Head` → 2-class classification
  - `Spoof Head` → 2-class classification (0=Fake, 1=Real)

> Why multi-task?  
> A shared backbone learns strong facial features once, then specialized heads learn age / gender / spoof efficiently.

---

## 📌 Datasets (Download Links)
⚠️ **Datasets are NOT included in this repo** (large + licensing).  
You must download them yourself.

### 1) Age & Gender Dataset (REAL faces)
**UTKFace** is recommended for training age + gender.
- Kaggle (common mirror): https://www.kaggle.com/datasets/jangedoo/utkface-new  
- Another Kaggle mirror: https://www.kaggle.com/datasets/abhikjha/utk-face-cropped

UTKFace filename format:
