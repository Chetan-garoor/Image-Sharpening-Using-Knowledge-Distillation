# 🔍 Image Sharpening Using Knowledge Distillation 

## 📌 Overview

This project focuses on sharpening images using advanced enhancement techniques such as **Unsharp Masking**, **Laplacian Filters**, and **Knowledge Distillation**. It aims to improve the clarity and edges of blurred or low-detail images by leveraging both classical image processing and deep learning models.

---

## 💡 Objectives

- Enhance the sharpness of low-quality or blurred images.
- Compare traditional filters vs. deep learning-based techniques.
- Implement a **knowledge distillation approach** to compress a large sharpening model into a lightweight version.
- Build a user-friendly web application for image upload and result preview.

---

## 📊 Key Features

- 🧠 Deep learning: CNN-based sharpening model.
- 🔄 Model distillation: Lightweight student model from a pre-trained teacher.
- 🌐 Simple Web Interface: Upload → Sharpen → Download.
- 📸 Before and After comparison.

---

## 🛠️ Tech Stack

| Area            | Technologies Used                      |
|-----------------|----------------------------------------|
| Language        | Python                                 |
| Libraries       | OpenCV, NumPy, Matplotlib, PIL          |
| DL Framework    | TensorFlow / PyTorch (select one)      |
| Web Framework   | Streamlit / Flask (select one)         |
| Deployment      | GitHub / Localhost                     |

---

## 🗂️ Project Structure

```bash
image-sharpening-project/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── input/
│   └── output/
│
├── src/
│   ├── Final_code.py
│   
│
├── models/
│   ├── teacher_model.h5
│   └── student_model.h5
│   
│
│── Application/
│   ├── Template/
│   ├── Static/
│   └── app.py
│   
│
├── results/
│   ├── before_after/
│   └── metrics/
│
└── docs/
    └── Project_Report.pdf
