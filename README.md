<div align="center">

<img src="https://img.shields.io/badge/-Python-black?style=for-the-badge&logoColor=white&logo=python&color=3776AB" alt="python" />
<img src="https://img.shields.io/badge/-Scikit_Learn-black?style=for-the-badge&logoColor=white&logo=scikitlearn&color=F7931E" alt="scikit-learn" />
<img src="https://img.shields.io/badge/-Streamlit-black?style=for-the-badge&logoColor=white&logo=streamlit&color=FF4B4B" alt="streamlit" />
<img src="https://img.shields.io/badge/-Pandas-black?style=for-the-badge&logoColor=white&logo=pandas&color=150458" alt="pandas" />
<img src="https://img.shields.io/badge/-NumPy-black?style=for-the-badge&logoColor=white&logo=numpy&color=013243" alt="numpy" />

<h1>🏋️‍♂️ NutriMindAI</h1>

### AI-Powered Health Product Analyser

*Predict whether a health product is beneficial and get personalised healthier alternatives — powered by Machine Learning.*


</div>

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [Tech Stack](#tech-stack)
3. [Features](#features)
4. [How It Works](#how-it-works)
5. [Project Structure](#project-structure)
6. [Quick Start](#quick-start)
7. [Model Details](#model-details)

---

## 🤖 Introduction

**NutriMindAI** is a machine learning web application that helps users evaluate health products before purchasing. Users enter product details such as brand, ratings, clinical approval, and effectiveness score — and the app predicts whether the product is **Beneficial** or **Not Beneficial**, then recommends the top 5 healthier alternatives from the dataset.

---

## ⚙️ Tech Stack

| Purpose | Technology |
|---------|-----------|
| **Frontend / UI** | Streamlit |
| **ML Model** | Scikit-learn (Random Forest Classifier) |
| **Data Processing** | Pandas, NumPy |
| **Model Persistence** | Joblib |
| **Preprocessing** | LabelEncoder, MinMaxScaler |

---

## ✨ Features

- **Beneficial / Not Beneficial Prediction** — classifies any health product based on 15 input features
- **Top 5 Recommendations** — suggests healthier alternatives filtered by your minimum Health Impact Score
- **Similarity-Based Ranking** — recommendations are ranked by how similar they are to your product input
- **Interactive UI** — sliders, dropdowns, and number inputs built with Streamlit
- **Instant Results** — no retraining needed; pre-trained model loaded via Joblib

---

## 🔄 How It Works

```
User enters product details
        ↓
Categorical features encoded using LabelEncoder
        ↓
Numerical features scaled using MinMaxScaler
        ↓
Random Forest Classifier predicts: Beneficial / Not Beneficial
        ↓
Similarity engine computes dot-product scores against dataset
        ↓
Top 5 products with Health Impact Score > user threshold displayed
```

---

## 🗂️ Project Structure

```
NutriMindAI/
├── app.py                    # Streamlit frontend and prediction logic
├── ML_project.ipynb          # Model training notebook (Google Colab)
├── model.pkl                 # Trained Random Forest model
├── scaler.pkl                # Fitted MinMaxScaler
├── encoders.pkl              # Fitted LabelEncoders for categorical columns
├── Ai_Health_Products.csv    # Dataset
└── README.md
```

---

## 🤸 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> **Note:** Make sure `model.pkl`, `scaler.pkl`, `encoders.pkl`, and `Ai_Health_Products.csv` are all in the same directory as `app.py`.

---

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| **Model** | Random Forest Classifier |
| **Estimators** | 250 |
| **Max Depth** | 18 |
| **Train/Test Split** | 80% / 20% |
| **Target Column** | `Beneficial_Label` (1 = Beneficial, 0 = Not Beneficial) |
| **Preprocessing** | LabelEncoding + MinMaxScaling |

### Input Features Used for Prediction

`Brand`, `Product_Type`, `Category`, `Price_INR`, `Avg_Rating`, `Num_Reviews`, `Ingredient_Quality(%)`, `Clinical_Approval(%)`, `Eco_Friendliness(%)`, `User_Trust_Score(%)`, `Effectiveness_Score(%)`, `Side_Effect_Rate(%)`, `Monthly_Sales`, `Country`, `Source_Type`

---


---

<div align="center">

**Made with 🤖 and Python**

[⬆ Back to Top](#-nutrimindai)

</div>
