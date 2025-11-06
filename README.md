# Customer Churn XAI Dashboard

This project is an **interactive web application** built with **Streamlit** to predict **customer churn** and — more importantly — to **explain why** a prediction is made using **Explainable AI (XAI)** techniques.

The dashboard provides a complete, end-to-end **data science workflow**, from **data exploration** and **model training** to **live prediction** and **model interpretation**.

---

## Features

### 1️⃣ Data Exploration
- Load and visualize the **Telco Customer Churn** dataset.  
- Explore data summaries, churn distribution, and feature relationships.

### 2️⃣ Model Performance
- Train and compare two models:
  - **Logistic Regression (baseline)**
  - **XGBoost Classifier (advanced)**
- Evaluate performance with:
  - **Classification Reports**
  - **Confusion Matrices**
  - **ROC Curves**

### 3️⃣ Live Churn Predictor
- **Manual Entry**: Fill out a form for a single customer and get an **instant churn prediction**.
- **Batch CSV Upload**: Upload a file of multiple customers to get **batch predictions**, with results downloadable as a CSV.

### 4️⃣ Explainable AI (XAI)
This dashboard integrates **SHAP** and **LIME** for interpretability.

- **SHAP (Global)**: Summary plots showing key drivers of churn across all customers.
- **SHAP (Local)**: Waterfall plots explaining the contribution of each feature for one customer.
- **LIME (Local)**: Highlights which features supported or opposed a churn prediction.

---

## Tech Stack

| Component | Tools / Libraries |
|------------|------------------|
| **Web Framework** | Streamlit |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning Pipeline** | Scikit-learn, Imbalanced-learn (SMOTE) |
| **Modeling** | XGBoost, Scikit-learn (Logistic Regression) |
| **Explainable AI (XAI)** | SHAP, LIME |
| **Visualization** | Matplotlib, Plotly |

---

## Project Structure

```bash
.
├── customerchurn.csv     # Dataset (Telco Customer Churn)
├── churn_app.py          # Main Streamlit app
├── ml_logic.py           # Data loading, preprocessing, model training, XAI logic
├── ui_components.py      # Streamlit UI components
└── requirements.txt      # Project dependencies
```

---

## How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/customer-churn-xai-dashboard.git
cd customer-churn-xai-dashboard
```

---

### 2️⃣ Set Up a Virtual Environment (Recommended)

```bash
# Create a new virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Streamlit App

```bash
streamlit run churn_app.py
```

The app will launch automatically in your default browser.

---

## Example Output

- **Model Performance Tab** → Visualizes confusion matrix, ROC curve, and key metrics.  
- **Explainability Tab** → Generates SHAP and LIME visualizations to explain predictions.  
- **Prediction Tab** → Instantly compute churn probability for individual or batch inputs.

---

## Author
Developed by **Abhinav Harbola**  
Data Science | Machine Learning | Explainable AI