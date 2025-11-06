import streamlit as st
import ml_logic      # Our backend processing and ML models
import ui_components # Our frontend Streamlit components
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Customer Churn XAI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. Main Application Title ---
st.title("🤖 Explainable AI (XAI) for Customer Churn")
st.markdown("""
Welcome to this interactive dashboard for predicting customer churn. This project demonstrates a complete
ML pipeline from data loading to model explainability.
1.  Load and explore the Telco Customer Churn dataset.
2.  Train two models: a simple **Logistic Regression** and a powerful **XGBoost** classifier.
3.  Predict churn for new customers, either manually or via CSV upload.
4.  Evaluate model performance and use **SHAP** and **LIME** to understand *why* our best model makes its predictions.
""")

# --- 3. Data Loading & Preparation ---
st.header("1. Data Exploration & Preparation", divider='blue')

# FIX: Load data from the local CSV file instead of a hardcoded URL.
DATA_PATH = "customerchurn.csv"
df = ml_logic.load_data(DATA_PATH)

# Render the data exploration UI
ui_components.render_data_exploration(df)

# Get the processed data and artifacts
# This is cached, so it runs once and is stored.
with st.spinner("Processing data and building preprocessor..."):
    (
        X_train, X_test, y_train, y_test, 
        X_train_smote, y_train_smote, 
        preprocessor, feature_names,
        numeric_features, categorical_features # <-- CAPTURE NEW LISTS
    ) = ml_logic.get_processed_data(df)
    
    # Get the raw column names for the LIME explainer
    class_names = ['Not Churn', 'Churn']


# --- 4. Model Training & Performance ---
st.header("2. Model Performance Comparison", divider='blue')
st.markdown("""
Here we train two models. The models shown are **full pipelines** that include preprocessing,
SMOTE for balancing, and the classifier. We evaluate them on the unseen test set.
""")

with st.spinner("Training Logistic Regression and XGBoost models..."):
    # This function is cached and returns the full, fitted pipelines
    (log_reg, xgb_model) = ml_logic.train_models(
        preprocessor, # Pass the *unfitted* preprocessor
        X_train_smote, # Pass the data for the cache key (though it's not ideal)
        y_train_smote  # Pass the data for the cache key (though it's not ideal)
    )
    
ui_components.render_model_performance(log_reg, xgb_model, X_test, y_test, preprocessor, class_names)


# --- 5. Live Churn Predictor ---
# NEW SECTION
st.header("3. Live Churn Predictor", divider='green')
st.markdown("Predict churn for new customers, either one-by-one or via a batch CSV upload.")
ui_components.render_prediction_ui(xgb_model, df, numeric_features, categorical_features)


# --- 6. Model Explainability (XAI) ---
st.header("4. Model Explainability (XAI)", divider='blue')
st.markdown("""
Now we use SHAP and LIME to understand the *why* behind our best model's (XGBoost) predictions.
This helps build trust and uncover business insights.
""")

# We need the processed test data for XAI
# The pipeline (xgb_model) has the preprocessor
# Let's get the processed data from the pipeline
X_test_processed = xgb_model.named_steps['preprocessor'].transform(X_test)
X_train_smote_processed = xgb_model.named_steps['preprocessor'].transform(X_train) # Get processed SMOTE'd data for SHAP background

# Get the explainers from our logic file
# These are cached resources
with st.spinner("Initializing XAI Explainers..."):
    shap_explainer = ml_logic.get_shap_explainer(
        xgb_model, # Pass the full pipeline
        X_train_smote_processed # Use processed SMOTE data as background
    )
    
    lime_explainer = ml_logic.get_lime_explainer(
        X_train_smote_processed, # Use processed SMOTE data as background
        feature_names, 
        class_names
    )

# PERFORMANCE FIX: Calculate SHAP values once and cache them.
with st.spinner("Calculating SHAP values... This is a one-time operation and may take a minute."):
    start_time = time.time()
    # Use the new cached function from ml_logic
    shap_values = ml_logic.get_shap_values(shap_explainer, X_test_processed)
    end_time = time.time()
    st.toast(f"SHAP values calculated in {end_time - start_time:.2f} seconds.")

# Render the XAI tabs, passing in all the necessary artifacts
ui_components.render_xai_tabs(
    model=xgb_model,
    shap_values=shap_values,
    lime_explainer=lime_explainer,
    X_test_processed=X_test_processed,
    y_test=y_test,
    class_names=class_names,
    feature_names=feature_names
)

# --- 7. Project Conclusion ---
st.header("5. Project Conclusion", divider='blue')
st.markdown("""
This project demonstrates an end-to-end data science workflow:
1.  **Data Processing:** Loaded, cleaned, and preprocessed data using a robust `sklearn` pipeline.
2.  **Model Training:** Handled class imbalance with `SMOTE` and trained both a baseline and an advanced model.
3.  **Model Evaluation:** Compared models using comprehensive metrics and visualizations.
4.  **Live Prediction:** Deployed the model for real-time predictions on new (manual or batch) data.
5.  **Explainability (XAI):** Used SHAP and LIME to interpret model behavior, moving from a "black box" to an explainable solution.
""")