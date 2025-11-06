import streamlit as st
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer

# === 1. Data Exploration ===

def render_data_exploration(df):
    """Renders the Data Exploration UI elements."""""
    st.subheader("Raw Data Sample")
    st.dataframe(df.head())

    # Show data summary
    st.subheader("Data Summary")
    # FIX: st.write() does not accept 'use_container_width'. 
    # Use st.dataframe() to display DataFrames.
    st.dataframe(df.describe())

    # Show churn distribution
    st.subheader("Churn Distribution")
    churn_dist = df['Churn'].value_counts()
    st.bar_chart(churn_dist)
    st.markdown(f"""
    - **Total Customers:** {df.shape[0]}
    - **Not Churn:** {churn_dist['No']} ({churn_dist['No']/df.shape[0]*100:.1f}%)
    - **Churn:** {churn_dist['Yes']} ({churn_dist['Yes']/df.shape[0]*100:.1f}%)
    
    This is an **imbalanced dataset**, which is why we use techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) \
    to improve model performance.""")

# === 2. Model Performance ===

def render_model_performance(log_reg, xgb_model, X_test_raw, y_test, preprocessor, class_names):
    """
    Renders the model performance metrics and comparison.
    Note: X_test_raw is the *unprocessed* test set. The pipelines handle preprocessing.
    """
    
    st.markdown("We compare a baseline Logistic Regression with an XGBoost Classifier.")
    
    # --- Generate Predictions ---
    # The models are full pipelines, so they take raw data
    y_pred_log_reg = log_reg.predict(X_test_raw)
    y_prob_log_reg = log_reg.predict_proba(X_test_raw)[:, 1]
    
    y_pred_xgb = xgb_model.predict(X_test_raw)
    y_prob_xgb = xgb_model.predict_proba(X_test_raw)[:, 1]

    # --- Display Metrics ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression (Baseline)")
        st.text("Classification Report:")
        report_log_reg = classification_report(y_test, y_pred_log_reg, target_names=class_names)
        st.code(report_log_reg)
        
    with col2:
        st.subheader("XGBoost (Optimized)")
        st.text("Classification Report:")
        report_xgb = classification_report(y_test, y_pred_xgb, target_names=class_names)
        st.code(report_xgb)

    # --- Display Confusion Matrix ---
    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
        fig_log_reg = px.imshow(cm_log_reg, text_auto=True, 
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=class_names, y=class_names,
                                title="Logistic Regression CM")
        st.plotly_chart(fig_log_reg, use_container_width=True)
        
    with col2:
        cm_xgb = confusion_matrix(y_test, y_pred_xgb)
        fig_xgb = px.imshow(cm_xgb, text_auto=True,
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=class_names, y=class_names,
                            title="XGBoost CM")
        st.plotly_chart(fig_xgb, use_container_width=True)

    # --- Display ROC Curve ---
    st.subheader("ROC Curve Comparison")
    
    fig, ax = plt.subplots()
    # FIX: Use RocCurveDisplay.from_predictions instead of .from_estimator
    # This avoids the ValueError as we pass the probabilities (y_prob_*) directly,
    # which we calculated earlier in this function.
    RocCurveDisplay.from_predictions(y_test, y_prob_log_reg, name="Logistic Regression", ax=ax)
    RocCurveDisplay.from_predictions(y_test, y_prob_xgb, name="XGBoost", ax=ax)
    ax.set_title("ROC Curve")
    st.pyplot(fig)


# === 3. Live Churn Predictor (NEW FUNCTION) ===

def render_prediction_ui(model, df, numeric_cols, categorical_cols):
    """
    Renders the UI for manual and CSV prediction.
    The 'model' passed here is the *full pipeline*, so it will
    handle preprocessing on its own.
    """
    
    manual_tab, csv_tab = st.tabs(["Manual Customer Entry", "Batch CSV Upload"])

    with manual_tab:
        st.subheader("Enter Customer Details Manually")
        
        # Use a form for a clean "Submit" button
        with st.form("manual_entry_form"):
            input_data = {}
            
            # Create columns for a cleaner layout
            col1, col2, col3 = st.columns(3)
            
            # Dynamically create widgets for all features
            all_cols = numeric_cols + categorical_cols
            
            # We need to sort the columns to make a sensible UI
            demographic_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
            service_cols = [c for c in all_cols if c not in numeric_cols and c not in demographic_cols]
            
            # --- Column 1: Demographics ---
            with col1:
                st.markdown("##### Demographics")
                for col in demographic_cols:
                    if col in categorical_cols:
                        # Use df.unique() to get options from the original data
                        options = df[col].unique()
                        input_data[col] = st.selectbox(col, options, key=f"manual_{col}")

            # --- Column 2: Services & Contract ---
            with col2:
                st.markdown("##### Services & Contract")
                for col in service_cols:
                     if col in categorical_cols:
                        options = df[col].unique()
                        input_data[col] = st.selectbox(col, options, key=f"manual_{col}")

            # --- Column 3: Charges & Tenure ---
            with col3:
                st.markdown("##### Charges & Tenure")
                for col in numeric_cols:
                    if col == 'tenure':
                        input_data[col] = st.number_input(col, min_value=0, max_value=120, value=1, key=f"manual_{col}")
                    elif col == 'TotalCharges':
                         # Set a default value, or it will be 0 and might be misleading
                         input_data[col] = st.number_input(col, min_value=0.0, value=0.0, format="%.2f", key=f"manual_{col}")
                    else: # MonthlyCharges
                        input_data[col] = st.number_input(col, min_value=0.0, value=0.0, format="%.2f", key=f"manual_{col}")

            submitted = st.form_submit_button("Predict Churn")

            if submitted:
                # Convert the dictionary to a DataFrame
                # The pipeline expects a DataFrame in the same format as the training data
                input_df = pd.DataFrame([input_data])
                
                # The model is a pipeline, so it will preprocess the raw data
                try:
                    prediction_proba = model.predict_proba(input_df)
                    churn_probability = prediction_proba[0][1] # Probability of 'Yes'

                    st.metric("Churn Probability", f"{churn_probability:.1%}")

                    if churn_probability > 0.5:
                        st.error("Verdict: High Churn Risk 🚨", icon="🚨")
                        st.warning("This customer is likely to churn. Consider retention strategies.")
                    else:
                        st.success("Verdict: Low Churn Risk ✅", icon="✅")
                        st.info("This customer is likely to stay.")
                
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.error("Please ensure all fields are filled correctly. 'TotalCharges' cannot be empty.")

    with csv_tab:
        st.subheader("Upload CSV for Batch Prediction")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Read the uploaded CSV
                input_df = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.dataframe(input_df.head())

                # Check if required columns are present
                required_cols = numeric_cols + categorical_cols
                missing_cols = [col for col in required_cols if col not in input_df.columns]
                
                if missing_cols:
                    st.error(f"The uploaded CSV is missing the following required columns: {', '.join(missing_cols)}")
                else:
                    st.success("All required columns are present.")

                    if st.button("Run Batch Prediction"):
                        with st.spinner("Predicting churn for all customers in file..."):
                            # Keep only the columns the model needs
                            input_df_processed = input_df[required_cols]
                            
                            # Get predictions and probabilities
                            predictions = model.predict(input_df_processed)
                            probabilities = model.predict_proba(input_df_processed)[:, 1]
                            
                            # Create the output DataFrame
                            output_df = input_df.copy()
                            output_df['Churn_Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
                            output_df['Churn_Probability'] = probabilities
                            
                            st.write("Prediction Results:")
                            st.dataframe(output_df)
                            
                            # Offer a download button for the results
                            @st.cache_data
                            def convert_df_to_csv(df_to_convert):
                                return df_to_convert.to_csv(index=False).encode('utf-8')

                            csv_output = convert_df_to_csv(output_df)
                            
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv_output,
                                file_name="churn_predictions.csv",
                                mime="text/csv",
                            )

            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")


# === 4. XAI (SHAP) ===

def render_shap_explanations(shap_values, X_test_processed, feature_names):
    """Renders the SHAP explanation plots."""
    
    st.subheader("Global Explanations (SHAP Summary)")
    st.markdown("""
    This plot shows the most important features driving the model's predictions across all customers.
    - **Feature Importance:** Features at the top are most important.
    - **Impact:** The horizontal axis shows the impact on the model's output (churn probability).
    - **Color:** Red dots mean a high feature value (e.g., high MonthlyCharges), blue dots mean a low value.
    """)
    # Create the SHAP summary plot
    # We use .values for the base SHAP values
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values.values, X_test_processed, feature_names=feature_names, show=False)
    st.pyplot(fig, width='stretch')

    st.subheader("Local Explanation (SHAP Waterfall)")
    st.markdown("""
    Select a single customer from the test set to see exactly how the model arrived
    at their specific churn prediction.
    """)
    
    # Select a customer
    customer_index = st.number_input(
        "Select a customer index from the test set:", 
        min_value=0, 
        max_value=X_test_processed.shape[0]-1, 
        value=0
    )
    
    # Generate and display the waterfall plot
    fig, ax = plt.subplots()
    # We use the shap_values object directly, indexing into it
    shap.waterfall_plot(shap_values[customer_index], show=False)
    st.pyplot(fig, width='stretch')


# === 5. XAI (LIME) ===

def render_lime_explanation(lime_explainer, model, X_test_processed, y_test, class_names, feature_names, customer_index):
    """Generates and renders a LIME explanation for a single instance."""
    
    # Get the specific customer's data
    instance = X_test_processed[customer_index]
    
    # Get the model's prediction for this customer
    # We need the *classifier* part of the model for LIME's predict_fn
    classifier = model.named_steps['classifier']
    
    # LIME's predict_fn needs a function that returns probabilities
    def predict_fn_lime(data):
        return classifier.predict_proba(data)
        
    # Generate the explanation
    with st.spinner("Generating LIME explanation..."):
        explanation = lime_explainer.explain_instance(
            instance,
            predict_fn_lime,
            num_features=10,
            top_labels=1
        )
    
    # Display the prediction
    prediction = classifier.predict(instance.reshape(1, -1))[0]
    st.write(f"**Model's Prediction:** `{class_names[prediction]}`")
    st.write(f"**Actual Class:** `{class_names[y_test.iloc[customer_index]]}`")

    # Display the LIME explanation as an HTML plot
    # Get the label for which to explain
    predicted_class_index = prediction
    st.write(f"**Showing explanation for class:** `{class_names[predicted_class_index]}`")
    
    # Show LIME explanation as a plot
    fig = explanation.as_pyplot_figure(label=predicted_class_index)
    st.pyplot(fig, width='stretch')
    
    st.write(
        "**How to Read:** The 'weights' show the feature's contribution. For example, "
        "'tenure < 10' might be a strong green feature for a 'Churn' prediction, "
        "while 'Contract = Two year' might be a strong red one (contradicting Churn)."
    )

# === 6. XAI Tab Container ===

def render_xai_tabs(model, shap_values, lime_explainer, X_test_processed, y_test, class_names, feature_names):
    """
    Creates the XAI tabs and calls the specific rendering functions.
    
    PERFORMANCE FIX: This function now accepts pre-computed `shap_values`
    instead of the `shap_explainer` and does NOT perform the slow
    calculation itself.
    """
    
    # Create tabs for SHAP and LIME
    shap_tab, lime_tab = st.tabs(["SHAP Explanations (Global & Local)", "LIME Explanations (Local)"])

    with shap_tab:
        # Pass the pre-computed values
        render_shap_explanations(
            shap_values=shap_values, 
            X_test_processed=X_test_processed, 
            feature_names=feature_names
        )

    with lime_tab:
        st.subheader("Local Explanations (LIME)")
        st.markdown("""
        LIME (Local Interpretable Model-agnostic Explanations) provides an alternative
        way to explain individual predictions. It builds a simpler, local model
        around a single prediction to understand what features drove it.
        """)
        
        # Select a customer
        customer_index_lime = st.number_input(
            "Select a customer index from the test set:", 
            min_value=0, 
            max_value=X_test_processed.shape[0]-1, 
            value=1, # Use a different default from SHAP
            key="lime_customer_index" # Add a key to make it distinct
        )
        
        render_lime_explanation(
            lime_explainer=lime_explainer,
            model=model, # Pass the full pipeline
            X_test_processed=X_test_processed,
            y_test=y_test,
            class_names=class_names,
            feature_names=feature_names,
            customer_index=customer_index_lime
        )