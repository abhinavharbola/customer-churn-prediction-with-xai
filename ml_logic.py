import streamlit as st
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# --- Caching Functions for ML Artifacts ---
# Use st.cache_data for functions that return serializable data (dataframes, dicts, etc.)

@st.cache_data
def load_data(path):
    """
    Loads the Telco Churn dataset from a local path and applies initial cleaning.
    FIX: Changed from URL to local file path.
    """
    df = pd.read_csv(path)
    # Basic cleaning: Convert TotalCharges to numeric, coercing errors
    # This will create NaNs, which our preprocessor will handle.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

@st.cache_data
def get_raw_splits(df):
    """
    Helper function to get the raw X and y splits before any processing.
    Used for LIME.
    """
    # Define features (X) and target (y)
    # Drop customerID as it's an identifier, not a feature
    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0) # Convert target to binary
    
    # Standard train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

@st.cache_data
def get_processed_data(_df):
    """
    Loads data, splits it, and returns all necessary artifacts for
    training, evaluation, and explainability.
    
    This function is cached to avoid re-running on every app interaction.
    The _df parameter (with an underscore) is a Streamlit convention
    to indicate that this function's cache depends on the *data* in df.
    """
    
    # Get initial data splits
    X_train, X_test, y_train, y_test = get_raw_splits(_df)
    
    # Define feature types for the preprocessor
    # We'll use these lists in the UI as well
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod'
    ]
    
    # Create a preprocessor
    # Numerical features: Impute NaNs (e.g., in TotalCharges) with median, then scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features: Impute NaNs with the most frequent value, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep any columns not listed
    )
    
    # Apply the preprocessor (fit on train, transform on train/test)
    # This pipeline only handles preprocessing
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding
    feature_names = preprocessor.get_feature_names_out()
    
    # Apply SMOTE to the processed training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
    
    return (
        X_train, X_test, y_train, y_test, 
        X_train_smote, y_train_smote, 
        preprocessor, feature_names,
        numeric_features, categorical_features  # <-- ADDED THESE
    )

@st.cache_data
def train_models(_preprocessor, X_train_smote, y_train_smote):
    """
    Trains and returns a Logistic Regression and XGBoost model.
    We pass the _preprocessor (even though it's already used)
    to show the cache dependency.
    """
    
    # 1. Logistic Regression (Baseline)
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    
    # 2. XGBoost (Advanced)
    xgb = XGBClassifier(
        eval_metric='logloss', 
        random_state=42,
        base_score=0.5 # Explicitly set for consistency
    )
    
    # Create full pipelines including SMOTE (for training) and the model
    # We use ImbPipeline to ensure SMOTE is *only* applied during .fit()
    pipeline_log_reg = ImbPipeline(steps=[
        ('preprocessor', _preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', log_reg)
    ])
    
    pipeline_xgb = ImbPipeline(steps=[
        ('preprocessor', _preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', xgb)
    ])
    
    # The `get_processed_data` function *already* gives us the preprocessed, SMOTE'd data.
    # The models should just be the classifiers themselves.
    # The preprocessor artifact will be combined with the model artifact into a new pipeline for the final prediction UI.
    
    # 1. Train Logistic Regression on processed, SMOTE'd data
    log_reg.fit(X_train_smote, y_train_smote)
    
    # 2. Train XGBoost on processed, SMOTE'd data
    xgb.fit(X_train_smote, y_train_smote)
    
    # 3. Create the final, deployable pipelines
    # These pipelines include preprocessing AND the already-trained classifier.
    # This is what we will use for prediction.
    
    # This pipeline structure is for inference
    # For *training*, we'd use ImbPipeline as shown above, but .fit() it on the *raw* X_train, y_train.
    
    # `train_models` will return pipelines that include preprocessing.
    
    # 1. Logistic Regression Pipeline (for training & inference)
    # This pipeline will preprocess, apply SMOTE, and then classify
    pipeline_log_reg = ImbPipeline(steps=[
        ('preprocessor', _preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # 2. XGBoost Pipeline (for training & inference)
    pipeline_xgb = ImbPipeline(steps=[
        ('preprocessor', _preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42, base_score=0.5))
    ])
    
    # `get_processed_data` gives us all we need.
    # `train_models` will just train the models on the processed data
    # and return the final pipelines for inference.
    
    # 1. Logistic Regression
    log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
    log_reg_model.fit(X_train_smote, y_train_smote)
    
    # 2. XGBoost
    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42, base_score=0.5)
    xgb_model.fit(X_train_smote, y_train_smote)
    
    # 3. Create and return the *full inference pipelines*
    # These pipelines bundle the preprocessor with the *trained* model
    final_pipeline_log_reg = Pipeline(steps=[
        ('preprocessor', _preprocessor),
        ('classifier', log_reg_model)
    ])
    
    final_pipeline_xgb = Pipeline(steps=[
        ('preprocessor', _preprocessor),
        ('classifier', xgb_model)
    ])
    
    # The XAI tools will need the *trained model* (xgb_model) and the
    # *processed data* (X_train_smote, X_test_processed)
    # The prediction UI will need the *full pipeline* (final_pipeline_xgb)
        
    df = load_data("customerchurn.csv")
    X_train_raw, _, y_train_raw, _ = get_raw_splits(df)
    
    # 1. Logistic Regression Pipeline
    pipeline_log_reg = ImbPipeline(steps=[
        ('preprocessor', _preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline_log_reg.fit(X_train_raw, y_train_raw)
        
    # 2. XGBoost Pipeline
    pipeline_xgb = ImbPipeline(steps=[
        ('preprocessor', _preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', XGBClassifier(eval_metric='logloss', random_state=42, base_score=0.5))
    ])
    pipeline_xgb.fit(X_train_raw, y_train_raw)
    
    return pipeline_log_reg, pipeline_xgb

@st.cache_resource
def get_shap_explainer(_model, X_train_processed):
    """
    Creates a SHAP TreeExplainer for the given model's classifier step.
    Cached as a resource.
    """
    # We need to pass the *classifier* part of the pipeline
    classifier = _model.named_steps['classifier']
    explainer = shap.TreeExplainer(classifier, X_train_processed)
    return explainer

@st.cache_data
def get_shap_values(_explainer, X_test_processed):
    """
    PERFORMANCE FIX: This is a new function to calculate SHAP values.
    This is the slowest step, so we cache the *data* (the values).
    """
    shap_values = _explainer(X_test_processed)
    return shap_values

@st.cache_resource
def get_lime_explainer(X_train_processed_values, feature_names, class_names):
    """
    Creates a LIME Tabular Explainer.
    Cached as a resource.
    """
    # LIME needs the data as numpy array
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_processed_values,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    return explainer
