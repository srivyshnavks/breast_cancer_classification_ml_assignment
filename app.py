import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide"
)

# Title and description
st.title("🎗️ Breast Cancer Classification App")
st.markdown("""
This app uses Machine Learning models to classify breast cancer tumors as **Malignant** or **Benign** 
based on the Breast Cancer Wisconsin (Diagnostic) Dataset.
""")

# Define model directory
MODEL_DIR = "model/"

# Define which models require scaling
MODELS_REQUIRING_SCALING = ['logistic_regression', 'knn']

# Model display names
MODEL_NAMES = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree Classifier',
    'knn': 'K-Nearest Neighbors',
    'naive_bayes': 'Gaussian Naive Bayes',
    'random_forest': 'Random Forest',
    'xgboost': 'XGBoost'
}

# Target names
TARGET_NAMES = ['malignant', 'benign']

# Expected feature columns (30 features from breast cancer dataset)
FEATURE_COLUMNS = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]


@st.cache_resource
def load_model(model_name):
    """Load a saved model from pickle file."""
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


@st.cache_resource
def load_scaler():
    """Load the saved scaler."""
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate all evaluation metrics."""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC Score': roc_auc_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'MCC Score': matthews_corrcoef(y_true, y_pred)
    }
    return metrics


def main():
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection dropdown
    selected_model_key = st.sidebar.selectbox(
        "Select Classification Model",
        options=list(MODEL_NAMES.keys()),
        format_func=lambda x: MODEL_NAMES[x]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Selected Model:** {MODEL_NAMES[selected_model_key]}")
    
    if selected_model_key in MODELS_REQUIRING_SCALING:
        st.sidebar.warning("⚠️ This model uses feature scaling.")
    
    # Main content
    st.header("📤 Upload Test Data")
    st.markdown("""
    Upload a CSV file containing the test data. The file should have:
    - **30 feature columns** (breast cancer features)
    - **1 target column** named `target` (0 = malignant, 1 = benign)
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV
            data = pd.read_csv(uploaded_file)
            
            # Display uploaded data info
            st.subheader("📊 Uploaded Data Preview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Samples", data.shape[0])
            col2.metric("Total Features", data.shape[1] - 1 if 'target' in data.columns else data.shape[1])
            col3.metric("Has Target Column", "Yes ✅" if 'target' in data.columns else "No ❌")
            
            with st.expander("View Data"):
                st.dataframe(data.head(20), use_container_width=True)
            
            # Check if target column exists
            if 'target' not in data.columns:
                st.error("❌ The uploaded CSV must contain a 'target' column for evaluation.")
                st.stop()
            
            # Separate features and target
            X = data.drop('target', axis=1)
            y = data['target']
            
            # Validate feature columns
            missing_cols = set(FEATURE_COLUMNS) - set(X.columns)
            if missing_cols:
                st.warning(f"⚠️ Some expected columns are missing: {missing_cols}")
            
            st.markdown("---")
            
            # Run inference button
            if st.button("🚀 Run Classification", type="primary", use_container_width=True):
                with st.spinner(f"Loading {MODEL_NAMES[selected_model_key]} model..."):
                    # Load model
                    model = load_model(selected_model_key)
                    
                    # Prepare features
                    X_inference = X.values
                    
                    # Apply scaling if required
                    if selected_model_key in MODELS_REQUIRING_SCALING:
                        scaler = load_scaler()
                        X_inference = scaler.transform(X_inference)
                
                with st.spinner("Running inference..."):
                    # Make predictions
                    y_pred = model.predict(X_inference)
                    
                    # Get probability predictions for AUC
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_inference)[:, 1]
                    else:
                        y_prob = model.decision_function(X_inference)
                
                st.success("✅ Classification completed!")
                
                # Display results
                st.markdown("---")
                st.header(f"📈 Results: {MODEL_NAMES[selected_model_key]}")
                
                # Calculate metrics
                metrics = calculate_metrics(y.values, y_pred, y_prob)
                
                # Display metrics in columns
                st.subheader("🎯 Evaluation Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                col2.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
                col3.metric("Precision", f"{metrics['Precision']:.4f}")
                
                col4, col5, col6 = st.columns(3)
                col4.metric("Recall", f"{metrics['Recall']:.4f}")
                col5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                col6.metric("MCC Score", f"{metrics['MCC Score']:.4f}")
                
                # Metrics table
                st.markdown("---")
                st.subheader("📋 Metrics Summary Table")
                metrics_df = pd.DataFrame([metrics])
                st.dataframe(metrics_df.round(4), use_container_width=True)
                
                # Confusion Matrix
                st.markdown("---")
                st.subheader("🔢 Confusion Matrix")
                cm = confusion_matrix(y.values, y_pred)
                cm_df = pd.DataFrame(
                    cm,
                    index=['Actual Malignant', 'Actual Benign'],
                    columns=['Predicted Malignant', 'Predicted Benign']
                )
                st.dataframe(cm_df, use_container_width=True)
                
                # Visual representation of confusion matrix
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**True Positives (Benign correctly classified):**")
                    st.info(f"{cm[1][1]}")
                with col2:
                    st.markdown("**True Negatives (Malignant correctly classified):**")
                    st.info(f"{cm[0][0]}")
                
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown("**False Positives (Malignant misclassified as Benign):**")
                    st.warning(f"{cm[0][1]}")
                with col4:
                    st.markdown("**False Negatives (Benign misclassified as Malignant):**")
                    st.warning(f"{cm[1][0]}")
                
                # Classification Report
                st.markdown("---")
                st.subheader("📑 Classification Report")
                report = classification_report(y.values, y_pred, target_names=TARGET_NAMES, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4), use_container_width=True)
                
                # Predictions Preview
                st.markdown("---")
                st.subheader("🔍 Predictions Preview")
                results_df = data.copy()
                results_df['Predicted'] = y_pred
                results_df['Predicted_Label'] = results_df['Predicted'].map({0: 'Malignant', 1: 'Benign'})
                results_df['Actual_Label'] = results_df['target'].map({0: 'Malignant', 1: 'Benign'})
                results_df['Correct'] = results_df['target'] == results_df['Predicted']
                
                # Show only relevant columns
                preview_cols = ['Actual_Label', 'Predicted_Label', 'Correct'] + list(X.columns[:5])
                st.dataframe(results_df[preview_cols].head(20), use_container_width=True)
                
                # Download predictions
                st.markdown("---")
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
    
    else:
        # Show sample data format when no file is uploaded
        st.info("👆 Please upload a CSV file to get started.")
        
        with st.expander("📝 Sample Data Format"):
            st.markdown("""
            Your CSV file should have the following structure:
            
            | mean radius | mean texture | ... | worst fractal dimension | target |
            |-------------|--------------|-----|-------------------------|--------|
            | 17.99       | 10.38        | ... | 0.11890                 | 0      |
            | 20.57       | 17.77        | ... | 0.08902                 | 0      |
            | ...         | ...          | ... | ...                     | ...    |
            
            - **Features**: 30 numerical features from the breast cancer dataset
            - **Target**: 0 = Malignant, 1 = Benign
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Breast Cancer Classification App | ML Assignment 2</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
