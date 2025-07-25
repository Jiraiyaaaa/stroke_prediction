import joblib
import pandas as pd
import numpy as np
import shap
import kagglehub
import os
from sklearn.impute import KNNImputer

# Load pickled imputer and NN model
imputer = joblib.load("imputer.pkl")
nn_model = joblib.load("nn_model.pkl")

def predict_stroke(new_patient_data, X_train_columns):
    """
    Predicts the probability of stroke for a new patient using the NN model.

    Args:
        new_patient_data (dict): A dictionary containing the patient's data.
        X_train_columns (list): List of columns from the training data after preprocessing.

    Returns:
        tuple: A tuple containing the prediction (0 or 1) and the SHAP values.
    """
    # Preprocess new patient data (impute, encode)
    df_new = pd.DataFrame(new_patient_data, index=[0])
    df_new[['bmi']] = imputer.transform(df_new[['bmi']])
    df_new = pd.get_dummies(df_new, columns=[
        'gender','ever_married','work_type','Residence_type','smoking_status'
    ], drop_first=True)

    # Align columns with the training data
    missing_cols = set(X_train_columns) - set(df_new.columns)
    for c in missing_cols:
        df_new[c] = 0
    df_new = df_new[X_train_columns]

    prediction = nn_model.predict(df_new)[0]

    # Produce SHAP explanations for clinician interpretation
    # For MLPClassifier, use KernelExplainer
    # Using a small background dataset for speed
    background_data = pd.DataFrame(np.random.rand(100, df_new.shape[1]), columns=df_new.columns)
    explainer_nn = shap.KernelExplainer(nn_model.predict_proba, background_data)
    shap_values_nn = explainer_nn.shap_values(df_new)

    # For binary classification, shap_values_nn will be a list of two arrays. Take the positive class.
    if isinstance(shap_values_nn, list) and len(shap_values_nn) == 2:
        shap_values_nn = shap_values_nn[1]

    return prediction, shap_values_nn

if __name__ == '__main__':
    # Example usage:
    new_patient = {
        'gender': 'Male',
        'age': 67.0,
        'hypertension': 0,
        'heart_disease': 1,
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 228.69,
        'bmi': 36.6,
        'smoking_status': 'formerly smoked'
    }

    # Load the original dataset to get the column names after one-hot encoding
    path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
    csv_file = os.path.join(path, "healthcare-dataset-stroke-data.csv")
    df_original = pd.read_csv(csv_file)

    # Apply the same preprocessing steps as in stroke_prediction.py to get X_train_columns
    df_original[['bmi']] = joblib.load("imputer.pkl").transform(df_original[['bmi']])
    X_train_processed = pd.get_dummies(df_original.drop(['id','stroke'], axis=1), columns=[
        'gender','ever_married','work_type','Residence_type','smoking_status'
    ], drop_first=True)
    X_train_columns = X_train_processed.columns.tolist()

    prediction, shap_values = predict_stroke(new_patient, X_train_columns)

    print(f"Prediction: {'Stroke' if prediction == 1 else 'No Stroke'}")
    print(f"SHAP Values: {shap_values}")