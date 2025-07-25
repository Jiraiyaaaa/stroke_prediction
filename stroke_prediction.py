
import kagglehub
import pandas as pd
import os

# Download latest version of the dataset
path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
print("Path to dataset files:", path)

# Load CSV
csv_file = os.path.join(path, "healthcare-dataset-stroke-data.csv")
df = pd.read_csv(csv_file)

# Inspect
print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Stroke distribution:\n", df['stroke'].value_counts(normalize=True))

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[['bmi']] = imputer.fit_transform(df[['bmi']])

df = pd.get_dummies(df, columns=[
'gender','ever_married','work_type','Residence_type','smoking_status'
], drop_first=True)

X = df.drop(['id','stroke'], axis=1)
y = df['stroke']

from sklearn.model_selection import StratifiedKFold
from imblearn.combine import SMOTEENN

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
smote_enn = SMOTEENN(random_state=42)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score, matthews_corrcoef
import matplotlib.pyplot as plt
import shap
import joblib

# --- Neural Network Model ---
# The assignment specifies a Feed Forward Neural Network (NN).
# We'll use MLPClassifier from scikit-learn for this.
# Hyperparameters are chosen to provide a reasonable starting point for "acceptable accuracy".
# Further tuning could be done if needed.
nn_model = MLPClassifier(
    hidden_layer_sizes=(100, 50), # Two hidden layers with 100 and 50 neurons
    max_iter=500,                 # Maximum number of iterations for the solver
    activation='relu',            # Rectified linear unit function
    solver='adam',                # Adam optimizer
    random_state=42,              # For reproducibility
    early_stopping=True,          # Stop training when validation score is not improving
    n_iter_no_change=10,          # Number of iterations with no improvement to wait before stopping
    verbose=True                  # Print progress messages to stdout
)

# --- 80/20 Train-Test Split ---
print("Performing 80/20 train-test split...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Balance training data (if needed, using SMOTEENN as it was the best performing) ---
print("Balancing training data with SMOTEENN...")
X_train_res, y_train_res = smote_enn.fit_resample(X_train, y_train)

# --- Train the Neural Network ---
print("Training the Neural Network model...")
nn_model.fit(X_train_res, y_train_res)

# --- Evaluate the Neural Network on the test set ---
print("\n--- Evaluating Neural Network Model ---")
y_pred_proba_nn = nn_model.predict_proba(X_test)[:, 1]
y_pred_nn = nn_model.predict(X_test)

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_test, y_pred_nn)
auc = roc_auc_score(y_test, y_pred_proba_nn)
f1 = f1_score(y_test, y_pred_nn)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred_nn)
mcc = matthews_corrcoef(y_test, y_pred_nn)

print(f"Accuracy: {accuracy:.3f}")
print(f"AUC: {auc:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"Balanced Accuracy: {balanced_accuracy:.3f}")
print(f"Matthews Corrcoef: {mcc:.3f}")

# --- SHAP Analysis for the Neural Network ---
print("\n--- Generating SHAP Plots for Neural Network ---")

# For MLPClassifier, we use KernelExplainer or DeepExplainer
# KernelExplainer is more general but slower. DeepExplainer is faster for Keras/TensorFlow models.
# For scikit-learn MLP, KernelExplainer is the way to go.
# Note: KernelExplainer requires a background dataset. Using a sample of X_train_res.
background = X_train_res.sample(100, random_state=42) # Use a smaller sample for speed
explainer_nn = shap.KernelExplainer(nn_model.predict_proba, background)
shap_values_nn = explainer_nn.shap_values(X_test)

# Ensure shap_values_nn is a list of arrays for binary classification
if isinstance(shap_values_nn, list) and len(shap_values_nn) == 2:
    shap_values_nn = shap_values_nn[1] # Take SHAP values for the positive class

shap.summary_plot(shap_values_nn, X_test, show=False)
plt.savefig("shap_summary_nn.png", bbox_inches='tight')
plt.close()
print("Neural Network SHAP plot saved to shap_summary_nn.png")

# --- Serialize the Neural Network Model ---
print("\n--- Serializing the Neural Network Model ---")
joblib.dump(imputer, "imputer.pkl") # Keep imputer as it's still needed for preprocessing
joblib.dump(nn_model, "nn_model.pkl")
print("Neural Network model has been serialized to nn_model.pkl.")





