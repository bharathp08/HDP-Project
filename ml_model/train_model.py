import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create directory if it doesn't exist
os.makedirs('c:\\HDP Project\\ml_model', exist_ok=True)

# Load the heart disease dataset
print("Loading heart disease dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
try:
    df = pd.read_csv(url, names=column_names, na_values='?')
except:
    print("Failed to download dataset. Using sample data instead.")
    # Create sample data if download fails
    np.random.seed(42)
    n_samples = 303
    df = pd.DataFrame({
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(94, 200, n_samples),
        'chol': np.random.randint(126, 565, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(71, 203, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6.2, n_samples).round(1),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples).astype(str),
        'thal': np.random.randint(0, 3, n_samples).astype(str),
        'target': np.random.randint(0, 2, n_samples)
    })

# Clean the data
print("Preprocessing data...")
df = df.dropna()

# Convert target to binary (0 = no disease, 1 = disease)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Handle categorical features
label_encoders = {}
for column in ['ca', 'thal']:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Feature engineering
print("Performing feature engineering...")
# Age groups
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3])
# Blood pressure categories
df['bp_category'] = pd.cut(df['trestbps'], bins=[0, 120, 140, 160, 200], labels=[0, 1, 2, 3])
# Cholesterol categories
df['chol_category'] = pd.cut(df['chol'], bins=[0, 200, 240, 300, 600], labels=[0, 1, 2, 3])
# Heart rate efficiency
df['heart_efficiency'] = df['thalach'] / df['age']

# Split the data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models and select the best one
print("Training and evaluating multiple models...")
models = {
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=200)
}

best_model = None
best_score = 0
best_model_name = ""

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Select the best model based on AUC score
    if auc > best_score:
        best_score = auc
        best_model = model
        best_model_name = name

print(f"\nSelected {best_model_name} as the best model with AUC: {best_score:.4f}")

# Fine-tune the best model
if best_model_name == "Random Forest":
    print("Fine-tuning Random Forest model...")
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [None, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'), 
                              param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
elif best_model_name == "Gradient Boosting":
    print("Fine-tuning Gradient Boosting model...")
    param_grid = {
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }
    grid_search = GridSearchCV(GradientBoostingClassifier(random_state=42, n_estimators=200), 
                              param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

# Final evaluation
y_pred = best_model.predict(X_test_scaled)
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\nFinal model performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(classification_report(y_test, y_pred))

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))

# Create a complete pipeline for prediction
pipeline = Pipeline([
    ('scaler', scaler),
    ('model', best_model)
])

# Save the model, pipeline and encoders
print("Saving model and encoders...")
joblib.dump(pipeline, 'c:\\HDP Project\\ml_model\\heart_disease_model.pkl')
joblib.dump(label_encoders, 'c:\\HDP Project\\ml_model\\label_encoders.pkl')
joblib.dump(X_train.columns.tolist(), 'c:\\HDP Project\\ml_model\\feature_names.pkl')

print("Model training complete! Files saved to c:\\HDP Project\\ml_model\\")
print("You can now use the model with your Flask application.")