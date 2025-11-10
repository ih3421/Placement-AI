import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
-
# (You can use your own CSV or the given Kaggle dataset)
df = pd.read_csv("college_student_placement_dataset.csv")

# To encode strings in to numbers
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])
#Change this if ur target column is different
target_col = "Placement"  

if target_col not in df.columns:
    raise ValueError(f"'{target_col}' column not found. Replace with the correct target variable name.")

X = df.drop(target_col, axis=1)
y = df[target_col]
# Divide data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Normalize input
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression (for probability prediction)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Random Forest (for feature importance)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# Evaluating models 
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return y_pred, y_prob

print(" Logistic Regression Results:")
y_pred_log, y_prob_log = evaluate_model(log_reg, X_test_scaled, y_test)

print(" Random Forest Results:")
y_pred_rf, y_prob_rf = evaluate_model(rf, X_test, y_test)

# Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance (Random Forest)
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=True).plot(kind='barh', figsize=(8, 6))
plt.title("Feature Importance for Placement Prediction")
plt.show()

#Save Model
import joblib
joblib.dump(log_reg, "/storage/emulated/0/logistic_model.pkl")
joblib.dump(rf, "/storage/emulated/0/randomForest_model.pkl")
joblib.dump(scaler, "/storage/emulated/0/scaler.pkl")
