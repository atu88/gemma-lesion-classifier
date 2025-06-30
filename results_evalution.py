import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)

# Load the ground truth and prediction results
truth_df = pd.read_csv("ground_truth.csv")     
pred_df = pd.read_csv("prediction_results.csv") 

# Merge by filename
merged = pd.merge(truth_df, pred_df, on="filename")

# Rename columns for clarity
y_true = merged["true_label"]  # label name on the ground truth file
y_pred = merged["prediction"]  # label name on the predictions file

# Accuracy
acc = accuracy_score(y_true, y_pred)
print("Accuracy:", round(acc, 2))

# Classification report
print("\nClassification Report for Balanced Dataset:\n")
print(classification_report(y_true, y_pred, zero_division=0))

# Optional Confusion Matrix for detailed analysis
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=sorted(y_true.unique()), yticklabels=sorted(y_true.unique()), cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("\nConfusion Matrix\n")
plt.show()