import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression

st.title("ROC Curve with Metrics and Confusion Matrix")

# Sidebar Controls
st.sidebar.header("Simulation Parameters")
n_samples = st.sidebar.slider("Number of Samples", 100, 5000, 1000)
noise = st.sidebar.slider("Noise Level", 0.0, 5.0, 1.0)

# Generate synthetic classification data
X, y = make_classification(n_samples=n_samples, n_features=20,
                           n_informative=2, n_redundant=10,
                           flip_y=noise / 10, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X, y)
y_scores = model.predict_proba(X)[:, 1]

# ROC curve values
fpr, tpr, thresholds = roc_curve(y, y_scores)
roc_auc = auc(fpr, tpr)

# Clean thresholds to avoid inf
finite_thresholds = thresholds[np.isfinite(thresholds)]

# Threshold slider
default_threshold = 0.5
threshold = st.slider(
    "Threshold", 
    float(np.min(finite_thresholds)), 
    float(np.max(finite_thresholds)), 
    float(default_threshold),
    step=0.01
)

# Predict based on selected threshold
y_pred = (y_scores >= threshold).astype(int)

# Confusion Matrix and Metrics
TP = np.sum((y == 1) & (y_pred == 1))
FN = np.sum((y == 1) & (y_pred == 0))
FP = np.sum((y == 0) & (y_pred == 1))
TN = np.sum((y == 0) & (y_pred == 0))

TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
FPR_val = FP / (FP + TN) if (FP + TN) != 0 else 0
youden_index = TPR - FPR_val

# Plot ROC Curve
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
ax.scatter(FPR_val, TPR, color='red', label='Current Threshold')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend(loc="lower right")
st.pyplot(fig)

# Display Metrics
st.subheader("Metrics at Selected Threshold")
col1, col2, col3, col4 = st.columns(4)
col1.metric("TPR (Sensitivity)", f"{TPR:.2f}")
col2.metric("FPR", f"{FPR_val:.2f}")
col3.metric("Youden Index", f"{youden_index:.2f}")
col4.metric("AUC", f"{roc_auc:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
conf_mat = confusion_matrix(y, y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="YlGnBu", cbar=False,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"])
ax_cm.set_ylabel("Actual")
ax_cm.set_xlabel("Predicted")
st.pyplot(fig_cm)
