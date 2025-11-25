
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def calculate_credit_risk_metrics(y_true, y_pred, model_name=""): # Added model_name parameter
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    # Ensure cm.ravel() can handle cases where only one class is predicted
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    elif cm.size == 1 and y_true.sum() == 0: # Only negatives predicted or actual
        TN = cm[0,0] if y_true.sum() == 0 else 0
        FP, FN, TP = 0, 0, 0
    elif cm.size == 1 and y_true.sum() == len(y_true): # Only positives predicted or actual
        TP = cm[0,0] if y_true.sum() == len(y_true) else 0
        FP, FN, TN = 0, 0, 0
    else: # Fallback for unexpected cases
        TN, FP, FN, TP = 0, 0, 0, 0


    # Handle division by zero for FPR and FNR
    fpr = FP / (FP + TN) if (FP + TN) != 0 else (0.0 if FP == 0 else np.nan)
    fnr = FN / (FN + TP) if (FN + TP) != 0 else (0.0 if FN == 0 else np.nan)

    # Ensure all metrics are floats, even if nan
    accuracy = float(accuracy)
    fpr = float(fpr) if fpr is not None else np.nan
    fnr = float(fnr) if fnr is not None else np.nan

    return {
        f"{model_name} Accuracy": accuracy,
        f"{model_name} FPR": fpr,
        f"{model_name} FNR": fnr
    }

@st.cache_data(ttl="2h")
def plot_data(X, y, title="Credit Risk Data Distribution"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, palette='coolwarm', s=80, alpha=0.7, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(X.columns[0])
    ax.set_ylabel(X.columns[1])
    ax.legend(title='Loan Default', labels=['Non-Default (0)', 'Default (1)'])
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

def main():
    st.markdown("## 7. Defining Evaluation Metrics for Credit Risk")
    st.markdown("""
    In credit risk modeling, it's not enough to simply know if a model is "accurate."
    The type of errors a model makes can have significant business implications.
    For instance, incorrectly classifying a defaulting customer as non-defaulting (a False Negative)
    can lead to financial losses, while incorrectly classifying a non-defaulting customer as defaulting
    (a False Positive) can result in missed business opportunities. Therefore, we use a set of specialized
    metrics to assess model performance more thoroughly, especially in the context of imbalanced datasets where
    the default class is often rare.

    We will focus on:
    *   **Accuracy:** The proportion of correctly classified instances (both defaults and non-defaults).
    *   **False Positive Rate (FPR):** The proportion of actual non-defaults that were incorrectly predicted as defaults.
    *   **False Negative Rate (FNR):** The proportion of actual defaults that were incorrectly predicted as non-defaults.

    ### Mathematical Definitions
    **False Positive Rate (FPR)**
    $$
    \text{False Positive Rate (FPR)} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}
    $$

    **False Negative Rate (FNR)**
    $$
    \text{False Negative Rate (FNR)} = \frac{\text{False Negatives}}{\text{False Negatives} + \text{True Positives}}
    $$

    **Important Note:** In credit risk, we typically define the 'Positive' class as the **loan default** (label 1).
    Therefore:
    *   A **False Positive** occurs when a customer who **does not default** (True Negative) is predicted to **default**.
    *   A **False Negative** occurs when a customer who **does default** (True Positive) is predicted to **not default**.

    Minimizing False Negatives is often a critical objective in credit risk to prevent significant financial losses.
    """)

    st.markdown("## 8. Visualizing the Initial Dataset")
    st.markdown("""
    Before diving into SVM training, let's visualize the synthetic credit risk dataset.
    This plot will show the distribution of 'Debt-to-Income Ratio' against 'Credit Score',
    with points colored according to their 'loan_default' status.
    This initial visualization helps us understand the inherent separability of the classes
    and gives us a baseline before introducing the SVM decision boundaries.
    """)

    if st.session_state.X is not None and st.session_state.y is not None:
        fig = plot_data(st.session_state.X, st.session_state.y, title="Synthetic Credit Risk Data: Debt-to-Income vs. Credit Score")
        st.pyplot(fig)
    else:
        st.warning("Please navigate to 'Generating Data' to create the dataset first.")

