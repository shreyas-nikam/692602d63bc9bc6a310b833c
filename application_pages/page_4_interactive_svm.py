
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def calculate_credit_risk_metrics(y_true, y_pred, model_name=""): # Redefine or import from a common util if needed
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        TN, FP, FN, TP = cm.ravel()
    elif cm.size == 1 and y_true.sum() == 0:
        TN = cm[0,0]
        FP, FN, TP = 0, 0, 0
    elif cm.size == 1 and y_true.sum() == len(y_true):
        TP = cm[0,0]
        FP, FN, TN = 0, 0, 0
    else:
        TN, FP, FN, TP = 0, 0, 0, 0

    fpr = FP / (FP + TN) if (FP + TN) != 0 else (0.0 if FP == 0 else np.nan)
    fnr = FN / (FN + TP) if (FN + TP) != 0 else (0.0 if FN == 0 else np.nan)

    accuracy = float(accuracy)
    fpr = float(fpr) if fpr is not None else np.nan
    fnr = float(fnr) if fnr is not None else np.nan

    return {
        f"{model_name} Accuracy": accuracy,
        f"{model_name} FPR": fpr,
        f"{model_name} FNR": fnr
    }

def plot_svm_decision_boundary(C_param, gamma_param='scale', X_train_data=None, y_train_data=None, X_test_data=None, y_test_data=None):
    if X_train_data is None or y_train_data is None or X_test_data is None or y_test_data is None:
        st.warning("Training and test data not found. Please ensure data is generated and split.")
        return

    svm_model = SVC(kernel='rbf', C=C_param, gamma=gamma_param, random_state=42)
    svm_model.fit(X_train_data, y_train_data)

    y_train_pred = svm_model.predict(X_train_data)
    y_test_pred = svm_model.predict(X_test_data)

    train_metrics = calculate_credit_risk_metrics(y_train_data, y_train_pred, "Train")
    test_metrics = calculate_credit_risk_metrics(y_test_data, y_test_pred, "Test")

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(x=X_train_data.iloc[:, 0], y=X_train_data.iloc[:, 1], hue=y_train_data,
                    palette='coolwarm', s=80, alpha=0.7, label='Training Data', ax=ax)
    sns.scatterplot(x=X_test_data.iloc[:, 0], y=X_test_data.iloc[:, 1], hue=y_test_data,
                    palette='coolwarm', marker='X', s=100, alpha=0.7, label='Test Data', ax=ax)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
    
    # Handle cases where decision_function might fail on an empty meshgrid or due to model issues
    try:
        Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax.contour(xx, yy, Z, colors=['gray', 'black', 'gray'], levels=[-1, 0, 1], alpha=0.7,
                   linestyles=['--', '-', '--'])
    except Exception as e:
        st.warning(f"Could not plot decision boundary: {e}")

    if len(svm_model.support_vectors_) > 0:
        ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=200,
                   linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')
    else:
        st.info("No Support Vectors identified for this C parameter. Consider adjusting the C value.")

    ax.set_title(f'SVM Decision Boundary (C={C_param:.2f}, gamma={gamma_param})')
    ax.set_xlabel(X_train_data.columns[0])
    ax.set_ylabel(X_train_data.columns[1])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    metrics_data = {
        "Metric": ["Accuracy", "FPR", "FNR"],
        "Train Set": [train_metrics["Train Accuracy"], train_metrics["Train FPR"], train_metrics["Train FNR"]],
        "Test Set": [test_metrics["Test Accuracy"], test_metrics["Test FPR"], test_metrics["Test FNR"]]
    }
    metrics_df = pd.DataFrame(metrics_data).set_index("Metric")
    st.write("### Classification Metrics")
    st.dataframe(metrics_df)

def main():
    st.markdown("## 9. Implementing the SVM Classifier and Decision Boundary Plotting")
    st.markdown("""
    This section introduces the core of our interactive demonstration: an SVM classifier.
    We will train an SVM with a Radial Basis Function (RBF) kernel on our synthetic credit risk data.
    The RBF kernel allows the SVM to find non-linear decision boundaries, which is often necessary
    for real-world data.

    The most critical parameter we will be exploring is the regularization parameter, $C$.
    """)

    st.markdown("## 10. Interactive Overfitting/Underfitting Demonstration")
    st.markdown("""
    This is the interactive heart of the application. Use the slider below to adjust the SVM regularization
    parameter $C$ and observe its profound impact on the model's decision boundary and performance metrics.

    As you change $C$, pay close attention to:
    *   **The Decision Boundary:** How does its shape and complexity change?
    *   **The Margins:** How wide or narrow do they become?
    *   **Support Vectors:** Which data points are chosen as support vectors?
    *   **Training vs. Test Metrics:** How do Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR)
        compare between the training and test sets? This is your key indicator for overfitting or underfitting.

    Recall that:
    *   **Small $C$ values** generally lead to **simpler models** (larger margins, more training errors tolerated),
        which can result in **underfitting**.
    *   **Large $C$ values** generally lead to **complex models** (smaller margins, fewer training errors tolerated),
        which can result in **overfitting**.

    The goal is to find a $C$ value that strikes a good balance, demonstrating strong performance on both
    training and unseen (test) data – a robust, generalized model.
    """)

    c_param_value = st.slider(
        'Select SVM Regularization Parameter $C$',
        min_value=0.01, max_value=100.0, value=1.0, step=0.1, format='%.2f',
        help="Controls the trade-off between classifying training points correctly and having a large margin. Small C = simpler boundary (underfitting), Large C = complex boundary (overfitting)."
    )
    plot_svm_decision_boundary(c_param_value, X_train_data=st.session_state.X_train, y_train_data=st.session_state.y_train, X_test_data=st.session_state.X_test, y_test_data=st.session_state.y_test)

    st.markdown("## 11. Interpreting the Interactive Results: Diagnosing Overfitting and Underfitting")
    st.markdown("""
    The interactive demonstration provides immediate visual and quantitative feedback on how the $C$ parameter
    influences SVM behavior. Here's how to interpret the results to diagnose underfitting, optimal fit, and overfitting:

    ### Underfitting (Small $C$ values, e.g., $C \approx 0.01$ to $0.1$)
    *   **Visuals:** The decision boundary will appear very smooth and generalized, often failing to separate the classes well.
        The margin will be wide. There might be many data points (even training points) incorrectly classified or within the margin.
    *   **Metrics:**
        *   **Low Training Accuracy:** The model struggles to fit the training data.
        *   **Low Test Accuracy:** Consequently, it performs poorly on unseen data.
        *   **High FPR and FNR on both training and test sets:** The model makes many errors in both directions,
            failing to correctly identify defaults and misclassifying non-defaults.
    *   **Diagnosis:** The model is too simple; it hasn't learned enough from the training data to capture the underlying patterns. It suffers from high bias.

    ### Optimal Fit (Moderate $C$ values, e.g., $C \approx 1.0$)
    *   **Visuals:** The decision boundary effectively separates the classes, showing a reasonable balance between complexity and smoothness.
        The margin is neither too wide nor too narrow. Support vectors are strategically placed.
    *   **Metrics:**
        *   **High Training Accuracy:** The model fits the training data well.
        *   **High Test Accuracy (similar to Training Accuracy):** Crucially, the model performs almost as well on unseen data.
        *   **Low FPR and FNR on both training and test sets, with values close to each other:** This indicates a good balance
            in correctly identifying positive and negative classes and generalizing well.
    *   **Diagnosis:** The model has found a good balance between bias and variance. It has learned the underlying patterns
        without memorizing the noise in the training data.

    ### Overfitting (Large $C$ values, e.g., $C \approx 10$ to $100$)
    *   **Visuals:** The decision boundary becomes highly convoluted and "wiggly," trying to perfectly encapsulate every training point.
        The margin will be very narrow, or even collapse around individual points.
        The boundary might appear to hug outliers in the training data.
    *   **Metrics:**
        *   **Very High Training Accuracy:** The model performs exceptionally well on the training data, often near 100%.
        *   **Significantly Lower Test Accuracy:** The model performs poorly on unseen data, indicating it has memorized the training set.
        *   **Low Training FPR/FNR but High Test FPR/FNR:** This is a classic sign. The model looks great on what it's seen but fails on new data.
            For example, it might have very low training FNR (identifies all defaults in training) but very high test FNR (misses many defaults in test).
    *   **Diagnosis:** The model is too complex; it has learned the noise and specific idiosyncrasies of the training data
        rather than the general underlying relationships. It suffers from high variance.

    By carefully observing these changes, Risk Managers can develop an intuitive understanding of how model complexity,
    controlled by the $C$ parameter, directly impacts a model's ability to generalize, a cornerstone of robust credit risk modeling.
    """)

