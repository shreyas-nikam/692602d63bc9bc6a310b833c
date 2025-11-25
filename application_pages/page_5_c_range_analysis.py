
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Assuming calculate_credit_risk_metrics is defined or imported in app.py or a common utils file
# For simplicity, redefine it here or ensure it's globally available in the Streamlit app context
def calculate_credit_risk_metrics(y_true, y_pred, model_name=""):
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


@st.cache_data(ttl="2h")
def analyze_c_range(C_values, X_train_data, y_train_data, X_test_data, y_test_data, gamma_param='scale'):
    train_accuracies, test_accuracies = [], []
    train_fprs, test_fprs = [], []
    train_fnrs, test_fnrs = [], []

    for C_val in C_values:
        svm_model = SVC(kernel='rbf', C=C_val, gamma=gamma_param, random_state=42)
        svm_model.fit(X_train_data, y_train_data)
        y_train_pred = svm_model.predict(X_train_data)
        y_test_pred = svm_model.predict(X_test_data)

        train_metrics = calculate_credit_risk_metrics(y_train_data, y_train_pred, "")
        test_metrics = calculate_credit_risk_metrics(y_test_data, y_test_pred, "")

        train_accuracies.append(train_metrics['Accuracy'])
        test_accuracies.append(test_metrics['Accuracy'])
        train_fprs.append(train_metrics['FPR'])
        test_fprs.append(test_metrics['FPR'])
        train_fnrs.append(train_metrics['FNR'])
        test_fnrs.append(test_metrics['FNR'])

    results_df = pd.DataFrame({
        'C': C_values,
        'Train Accuracy': train_accuracies,
        'Test Accuracy': test_accuracies,
        'Train FPR': train_fprs,
        'Test FPR': test_fprs,
        'Train FNR': train_fnrs,
        'Test FNR': test_fnrs,
    })
    return results_df

def plot_accuracy_vs_c(results_df):
    # Plot 1: Accuracy
    fig_acc, ax_acc = plt.subplots(figsize=(12, 7))
    ax_acc.plot(results_df['C'], results_df['Train Accuracy'], label='Training Accuracy', marker='o', linestyle='--', alpha=0.7)
    ax_acc.plot(results_df['C'], results_df['Test Accuracy'], label='Test Accuracy', marker='x', linestyle='-', alpha=0.9)
    ax_acc.set_xscale('log')
    ax_acc.set_xlabel('C Parameter (log scale)')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('SVM Training vs. Test Accuracy across different C values')
    ax_acc.grid(True, linestyle='--', alpha=0.6)
    ax_acc.legend()
    st.pyplot(fig_acc)

    # Plot 2: FPR/FNR
    fig_rates, ax_rates = plt.subplots(figsize=(12, 7))
    ax_rates.plot(results_df['C'], results_df['Train FPR'], label='Training FPR', marker='o', linestyle='--', alpha=0.7, color='red')
    ax_rates.plot(results_df['C'], results_df['Test FPR'], label='Test FPR', marker='x', linestyle='-', alpha=0.9, color='darkred')
    ax_rates.plot(results_df['C'], results_df['Train FNR'], label='Training FNR', marker='o', linestyle='--', alpha=0.7, color='blue')
    ax_rates.plot(results_df['C'], results_df['Test FNR'], label='Test FNR', marker='x', linestyle='-', alpha=0.9, color='darkblue')
    ax_rates.set_xscale('log')
    ax_rates.set_xlabel('C Parameter (log scale)')
    ax_rates.set_ylabel('Rate')
    ax_rates.set_title('SVM Training vs. Test FPR and FNR across different C values')
    ax_rates.grid(True, linestyle='--', alpha=0.6)
    ax_rates.legend()
    st.pyplot(fig_rates)

def main():
    st.markdown("## 12. Analyzing Performance Across a Range of C Values")
    st.markdown("""
    While the interactive demo provides a real-time feel for the impact of the $C$ parameter,
    it's also beneficial to analyze model performance systematically across a broad range of $C$ values.
    This section performs an automated sweep of the $C$ parameter, training an SVM for each value
    and recording the key performance metrics on both the training and test sets.

    The results will highlight the trends of accuracy, False Positive Rate (FPR), and False Negative Rate (FNR)
    as model complexity changes, offering a more comprehensive view of the bias-variance trade-off.
    """)

    if st.session_state.X_train is not None and st.session_state.y_train is not None and \
       st.session_state.X_test is not None and st.session_state.y_test is not None:

        C_range = np.logspace(-2, 2, 50) # From 0.01 to 100, 50 points
        c_analysis_results = analyze_c_range(C_range, st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test)
        
        st.write("First 5 rows of C-parameter analysis results:")
        st.dataframe(c_analysis_results.head())

        st.markdown("## 13. Visualizing Training vs. Test Accuracy for Different C Values")
        st.markdown("""
        The plots below summarize the performance of the SVM model across the range of $C$ values.
        Observe the trends for both training and test sets.

        **Accuracy Plot:**
        *   Look for the region where training accuracy is high, and test accuracy is also high and close to training accuracy.
            This typically indicates a good generalization.
        *   A large gap where training accuracy is significantly higher than test accuracy points to **overfitting**.
        *   When both training and test accuracy are low, it suggests **underfitting**.

        **FPR and FNR Plot:**
        *   Analyze how False Positive Rate (FPR) and False Negative Rate (FNR) change with $C$.
        *   In credit risk, minimizing FNR (missing actual defaults) is often a priority.
        *   Notice if the FPR and FNR for the training set diverge significantly from the test set, indicating poor generalization.

        These visualizations provide a powerful way to identify the optimal range for the $C$ parameter,
        balancing model fit and generalization capability.
        """)

        plot_accuracy_vs_c(c_analysis_results)
    else:
        st.warning("Please navigate to 'Generating Data' and 'Initial Data Visualization' to prepare the data first.")



