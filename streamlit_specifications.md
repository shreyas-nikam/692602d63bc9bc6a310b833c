
# Streamlit Application Specification: Credit Risk Overfit Identifier

This document outlines the design and functional requirements for a Streamlit application based on the provided Jupyter Notebook and user requirements. The application aims to educate Risk Managers on overfitting and underfitting in credit risk models using an interactive Support Vector Machine (SVM) demonstration.

## 1. Application Overview

The "Credit Risk Overfit Identifier" application will provide an interactive platform for Risk Managers to understand the impact of model complexity on generalization, specifically in the context of credit default prediction using Support Vector Machines. Users will be able to manipulate the SVM regularization parameter $C$ and observe real-time changes in decision boundaries and key credit risk metrics across training and test datasets.

### Learning Goals

By interacting with this application, Risk Managers will be able to:
*   Define and differentiate between overfitting and underfitting in predictive models.
*   Understand the role of the SVM $C$ parameter in controlling model complexity and its trade-off between fitting training data and generalizing to unseen data.
*   Interpret key classification metrics (Accuracy, False Positive Rate, False Negative Rate) for both training and test sets to diagnose model performance issues.
*   Visually identify how decision boundaries change with model complexity.
*   Appreciate the importance of model generalization for robust credit risk management.

## 2. User Interface Requirements

The application will feature a clear, intuitive layout designed for a Risk Manager persona, emphasizing clarity, interactive components, and comprehensive explanations.

### Layout and Navigation Structure

*   **Main Title:** "Credit Risk Overfit Identifier: Understanding Model Complexity and Generalization"
*   **Sections:** The content will be organized into logical, collapsible sections or using `st.container` with `st.expander` for longer blocks, mirroring the notebook's flow:
    *   Introduction
    *   Learning Objectives
    *   Support Vector Machines (SVM) Fundamentals
    *   Setting Up the Environment (Brief mention, code not displayed)
    *   Generating the Synthetic Credit Risk Dataset
    *   Splitting Data into Training and Test Sets
    *   Defining Evaluation Metrics for Credit Risk
    *   Visualizing the Initial Dataset
    *   Interactive Overfitting/Underfitting Demonstration (Main interactive section)
    *   Interpreting the Interactive Results: Diagnosing Overfitting and Underfitting
    *   Analyzing Performance Across a Range of $C$ Values
    *   Visualizing Training vs. Test Accuracy for Different $C$ Values
    *   Conclusion: Generalization in Credit Risk Management

### Input Widgets and Controls

*   **SVM Regularization Parameter $C$ Slider:**
    *   **Type:** `st.slider`
    *   **Label:** "Select SVM Regularization Parameter $C$"
    *   **Range:** `min_value=0.01`, `max_value=100.0`
    *   **Step:** `0.1`
    *   **Default Value:** `1.0`
    *   **Format:** `%.2f`
    *   **Placement:** Prominently featured in the "Interactive Overfitting/Underfitting Demonstration" section.

### Visualization Components

*   **Initial Data Distribution Plot:**
    *   **Type:** `matplotlib` scatter plot displayed via `st.pyplot`.
    *   **Content:** 'Debt-to-Income Ratio' vs. 'Credit Score', with data points colored by 'loan_default' status (0: Non-default, 1: Default).
    *   **Labels:** Clear title, 'Debt-to-Income Ratio' (x-axis), 'Credit Score' (y-axis).
    *   **Legend:** 'Loan Default' status with 'Non-Default (0)' and 'Default (1)'.

*   **Interactive SVM Decision Boundary Plot:**
    *   **Type:** `matplotlib` plot displayed via `st.pyplot`, updating dynamically with slider changes.
    *   **Content:**
        *   Scatter plot of `X_train` and `X_test` data points.
        *   Training data points distinguished from test data points (e.g., different markers, `alpha` values, or colors).
        *   SVM decision boundary hyperplane.
        *   Margin lines (corresponding to decision function levels of $-1$, $0$, and $1$).
        *   Highlighted Support Vectors (encircled).
    *   **Labels:** Dynamic title showing current $C$ value and gamma, 'Debt-to-Income Ratio' (x-axis), 'Credit Score' (y-axis).
    *   **Legend:** For Training Data, Test Data, Decision Boundary, Margins, and Support Vectors.

*   **Classification Metrics Table:**
    *   **Type:** `st.dataframe` or `st.table` for a structured display.
    *   **Content:** A table showing Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR) for both **Training** and **Test** sets.
    *   **Dynamic Update:** Metrics should update in real-time as the $C$ parameter is adjusted via the slider.

*   **Accuracy vs. $C$ Plot:**
    *   **Type:** `matplotlib` line plot displayed via `st.pyplot`.
    *   **Content:** Two lines plotting 'Training Accuracy' and 'Test Accuracy' against 'C Parameter'.
    *   **Scaling:** X-axis ('C Parameter') must use a logarithmic scale.
    *   **Labels:** Clear title, 'C Parameter (log scale)' (x-axis), 'Accuracy' (y-axis).
    *   **Legend:** 'Training Accuracy', 'Test Accuracy'.

*   **FPR and FNR vs. $C$ Plot:**
    *   **Type:** `matplotlib` line plot displayed via `st.pyplot`.
    *   **Content:** Four lines plotting 'Training FPR', 'Test FPR', 'Training FNR', and 'Test FNR' against 'C Parameter'.
    *   **Scaling:** X-axis ('C Parameter') must use a logarithmic scale.
    *   **Labels:** Clear title, 'C Parameter (log scale)' (x-axis), 'Rate' (y-axis).
    *   **Legend:** 'Training FPR', 'Test FPR', 'Training FNR', 'Test FNR'.

### Interactive Elements and Feedback Mechanisms

*   The SVM $C$ parameter slider will be the primary interactive element. Changes to the slider value will automatically trigger a re-run of the relevant model training, metric calculation, and plot generation functions, providing immediate visual and quantitative feedback.
*   Textual interpretations of underfitting, optimal fit, and overfitting will be strategically placed to explain the observed changes in plots and metrics.

## 3. Additional Requirements

### Annotation and Tooltip Specifications

*   All plots will include clear, descriptive titles, axis labels, and legends.
*   Mathematical formulas for SVM primal optimization and error rates will be displayed using Streamlit's LaTeX capabilities, `st.latex` or `st.markdown` for display equations ($$...$$) and inline equations ($...$), ensuring no asterisks are used for mathematical variables.
*   Detailed explanations of FPR and FNR, explicitly defining 'default' (label 1) as the positive class, will be provided alongside the metrics table and plots.
*   Sections explaining 'Underfitting (Small $C$ values)', 'Optimal Fit (Moderate $C$ values)', and 'Overfitting (Large $C$ values)' will directly refer to the interactive plot and metrics to guide the user's interpretation.

### Save the states of the fields properly so that changes are not lost

*   Streamlit's inherent rerun mechanism handles the slider's state. When the slider is moved, the entire script reruns with the new slider value, thus automatically preserving and utilizing the updated state.
*   Data generation and range analysis results will be cached using `@st.cache_data` to optimize performance and avoid redundant computations during interactions.

## 4. Notebook Content and Code Requirements

This section details how the Jupyter Notebook's markdown content and code will be integrated into the Streamlit application.

### Extracted Markdown Content

All markdown cells from the notebook will be incorporated into the Streamlit application using `st.markdown` or `st.write`. Special attention will be paid to LaTeX formatting.

*   **Main Title:** `# Credit Risk Overfit Identifier: Understanding Model Complexity and Generalization`
*   **Introduction:** `## 1. Introduction to Overfitting and Underfitting in Credit Risk` followed by the full explanatory text.
*   **Learning Objectives:** `## 2. Learning Objectives` followed by the bulleted list.
*   **SVM Fundamentals:** `## 3. Support Vector Machines (SVM) Fundamentals` followed by the explanatory text including the primal optimization problem and $C$ parameter explanation.
    *   **Mathematical Content (from User Requirements & OCR Page 4):**
        *   SVM Primal Optimization Problem:
            $$
            \min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2} ||\mathbf{w}||^2 + C \sum_{i=1}^{I} \xi_i
            $$
            subject to:
            $$
            y_i(\mathbf{w} \cdot \mathbf{x}_i - b) \ge 1 - \xi_i, \quad i = 1, \dots, I
            $$
            $$
            \xi_i \ge 0, \quad i = 1, \dots, I
            $$
        *   Explanation of variables: $\mathbf{w}$ (weight vector), $b$ (bias term), $\boldsymbol{\xi}$ (slack variables), $y_i$ (true class label), $\mathbf{x}_i$ (feature vector), $C$ (regularization parameter).
        *   Interpretation of small and large $C$ values.
*   **Setting Up the Environment:** `## 4. Setting Up the Environment` followed by the text. (No code display for imports).
*   **Generating Synthetic Data:** `## 5. Generating the Synthetic Credit Risk Dataset` followed by the text, and `st.write` for `credit_data.head()` and `value_counts()`.
*   **Splitting Data:** `## 6. Splitting Data into Training and Test Sets` followed by the text, and `st.write` for dataset sizes and distributions.
*   **Evaluation Metrics:** `## 7. Defining Evaluation Metrics for Credit Risk` followed by the text.
    *   **Mathematical Content (from User Requirements):**
        *   False Positive Rate (FPR):
            $$
            \text{False Positive Rate (FPR)} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}
            $$
        *   False Negative Rate (FNR):
            $$
            \text{False Negative Rate (FNR)} = \frac{\text{False Negatives}}{\text{False Negatives} + \text{True Positives}}
            $$
        *   Explanation: 'Positive' refers to the 'default' class (label 1).
*   **Visualizing Initial Data:** `## 8. Visualizing the Initial Dataset` followed by the text.
*   **Implementing SVM:** `## 9. Implementing the SVM Classifier and Decision Boundary Plotting` followed by the text.
*   **Interactive Demo:** `## 10. Interactive Overfitting/Underfitting Demonstration` followed by the text explaining $C$ values and bias-variance trade-off.
*   **Interpreting Results:** `## 11. Interpreting the Interactive Results: Diagnosing Overfitting and Underfitting` followed by the full explanatory text for underfitting, optimal fit, and overfitting.
*   **Analyzing C Range:** `## 12. Analyzing Performance Across a Range of C Values` followed by the text, and `st.dataframe` for `c_analysis_results.head()`.
*   **Visualizing Acc/FPR/FNR vs. C:** `## 13. Visualizing Training vs. Test Accuracy for Different C Values` followed by the text.
*   **Conclusion:** `## 14. Conclusion: Generalization in Credit Risk Management` followed by the full explanatory text.

### Extracted Code Stubs and Streamlit Integration

The core Python functions from the notebook will be adapted for Streamlit. `st.pyplot` will be used for displaying `matplotlib` figures, `st.dataframe` for dataframes, and `st.slider` for the interactive control. Functions that generate data or perform heavy computations that don't change based on user interaction will be decorated with `@st.cache_data`.

1.  **Imports:**
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix
    import streamlit as st
    ```

2.  **`generate_credit_risk_data` Function:**
    *   **Description:** Generates a synthetic dataset.
    *   **Streamlit Use:**
        ```python
        @st.cache_data
        def generate_credit_risk_data(n_samples=1000, random_state=42):
            X, y = make_classification(
                n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0,
                n_repeated=0, n_classes=2, n_clusters_per_class=2,
                weights=[0.85, 0.15], flip_y=0.05, random_state=random_state
            )
            df = pd.DataFrame(X, columns=['Debt_to_Income_Ratio', 'Credit_Score'])
            df['loan_default'] = y
            df['Debt_to_Income_Ratio'] = np.interp(df['Debt_to_Income_Ratio'], (df['Debt_to_Income_Ratio'].min(), df['Debt_to_Income_Ratio'].max()), (0.1, 0.6))
            df['Credit_Score'] = np.interp(df['Credit_Score'], (df['Credit_Score'].min(), df['Credit_Score'].max()), (300, 850))
            return df

        credit_data = generate_credit_risk_data(n_samples=1000, random_state=42)
        # Display snippets:
        st.write("First 5 rows of the synthetic credit data:")
        st.dataframe(credit_data.head())
        st.write("\nClass distribution:")
        st.dataframe(credit_data['loan_default'].value_counts())
        ```

3.  **Data Splitting:**
    *   **Streamlit Use:**
        ```python
        X = credit_data[['Debt_to_Income_Ratio', 'Credit_Score']]
        y = credit_data['loan_default']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        st.write(f"Training set size: {len(X_train)} samples")
        st.write(f"Test set size: {len(X_test)} samples")
        st.write(f"Training set class distribution:\n")
        st.dataframe(y_train.value_counts(normalize=True))
        st.write(f"Test set class distribution:\n")
        st.dataframe(y_test.value_counts(normalize=True))
        ```

4.  **`calculate_credit_risk_metrics` Function:**
    *   **Description:** Calculates accuracy, FPR, and FNR.
    *   **Streamlit Use:**
        ```python
        def calculate_credit_risk_metrics(y_true, y_pred, model_name=""):
            accuracy = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0)

            fpr = FP / (FP + TN) if (FP + TN) != 0 else np.nan
            fnr = FN / (FN + TP) if (FN + TP) != 0 else np.nan

            return {
                f"{model_name} Accuracy": accuracy,
                f"{model_name} FPR": fpr,
                f"{model_name} FNR": fnr
            }
        ```

5.  **`plot_data` Function (Initial Data Viz):**
    *   **Description:** Plots the synthetic dataset.
    *   **Streamlit Use:**
        ```python
        def plot_data(X, y, title="Credit Risk Data Distribution"):
            fig, ax = plt.subplots(figsize=(8, 6)) # Create figure and axes
            sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, palette='coolwarm', s=80, alpha=0.7, ax=ax)
            ax.set_title(title)
            ax.set_xlabel(X.columns[0])
            ax.set_ylabel(X.columns[1])
            ax.legend(title='Loan Default', labels=['Non-Default (0)', 'Default (1)'])
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig) # Display the figure

        plot_data(X, y, title="Synthetic Credit Risk Data: Debt-to-Income vs. Credit Score")
        ```

6.  **`plot_svm_decision_boundary` Function (Interactive Demo Core):**
    *   **Description:** Trains SVM, plots decision boundary, displays metrics.
    *   **Streamlit Use:** This function will be called directly when the slider value changes.
        ```python
        def plot_svm_decision_boundary(C_param, gamma_param='scale', X_train_data=X_train, y_train_data=y_train, X_test_data=X_test, y_test_data=y_test):
            svm_model = SVC(kernel='rbf', C=C_param, gamma=gamma_param, random_state=42)
            svm_model.fit(X_train_data, y_train_data)

            y_train_pred = svm_model.predict(X_train_data)
            y_test_pred = svm_model.predict(X_test_data)

            train_metrics = calculate_credit_risk_metrics(y_train_data, y_train_pred, "Train")
            test_metrics = calculate_credit_risk_metrics(y_test_data, y_test_pred, "Test")

            fig, ax = plt.subplots(figsize=(10, 8)) # Create figure and axes

            sns.scatterplot(x=X_train_data.iloc[:, 0], y=X_train_data.iloc[:, 1], hue=y_train_data,
                            palette='coolwarm', s=80, alpha=0.7, label='Training Data', ax=ax)
            sns.scatterplot(x=X_test_data.iloc[:, 0], y=X_test_data.iloc[:, 1], hue=y_test_data,
                            palette='coolwarm', marker='X', s=100, alpha=0.7, label='Test Data', ax=ax)

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
            Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            ax.contour(xx, yy, Z, colors=['gray', 'black', 'gray'], levels=[-1, 0, 1], alpha=0.7,
                       linestyles=['--', '-', '--'])
            ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=200,
                       linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

            ax.set_title(f'SVM Decision Boundary (C={C_param:.2f}, gamma={gamma_param})')
            ax.set_xlabel(X_train_data.columns[0])
            ax.set_ylabel(X_train_data.columns[1])
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig) # Display the figure

            metrics_data = {
                "Metric": ["Accuracy", "FPR", "FNR"],
                "Train Set": [train_metrics["Train Accuracy"], train_metrics["Train FPR"], train_metrics["Train FNR"]],
                "Test Set": [test_metrics["Test Accuracy"], test_metrics["Test FPR"], test_metrics["Test FNR"]]
            }
            metrics_df = pd.DataFrame(metrics_data).set_index("Metric")
            st.write("### Classification Metrics")
            st.dataframe(metrics_df)

        # Interactive Slider integration in Streamlit:
        st.markdown("### Interactive Overfitting/Underfitting Demonstration")
        c_param_value = st.slider(
            'Select SVM Regularization Parameter $C$',
            min_value=0.01, max_value=100.0, value=1.0, step=0.1, format='%.2f',
            help="Controls the trade-off between classifying training points correctly and having a large margin. Small C = simpler boundary (underfitting), Large C = complex boundary (overfitting)."
        )
        plot_svm_decision_boundary(c_param_value, X_train_data=X_train, y_train_data=y_train, X_test_data=X_test, y_test_data=y_test)
        ```

7.  **`analyze_c_range` Function:**
    *   **Description:** Trains multiple SVMs across a range of $C$ values.
    *   **Streamlit Use:**
        ```python
        @st.cache_data
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

        C_range = np.logspace(-2, 2, 50) # From 0.01 to 100, 50 points
        c_analysis_results = analyze_c_range(C_range, X_train, y_train, X_test, y_test)
        st.write("First 5 rows of C-parameter analysis results:")
        st.dataframe(c_analysis_results.head())
        ```

8.  **`plot_accuracy_vs_c` Function (C-Range Analysis Plots):**
    *   **Description:** Plots training/test accuracy and FPR/FNR against $C$ values.
    *   **Streamlit Use:**
        ```python
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

        plot_accuracy_vs_c(c_analysis_results)
        ```
