id: 692602d63bc9bc6a310b833c_documentation
summary: Support vector machines Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Credit Risk Overfit Identifier: Understanding Model Complexity and Generalization

## 1. Introduction: Navigating Credit Risk with SVMs
Duration: 0:10

In the dynamic landscape of credit risk management, accurate prediction of loan defaults is paramount. Financial institutions rely heavily on robust predictive models to assess borrower creditworthiness, set appropriate interest rates, and manage their portfolios effectively. However, even sophisticated models can falter if they suffer from common pitfalls: **overfitting** or **underfitting**.

This application serves as an interactive educational tool designed specifically for Risk Managers and developers. It demystifies the concepts of overfitting and underfitting using a Support Vector Machine (SVM) classifier in the context of credit default prediction. By manipulating the SVM's key regularization parameter, $C$, you will gain a hands-on understanding of how model complexity impacts its ability to generalize from historical data to new, unseen credit applications.

Understanding the nuances of model complexity and generalization is not just an academic exercise; it's a critical skill for building reliable and stable credit risk models that perform consistently in real-world scenarios, preventing costly misjudgments in lending decisions.

### Application Overview and Learning Objectives

This lab, "Credit Risk Overfit Identifier: Understanding Model Complexity and Generalization," guides you through an interactive journey to explore the critical concepts of overfitting and underfitting. The Streamlit application allows you to:

*   Define and differentiate between overfitting and underfitting in predictive models.
*   Understand the role of the SVM $C$ parameter in controlling model complexity and its trade-off between fitting training data and generalizing to unseen data.
*   Interpret key classification metrics (Accuracy, False Positive Rate, False Negative Rate) for both training and test sets to diagnose model performance issues.
*   Visually identify how decision boundaries change with model complexity.
*   Appreciate the importance of model generalization for robust credit risk management.

### Support Vector Machines (SVM) Fundamentals

Support Vector Machines (SVMs) are powerful supervised learning models used for classification and regression tasks. In essence, an SVM constructs a hyperplane or a set of hyperplanes in a high-dimensional space, which can be used for classification, regression, or other tasks. Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training data point of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.

For classification, SVMs aim to find the hyperplane that best separates data points of different classes. For non-linearly separable data, SVMs use a "kernel trick" to transform the input data into a higher-dimensional space where a linear separation might be possible. The Radial Basis Function (RBF) kernel, which we will use, is a popular choice.

#### SVM Primal Optimization Problem
The core idea behind SVMs can be formalized as an optimization problem. For a linearly separable case, the goal is to maximize the margin between classes while minimizing classification errors. The primal form of the SVM optimization problem (for soft-margin classification, which allows for some misclassifications) is given by:
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

Here's what these variables mean:
*   $\mathbf{w}$: This is the **weight vector** that defines the orientation of the hyperplane.
*   $b$: This is the **bias term** (or intercept) that determines the position of the hyperplane relative to the origin.
*   $\boldsymbol{\xi}$: These are the **slack variables**, representing the degree of misclassification for each data point.
    If $\xi_i = 0$, the point is correctly classified and outside the margin. If $0 < \xi_i < 1$, the point is correctly
    classified but inside the margin. If $\xi_i \ge 1$, the point is misclassified.
*   $y_i$: The true class label for the $i$-th training example (either -1 or 1).
*   $\mathbf{x}_i$: The feature vector for the $i$-th training example.
*   $C$: The **regularization parameter**.

#### The Role of the Regularization Parameter $C$
The parameter $C$ is crucial in controlling the trade-off between achieving a low training error and a large margin.

*   **Small $C$ values (e.g., $C \approx 0.01$):** A small $C$ puts a higher penalty on the margin maximization term
    (i.e., less emphasis on fitting individual training points perfectly). This leads to a larger margin but potentially
    more misclassifications on the training data. The model becomes **simpler** and more prone to **underfitting**,
    as it might not capture the underlying patterns in the data well.

*   **Large $C$ values (e.g., $C \approx 100$):** A large $C$ places a higher penalty on misclassifications
    (i.e., less tolerance for errors on training points). This forces the model to classify almost all training
    data points correctly, potentially leading to a smaller margin. The model becomes **more complex** and tends to
    **overfit** the training data, meaning it learns the noise and specific patterns of the training set too well,
    and performs poorly on unseen data.

Finding the optimal $C$ value is key to building an SVM model that generalizes well.

### Setting Up the Environment
For this application, we use standard Python libraries for data manipulation (`pandas`, `numpy`), machine learning (`scikit-learn`), and visualization (`matplotlib`, `seaborn`). The necessary libraries are assumed to be pre-installed in your environment. The application is built using Streamlit, which handles the interactive user interface.

<aside class="positive">
  The application utilizes <code>st.session_state</code> to persist generated data and model components across different pages (modules) of the Streamlit application. This ensures a seamless user experience as you navigate through the lab.
</aside>

## 2. Generating and Splitting Synthetic Credit Risk Data
Duration: 0:08

To effectively demonstrate overfitting and underfitting in a controlled environment, we will use a synthetic dataset. This dataset is designed to mimic key characteristics of credit risk data, specifically featuring two primary drivers of loan default: **Debt-to-Income Ratio** and **Credit Score**. The dataset will contain two distinct classes: 'Non-Default' (label 0) and 'Default' (label 1), with an imbalanced distribution to reflect real-world credit scenarios.

The features have been scaled to be within typical ranges for these financial indicators.

The `generate_credit_risk_data` function in `application_pages/page_2_data_generation.py` uses `sklearn.datasets.make_classification` to create this dataset.

```python
# application_pages/page_2_data_generation.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Global variables to store data, accessible across pages
# Initialize in session state to persist across reruns
if "credit_data" not in st.session_state:
    st.session_state.credit_data = None
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_train" not in st.session_state:
    st.session_state.y_train = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None

@st.cache_data(ttl="2h")
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

def main():
    st.markdown("## 5. Generating the Synthetic Credit Risk Dataset")
    # ... (rest of the markdown content)
    st.session_state.credit_data = generate_credit_risk_data(n_samples=1000, random_state=42)
    st.write("First 5 rows of the synthetic credit data:")
    st.dataframe(st.session_state.credit_data.head())
    st.write("\nClass distribution:")
    st.dataframe(st.session_state.credit_data['loan_default'].value_counts())
    # ... (rest of the main function content)
```

**Output in the app:**
First 5 rows of the synthetic credit data:
| Debt_to_Income_Ratio | Credit_Score | loan_default |
|-|--|--|
| 0.35                 | 570.2        | 0            |
| 0.28                 | 610.5        | 0            |
| 0.51                 | 480.1        | 1            |
| 0.40                 | 530.8        | 0            |
| 0.22                 | 700.3        | 0            |

Class distribution:
| loan_default | count |
|--|-|
| 0            | 850   |
| 1            | 150   |

### Splitting Data into Training and Test Sets

Before training any machine learning model, it is crucial to split the dataset into distinct training and test sets. This practice is fundamental for evaluating a model's ability to generalize to unseen data and detect potential overfitting.

*   **Training Set:** Used to train the SVM model. The model learns patterns and decision boundaries from this data.
*   **Test Set:** Used to evaluate the trained model's performance on data it has never seen before. This provides an unbiased estimate of the model's generalization capability.

We perform a stratified split to ensure that the proportion of loan default cases is maintained in both the training and test sets, which is particularly important for imbalanced datasets.

```python
# application_pages/page_2_data_generation.py (continued)

def main():
    # ... (previous content)
    st.markdown("## 6. Splitting Data into Training and Test Sets")
    # ... (rest of the markdown content)

    st.session_state.X = st.session_state.credit_data[['Debt_to_Income_Ratio', 'Credit_Score']]
    st.session_state.y = st.session_state.credit_data['loan_default']

    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = train_test_split(
        st.session_state.X, st.session_state.y, test_size=0.2, random_state=42, stratify=st.session_state.y
    )
    st.write(f"Training set size: {len(st.session_state.X_train)} samples")
    st.write(f"Test set size: {len(st.session_state.X_test)} samples")
    st.write(f"Training set class distribution:\n")
    st.dataframe(st.session_state.y_train.value_counts(normalize=True))
    st.write(f"Test set class distribution:\n")
    st.dataframe(st.session_state.y_test.value_counts(normalize=True))
```

**Output in the app:**
Training set size: 800 samples
Test set size: 200 samples
Training set class distribution:

| loan_default | proportion |
|--||
| 0            | 0.85       |
| 1            | 0.15       |

Test set class distribution:

| loan_default | proportion |
|--||
| 0            | 0.85       |
| 1            | 0.15       |

## 3. Understanding Credit Risk Evaluation Metrics and Initial Data Visualization
Duration: 0:07

In credit risk modeling, it's not enough to simply know if a model is "accurate." The type of errors a model makes can have significant business implications. For instance, incorrectly classifying a defaulting customer as non-defaulting (a False Negative) can lead to financial losses, while incorrectly classifying a non-defaulting customer as defaulting (a False Positive) can result in missed business opportunities. Therefore, we use a set of specialized metrics to assess model performance more thoroughly, especially in the context of imbalanced datasets where the default class is often rare.

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

**Important Note:** In credit risk, we typically define the 'Positive' class as the **loan default** (label 1). Therefore:
*   A **False Positive** occurs when a customer who **does not default** (True Negative) is predicted to **default**. This could lead to denying a loan to a creditworthy applicant.
*   A **False Negative** occurs when a customer who **does default** (True Positive) is predicted to **not default**. This is often considered more costly as it leads to potential loan losses.

Minimizing False Negatives is often a critical objective in credit risk to prevent significant financial losses.

The `calculate_credit_risk_metrics` function, defined in `application_pages/page_3_metrics_and_initial_viz.py` (and also used in subsequent pages), computes these metrics.

```python
# application_pages/page_3_metrics_and_initial_viz.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def calculate_credit_risk_metrics(y_true, y_pred, model_name=""):
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
```

### Visualizing the Initial Dataset

Before diving into SVM training, let's visualize the synthetic credit risk dataset. This plot will show the distribution of 'Debt-to-Income Ratio' against 'Credit_Score', with points colored according to their 'loan_default' status. This initial visualization helps us understand the inherent separability of the classes and gives us a baseline before introducing the SVM decision boundaries.

The `plot_data` function generates this scatter plot.

```python
# application_pages/page_3_metrics_and_initial_viz.py (continued)

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
    # ... (previous content)
    st.markdown("## 8. Visualizing the Initial Dataset")
    # ... (rest of the markdown content)
    if st.session_state.X is not None and st.session_state.y is not None:
        fig = plot_data(st.session_state.X, st.session_state.y, title="Synthetic Credit Risk Data: Debt-to-Income vs. Credit Score")
        st.pyplot(fig)
    else:
        st.warning("Please navigate to 'Generating & Splitting Data' to create the dataset first.")
```

## 4. Implementing the Interactive SVM Classifier
Duration: 0:15

This section introduces the core of our interactive demonstration: an SVM classifier. We will train an SVM with a Radial Basis Function (RBF) kernel on our synthetic credit risk data. The RBF kernel allows the SVM to find non-linear decision boundaries, which is often necessary for real-world data.

The most critical parameter we will be exploring is the regularization parameter, $C$.

### Interactive Overfitting/Underfitting Demonstration

This is the interactive heart of the application. Use the slider below (in the actual Streamlit app) to adjust the SVM regularization parameter $C$ and observe its profound impact on the model's decision boundary and performance metrics.

As you change $C$, pay close attention to:
*   **The Decision Boundary:** How does its shape and complexity change?
*   **The Margins:** How wide or narrow do they become?
*   **Support Vectors:** Which data points are chosen as support vectors?
*   **Training vs. Test Metrics:** How do Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR) compare between the training and test sets? This is your key indicator for overfitting or underfitting.

Recall that:
*   **Small $C$ values** generally lead to **simpler models** (larger margins, more training errors tolerated), which can result in **underfitting**.
*   **Large $C$ values** generally lead to **complex models** (smaller margins, fewer training errors tolerated), which can result in **overfitting**.

The goal is to find a $C$ value that strikes a good balance, demonstrating strong performance on both training and unseen (test) data – a robust, generalized model.

The `plot_svm_decision_boundary` function handles the model training, prediction, metric calculation, and visualization.

```python
# application_pages/page_4_interactive_svm.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# calculate_credit_risk_metrics is assumed to be defined as in page_3 or a common util.

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
    # ... (rest of the markdown content)
    st.markdown("## 10. Interactive Overfitting/Underfitting Demonstration")
    # ... (rest of the markdown content)

    c_param_value = st.slider(
        'Select SVM Regularization Parameter $C$',
        min_value=0.01, max_value=100.0, value=1.0, step=0.1, format='%.2f',
        help="Controls the trade-off between classifying training points correctly and having a large margin. Small C = simpler boundary (underfitting), Large C = complex boundary (overfitting)."
    )
    plot_svm_decision_boundary(c_param_value, X_train_data=st.session_state.X_train, y_train_data=st.session_state.y_train, X_test_data=st.session_state.X_test, y_test_data=st.session_state.y_test)
```

### Interpreting the Interactive Results: Diagnosing Overfitting and Underfitting

The interactive demonstration provides immediate visual and quantitative feedback on how the $C$ parameter influences SVM behavior. Here's how to interpret the results to diagnose underfitting, optimal fit, and overfitting:

#### Underfitting (Small $C$ values, e.g., $C \approx 0.01$ to $0.1$)
*   **Visuals:** The decision boundary will appear very smooth and generalized, often failing to separate the classes well. The margin will be wide. There might be many data points (even training points) incorrectly classified or within the margin.
*   **Metrics:**
    *   **Low Training Accuracy:** The model struggles to fit the training data.
    *   **Low Test Accuracy:** Consequently, it performs poorly on unseen data.
    *   **High FPR and FNR on both training and test sets:** The model makes many errors in both directions, failing to correctly identify defaults and misclassifying non-defaults.
*   **Diagnosis:** The model is too simple; it hasn't learned enough from the training data to capture the underlying patterns. It suffers from high bias.

#### Optimal Fit (Moderate $C$ values, e.g., $C \approx 1.0$)
*   **Visuals:** The decision boundary effectively separates the classes, showing a reasonable balance between complexity and smoothness. The margin is neither too wide nor too narrow. Support vectors are strategically placed.
*   **Metrics:**
    *   **High Training Accuracy:** The model fits the training data well.
    *   **High Test Accuracy (similar to Training Accuracy):** Crucially, the model performs almost as well on unseen data.
    *   **Low FPR and FNR on both training and test sets, with values close to each other:** This indicates a good balance in correctly identifying positive and negative classes and generalizing well.
*   **Diagnosis:** The model has found a good balance between bias and variance. It has learned the underlying patterns without memorizing the noise in the training data.

#### Overfitting (Large $C$ values, e.g., $C \approx 10$ to $100$)
*   **Visuals:** The decision boundary becomes highly convoluted and "wiggly," trying to perfectly encapsulate every training point. The margin will be very narrow, or even collapse around individual points. The boundary might appear to hug outliers in the training data.
*   **Metrics:**
    *   **Very High Training Accuracy:** The model performs exceptionally well on the training data, often near 100%.
    *   **Significantly Lower Test Accuracy:** The model performs poorly on unseen data, indicating it has memorized the training set.
    *   **Low Training FPR/FNR but High Test FPR/FNR:** This is a classic sign. The model looks great on what it's seen but fails on new data. For example, it might have very low training FNR (identifies all defaults in training) but very high test FNR (misses many defaults in test).
*   **Diagnosis:** The model is too complex; it has learned the noise and specific idiosyncrasies of the training data rather than the general underlying relationships. It suffers from high variance.

By carefully observing these changes, Risk Managers can develop an intuitive understanding of how model complexity, controlled by the $C$ parameter, directly impacts a model's ability to generalize, a cornerstone of robust credit risk modeling.

## 5. Analyzing Performance Across a Range of C Values
Duration: 0:10

While the interactive demo provides a real-time feel for the impact of the $C$ parameter, it's also beneficial to analyze model performance systematically across a broad range of $C$ values. This section performs an automated sweep of the $C$ parameter, training an SVM for each value and recording the key performance metrics on both the training and test sets.

The results will highlight the trends of accuracy, False Positive Rate (FPR), and False Negative Rate (FNR) as model complexity changes, offering a more comprehensive view of the bias-variance trade-off.

The `analyze_c_range` function in `application_pages/page_5_c_range_analysis.py` is responsible for this sweep.

```python
# application_pages/page_5_c_range_analysis.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# calculate_credit_risk_metrics is assumed to be defined as in page_3 or a common util.

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

def main():
    st.markdown("## 12. Analyzing Performance Across a Range of C Values")
    # ... (rest of the markdown content)

    if st.session_state.X_train is not None and st.session_state.y_train is not None and \
       st.session_state.X_test is not None and st.session_state.y_test is not None:

        C_range = np.logspace(-2, 2, 50) # From 0.01 to 100, 50 points
        c_analysis_results = analyze_c_range(C_range, st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test)
        
        st.write("First 5 rows of C-parameter analysis results:")
        st.dataframe(c_analysis_results.head())
    else:
        st.warning("Please navigate to 'Generating & Splitting Data' and 'Evaluation Metrics & Initial Data Viz' to prepare the data first.")
```

**Output in the app (example first 5 rows):**
| C    | Train Accuracy | Test Accuracy | Train FPR | Test FPR | Train FNR | Test FNR |
||-||--|-|--|-|
| 0.01 | 0.8500         | 0.8500        | 0.0000    | 0.0000   | 1.0000    | 1.0000   |
| 0.02 | 0.8512         | 0.8500        | 0.0012    | 0.0000   | 0.9917    | 1.0000   |
| 0.03 | 0.8650         | 0.8600        | 0.0106    | 0.0118   | 0.8917    | 0.9667   |
| 0.04 | 0.8700         | 0.8650        | 0.0153    | 0.0118   | 0.8500    | 0.9333   |
| 0.05 | 0.8712         | 0.8650        | 0.0165    | 0.0118   | 0.8417    | 0.9333   |

### Visualizing Training vs. Test Accuracy for Different C Values

The plots below summarize the performance of the SVM model across the range of $C$ values. Observe the trends for both training and test sets.

The `plot_accuracy_vs_c` function generates these plots.

```python
# application_pages/page_5_c_range_analysis.py (continued)

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
    # ... (previous content)
    st.markdown("## 13. Visualizing Training vs. Test Accuracy for Different C Values")
    # ... (rest of the markdown content)
    plot_accuracy_vs_c(c_analysis_results)
```

**Interpretation of the Plots:**
*   **Accuracy Plot:**
    *   Look for the region where training accuracy is high, and test accuracy is also high and close to training accuracy. This typically indicates a good generalization.
    *   A large gap where training accuracy is significantly higher than test accuracy points to **overfitting**.
    *   When both training and test accuracy are low, it suggests **underfitting**.

*   **FPR and FNR Plot:**
    *   Analyze how False Positive Rate (FPR) and False Negative Rate (FNR) change with $C$.
    *   In credit risk, minimizing FNR (missing actual defaults) is often a priority.
    *   Notice if the FPR and FNR for the training set diverge significantly from the test set, indicating poor generalization.

These visualizations provide a powerful way to identify the optimal range for the $C$ parameter, balancing model fit and generalization capability.

## 6. Conclusion: Generalization for Robust Credit Risk Management
Duration: 0:05

This interactive lab has demonstrated the critical importance of model generalization in credit risk management. Through the lens of Support Vector Machines and the regularization parameter $C$, we've observed how a model's complexity directly influences its ability to perform accurately on new, unseen data.

*   **Underfitting** (too simple, low $C$) leads to models that fail to capture essential patterns, performing poorly on both training and test sets.
*   **Overfitting** (too complex, high $C$) results in models that memorize the training data, exhibiting excellent performance there but failing drastically on new data.
*   **Optimal Generalization** (balanced $C$) achieves a strong performance across both training and test sets, indicating a robust and reliable model.

<aside class="positive">
  For Risk Managers, the key takeaway is that a model's true value lies not in its ability to perfectly explain historical data, but in its capacity to accurately predict future events.
</aside>

Understanding and mitigating the risks of overfitting and underfitting is fundamental to building predictive models that support sound lending decisions, minimize financial exposure, and ensure the long-term stability of credit portfolios. Continuously monitoring model performance on new data and recalibrating as necessary are essential practices for effective credit risk management.
