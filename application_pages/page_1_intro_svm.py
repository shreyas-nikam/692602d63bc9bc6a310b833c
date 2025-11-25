
import streamlit as st

def main():
    st.markdown("# Credit Risk Overfit Identifier: Understanding Model Complexity and Generalization")

    st.markdown("## 1. Introduction to Overfitting and Underfitting in Credit Risk")
    st.markdown("""
    In the dynamic landscape of credit risk management, accurate prediction of loan defaults is paramount.
    Financial institutions rely heavily on robust predictive models to assess borrower creditworthiness,
    set appropriate interest rates, and manage their portfolios effectively. However, even sophisticated models
    can falter if they suffer from common pitfalls: **overfitting** or **underfitting**.

    This application serves as an interactive educational tool designed specifically for Risk Managers.
    It demystifies the concepts of overfitting and underfitting using a Support Vector Machine (SVM)
    classifier in the context of credit default prediction. By manipulating the SVM's key regularization
    parameter, $C$, you will gain a hands-on understanding of how model complexity impacts its ability
    to generalize from historical data to new, unseen credit applications.

    Understanding the nuances of model complexity and generalization is not just an academic exercise;
    it's a critical skill for building reliable and stable credit risk models that perform consistently
    in real-world scenarios, preventing costly misjudgments in lending decisions.
    """)

    st.markdown("## 2. Learning Objectives")
    st.markdown("""
    By interacting with this application, Risk Managers will be able to:
    *   Define and differentiate between overfitting and underfitting in predictive models.
    *   Understand the role of the SVM $C$ parameter in controlling model complexity and its trade-off
        between fitting training data and generalizing to unseen data.
    *   Interpret key classification metrics (Accuracy, False Positive Rate, False Negative Rate)
        for both training and test sets to diagnose model performance issues.
    *   Visually identify how decision boundaries change with model complexity.
    *   Appreciate the importance of model generalization for robust credit risk management.
    """)

    st.markdown("## 3. Support Vector Machines (SVM) Fundamentals")
    st.markdown("""
    Support Vector Machines (SVMs) are powerful supervised learning models used for classification and regression tasks.
    In essence, an SVM constructs a hyperplane or a set of hyperplanes in a high-dimensional space,
    which can be used for classification, regression, or other tasks. Intuitively, a good separation is achieved
    by the hyperplane that has the largest distance to the nearest training data point of any class
    (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.

    For classification, SVMs aim to find the hyperplane that best separates data points of different classes.
    For non-linearly separable data, SVMs use a "kernel trick" to transform the input data into a higher-dimensional space
    where a linear separation might be possible. The Radial Basis Function (RBF) kernel, which we will use, is a popular choice.

    ### SVM Primal Optimization Problem
    The core idea behind SVMs can be formalized as an optimization problem. For a linearly separable case,
    the goal is to maximize the margin between classes while minimizing classification errors. The primal form of
    the SVM optimization problem (for soft-margin classification, which allows for some misclassifications)
    is given by:
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

    ### The Role of the Regularization Parameter $C$
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
    """)

    st.markdown("## 4. Setting Up the Environment")
    st.markdown("""
    For this application, we will be using standard Python libraries for data manipulation, machine learning,
    and visualization. The necessary libraries are pre-installed in the environment. We will not display
    the import statements within the main content of the application, but they are included in the underlying code.
    """)

