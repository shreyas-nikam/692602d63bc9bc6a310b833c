
# Technical Specification for Jupyter Notebook: Credit Risk Overfit Identifier

## 1. Notebook Overview

### Learning Goals
This Jupyter Notebook aims to equip Risk Managers with a practical understanding of how model complexity impacts performance in credit risk assessment. Upon completing this notebook, users will be able to:
*   Understand the definitions and practical implications of overfitting and underfitting in predictive models for credit risk.
*   Identify how the Support Vector Machine (SVM) $C$ parameter influences the balance between model complexity and generalization.
*   Interpret training and test set performance metrics (accuracy, False Positive Rate, False Negative Rate) to diagnose overfitting or underfitting.
*   Appreciate the importance of generalization for robust credit scoring and risk management, relating it to the structural risk minimization principle.
*   Visually assess decision boundary changes in response to model parameter adjustments.

### Target Audience
This notebook is specifically targeted at **Risk Managers** and financial professionals who utilize predictive models for credit risk assessment. It assumes a basic understanding of machine learning concepts and statistical metrics.

## 2. Code Requirements

### Expected Libraries
The following Python libraries are expected to be used:
*   `numpy`: For numerical operations and array manipulation.
*   `matplotlib.pyplot`: For generating static plots and visualizations.
*   `seaborn`: For enhanced statistical data visualization.
*   `pandas`: For data manipulation and tabular data structures.
*   `sklearn.datasets`: For generating synthetic classification datasets.
*   `sklearn.model_selection`: For splitting datasets into training and testing subsets.
*   `sklearn.svm`: For implementing the Support Vector Classifier (`SVC`).
*   `sklearn.metrics`: For calculating classification performance metrics such as accuracy and confusion matrix.
*   `ipywidgets`: For creating interactive elements like sliders.
*   `IPython.display`: For displaying interactive widgets.

### Algorithms or Functions to be Implemented
The notebook will utilize and implement the following algorithms and functions:

*   **Dataset Generation:**
    *   `sklearn.datasets.make_classification`: To create a synthetic binary classification dataset simulating credit risk data. The dataset will include two informative features and a binary target variable.
*   **Data Splitting:**
    *   `sklearn.model_selection.train_test_split`: To divide the generated dataset into training and independent test sets, ensuring consistent evaluation.
*   **Model Training:**
    *   `sklearn.svm.SVC`: To train a Support Vector Machine classifier with a radial basis function (RBF) kernel. The `C` parameter will be dynamically adjusted, while `gamma` will be set to 'scale' or a fixed value.
*   **Prediction:**
    *   `SVC.predict()`: To make predictions on both the training and test datasets using the trained SVM model.
*   **Performance Metrics Calculation:**
    *   `sklearn.metrics.accuracy_score`: To calculate the overall classification accuracy.
    *   `sklearn.metrics.confusion_matrix`: To generate the confusion matrix for deriving False Positive Rate (FPR) and False Negative Rate (FNR).
    *   Custom function `calculate_credit_risk_metrics`: This function will take true labels and predicted labels as input and return accuracy, FPR, and FNR.
*   **Interactive Controls:**
    *   `ipywidgets.FloatSlider`: To provide an interactive slider for adjusting the SVM `C` parameter.
    *   `ipywidgets.interactive`: To link the slider's value to a function that re-trains the model, recalculates metrics, and updates visualizations.
*   **Decision Boundary Plotting:**
    *   Custom function `plot_decision_boundary`: This function will generate a 2D scatter plot of the data points, overlay the SVM's decision boundary, and highlight support vectors. It will create a meshgrid to plot the decision surface.

### Visualization Requirements
The following visualizations will be generated:

*   **Initial Dataset Scatter Plot:** A 2D scatter plot showing the synthetic credit risk data points, colored by their class (default/non-default), in the feature space.
*   **Dynamic Decision Boundary Plot:** A 2D scatter plot that dynamically updates. It will display the training data points, the SVM decision boundary, and the margin lines. The decision boundary and support vectors will change as the `C` parameter is adjusted via an interactive slider. This plot will be accompanied by a dashboard-like display of accuracy, FPR, and FNR for both training and test sets.
*   **Training vs. Test Accuracy Plot:** A line plot showing how training accuracy and test accuracy vary across a wide range of `C` values. This plot will clearly illustrate the regions of underfitting, optimal fit, and overfitting, providing a visual guide to the generalization sweet spot.

## 3. Notebook Sections (in detail)

### Section 1: Introduction to Overfitting and Underfitting in Credit Risk

*   **Markdown Cell:**
    This notebook introduces the critical concepts of overfitting and underfitting within the context of credit risk assessment. For Risk Managers, building models that generalize well to new, unseen loan applications is paramount. An overfitted model might perform excellently on historical data but fail dramatically on new applicants, leading to significant financial losses or missed opportunities. Conversely, an underfitted model might be too simplistic to capture the underlying patterns, resulting in consistently poor predictions.

    We will explore these concepts using a Support Vector Machine (SVM) model, focusing on the impact of its regularization parameter, $C$.

*   **Code Cell (Function Implementation/Setup - Not Applicable)**

*   **Code Cell (Execution - Not Applicable)**

*   **Markdown Cell (Explanation for Execution - Not Applicable)**

### Section 2: Learning Objectives

*   **Markdown Cell:**
    By the end of this notebook, you will be able to:
    *   Define and differentiate between overfitting and underfitting in predictive models.
    *   Understand the role of the SVM $C$ parameter in controlling model complexity and its trade-off between fitting training data and generalizing to unseen data.
    *   Interpret key classification metrics (Accuracy, False Positive Rate, False Negative Rate) for both training and test sets to diagnose model performance issues.
    *   Visually identify how decision boundaries change with model complexity.
    *   Appreciate the importance of model generalization for robust credit risk management.

### Section 3: Support Vector Machines (SVM) Fundamentals

*   **Markdown Cell:**
    Support Vector Machines (SVMs) are powerful supervised learning models used for classification and regression tasks. In classification, an SVM constructs a hyperplane or a set of hyperplanes in a high-dimensional space, which can be used for classification. The goal is to find a hyperplane that has the largest minimum distance to the training samples of any class (the "maximal margin").

    For cases where data is not perfectly linearly separable, SVMs introduce *slack variables* ($\xi_i$) to allow some misclassifications or points to lie within the margin. The regularization parameter $C$ controls the trade-off between achieving a low training error and a large margin.

    The primal optimization problem for an SVM is given by:
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
    Here:
    *   $\mathbf{w}$ is the weight vector, orthogonal to the hyperplane.
    *   $b$ is the bias term.
    *   $\boldsymbol{\xi} = (\xi_1, \dots, \xi_I)$ are the slack variables, where $\xi_i > 0$ for misclassified points or points within the margin.
    *   $y_i$ is the true class label for sample $i$ ($+1$ or $-1$).
    *   $\mathbf{x}_i$ is the feature vector for sample $i$.
    *   $C$ is the regularization parameter.
        *   A small $C$ leads to a simpler decision boundary, tolerating more misclassifications (potential underfitting).
        *   A large $C$ aims to classify all training points correctly, leading to a more complex boundary (potential overfitting).

### Section 4: Setting Up the Environment

*   **Markdown Cell:**
    Before we begin, we need to import the necessary libraries. This step ensures all required functionalities for data generation, model training, evaluation, and visualization are available.

*   **Code Cell (Function Implementation/Setup):**
    ```python
    # Import necessary libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    from ipywidgets import FloatSlider, interactive
    from IPython.display import display, HTML
    ```

*   **Code Cell (Execution):**
    ```python
    # Execute imports
    # (No explicit output for this cell, just loads libraries)
    ```

*   **Markdown Cell (Explanation for Execution):**
    The essential libraries have been loaded. `numpy` and `pandas` will handle data manipulation. `sklearn` will provide tools for dataset generation, model training, and performance evaluation. `matplotlib.pyplot` and `seaborn` are for plotting, and `ipywidgets` will enable interactive controls for the SVM parameter.

### Section 5: Generating the Synthetic Credit Risk Dataset

*   **Markdown Cell:**
    To illustrate overfitting and underfitting in a credit risk context, we will generate a synthetic dataset representing historical loan applications. This dataset will have two continuous numerical features: 'Debt-to-Income Ratio' and 'Credit Score', and a binary target variable 'loan_default' (0 for non-default, 1 for default). The data will be designed to exhibit some non-linear separability to highlight the effects of model complexity.

*   **Code Cell (Function Implementation/Setup):**
    ```python
    def generate_credit_risk_data(n_samples=1000, random_state=42):
        """
        Generates a synthetic dataset for credit risk assessment.
        Features: Debt-to-Income Ratio, Credit Score
        Target: loan_default (0: Non-default, 1: Default)
        """
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=2, # Creates non-linear separable clusters
            weights=[0.85, 0.15], # Simulates imbalanced data (e.g., more non-defaults)
            flip_y=0.05, # Adds some noise
            random_state=random_state
        )
        df = pd.DataFrame(X, columns=['Debt_to_Income_Ratio', 'Credit_Score'])
        df['loan_default'] = y
        # Scale features to more realistic ranges
        df['Debt_to_to_Income_Ratio'] = np.interp(df['Debt_to_Income_Ratio'], (df['Debt_to_Income_Ratio'].min(), df['Debt_to_Income_Ratio'].max()), (0.1, 0.6))
        df['Credit_Score'] = np.interp(df['Credit_Score'], (df['Credit_Score'].min(), df['Credit_Score'].max()), (300, 850))

        return df
    ```

*   **Code Cell (Execution):**
    ```python
    credit_data = generate_credit_risk_data(n_samples=1000, random_state=42)
    print("First 5 rows of the synthetic credit data:")
    print(credit_data.head())
    print("\nClass distribution:")
    print(credit_data['loan_default'].value_counts())
    ```

*   **Markdown Cell (Explanation for Execution):**
    We have successfully generated a synthetic dataset of 1000 loan applications. The `make_classification` function was used with parameters set to create a challenging, non-linearly separable problem, mimicking real-world credit risk scenarios. The `weights` parameter was adjusted to simulate a class imbalance typical in default prediction (fewer defaults than non-defaults). The features `Debt_to_Income_Ratio` and `Credit_Score` have been scaled to more intuitive ranges. The initial rows and class distribution confirm the dataset's structure.

### Section 6: Splitting Data into Training and Test Sets

*   **Markdown Cell:**
    To properly evaluate our model's ability to generalize, it is crucial to split the dataset into two independent parts: a training set and a test set. The model will be trained exclusively on the training set, and its performance will then be assessed on the unseen test set. This simulates how the model would perform on new loan applications in the real world. We will use an 80/20 split, with 80% for training and 20% for testing.

*   **Code Cell (Function Implementation/Setup - Not Applicable)**

*   **Code Cell (Execution):**
    ```python
    X = credit_data[['Debt_to_Income_Ratio', 'Credit_Score']]
    y = credit_data['loan_default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Training set class distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"Test set class distribution:\n{y_test.value_counts(normalize=True)}")
    ```

*   **Markdown Cell (Explanation for Execution):**
    The dataset has been divided into training and test sets. By using `stratify=y`, we ensure that the class distribution (proportion of defaults vs. non-defaults) is preserved in both the training and test sets, which is important for imbalanced datasets. This split prepares our data for unbiased model evaluation.

### Section 7: Defining Evaluation Metrics for Credit Risk

*   **Markdown Cell:**
    For credit risk assessment, simply looking at accuracy can be misleading, especially with imbalanced datasets. Risk Managers are often more concerned with correctly identifying potential defaults (True Positives) and minimizing False Negatives (missing a default) or False Positives (incorrectly predicting a default). We will calculate:
    *   **Accuracy:** Overall correctness of predictions.
    *   **False Positive Rate (FPR):** The proportion of actual non-defaults that were incorrectly predicted as defaults. This is crucial for avoiding rejecting creditworthy applicants.
    $$
    \text{False Positive Rate (FPR)} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}
    $$
    *   **False Negative Rate (FNR):** The proportion of actual defaults that were incorrectly predicted as non-defaults. This is critical for avoiding lending to high-risk applicants.
    $$
    \text{False Negative Rate (FNR)} = \frac{\text{False Negatives}}{\text{False Negatives} + \text{True Positives}}
    $$
    Here, 'Positive' typically refers to the 'default' class (label 1).

*   **Code Cell (Function Implementation/Setup):**
    ```python
    def calculate_credit_risk_metrics(y_true, y_pred, model_name=""):
        """
        Calculates accuracy, False Positive Rate (FPR), and False Negative Rate (FNR).
        Assumes '1' is the positive class (default).
        """
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        # Assuming binary classification:
        # cm[0,0] = True Negatives (TN) - Correctly predicted non-defaults
        # cm[0,1] = False Positives (FP) - Incorrectly predicted defaults (actual non-defaults)
        # cm[1,0] = False Negatives (FN) - Incorrectly predicted non-defaults (actual defaults)
        # cm[1,1] = True Positives (TP) - Correctly predicted defaults

        TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0,0,0,0) # Handle edge cases for empty classes

        if (FP + TN) == 0:
            fpr = np.nan # Avoid division by zero
        else:
            fpr = FP / (FP + TN)

        if (FN + TP) == 0:
            fnr = np.nan # Avoid division by zero
        else:
            fnr = FN / (FN + TP)

        return {
            f"{model_name} Accuracy": accuracy,
            f"{model_name} FPR": fpr,
            f"{model_name} FNR": fnr
        }
    ```

*   **Code Cell (Execution):**
    ```python
    # No immediate execution here, this function will be called later for evaluation.
    print("Function `calculate_credit_risk_metrics` defined.")
    ```

*   **Markdown Cell (Explanation for Execution):**
    The `calculate_credit_risk_metrics` function has been defined. This custom function will be crucial for evaluating our SVM model, providing specific insights into the types of errors it makes, which is of paramount importance for credit risk analysis. We prioritize the 'default' class as the positive class for FPR and FNR calculation.

### Section 8: Visualizing the Initial Dataset

*   **Markdown Cell:**
    Let's visualize our synthetic credit risk dataset in a 2D scatter plot. This will give us an initial understanding of the data distribution and the inherent separability challenge before applying any machine learning models. We will plot 'Debt-to-Income Ratio' against 'Credit Score', coloring points by their 'loan_default' status.

*   **Code Cell (Function Implementation/Setup):**
    ```python
    def plot_data(X, y, title="Credit Risk Data Distribution"):
        """Plots the 2D dataset with different colors for each class."""
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, palette='coolwarm', s=80, alpha=0.7)
        plt.title(title)
        plt.xlabel(X.columns[0])
        plt.ylabel(X.columns[1])
        plt.legend(title='Loan Default', labels=['Non-Default (0)', 'Default (1)'])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
    ```

*   **Code Cell (Execution):**
    ```python
    plot_data(X, y, title="Synthetic Credit Risk Data: Debt-to-Income vs. Credit Score")
    ```

*   **Markdown Cell (Explanation for Execution):**
    The scatter plot visually represents the distribution of loan applicants based on their 'Debt-to-Income Ratio' and 'Credit Score', colored by their default status. We can observe that the classes are not perfectly linearly separable, indicating that a simple linear model might struggle and highlighting the need for a more complex boundary or allowing for some misclassifications.

### Section 9: Implementing the SVM Classifier and Decision Boundary Plotting

*   **Markdown Cell:**
    Now we will define a function to train the SVM classifier and visualize its decision boundary. This function will take the regularization parameter $C$ as input, train an `SVC` model with an RBF kernel, make predictions, and then plot the decision boundary along with the data points. It will also calculate and display the key credit risk metrics for both training and test sets.

*   **Code Cell (Function Implementation/Setup):**
    ```python
    def plot_svm_decision_boundary(C_param, gamma_param='scale', X_train_data=X_train, y_train_data=y_train, X_test_data=X_test, y_test_data=y_test):
        """
        Trains an SVM model, plots its decision boundary, and displays performance metrics.
        C_param: Regularization parameter for SVM.
        gamma_param: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        """
        # Train the SVM model
        svm_model = SVC(kernel='rbf', C=C_param, gamma=gamma_param, random_state=42)
        svm_model.fit(X_train_data, y_train_data)

        # Make predictions
        y_train_pred = svm_model.predict(X_train_data)
        y_test_pred = svm_model.predict(X_test_data)

        # Calculate metrics
        train_metrics = calculate_credit_risk_metrics(y_train_data, y_train_pred, "Train")
        test_metrics = calculate_credit_risk_metrics(y_test_data, y_test_pred, "Test")

        # Create plot
        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # Plot data points
        sns.scatterplot(x=X_train_data.iloc[:, 0], y=X_train_data.iloc[:, 1], hue=y_train_data,
                        palette='coolwarm', s=80, alpha=0.7, label='Training Data')
        # Overlay test data (distinguished marker/color for clarity)
        sns.scatterplot(x=X_test_data.iloc[:, 0], y=X_test_data.iloc[:, 1], hue=y_test_data,
                        palette='coolwarm', marker='X', s=100, alpha=0.7, label='Test Data')

        # Plot decision boundary
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                             np.linspace(ylim[0], ylim[1], 100))
        Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary and margins
        ax.contour(xx, yy, Z, colors=['gray', 'black', 'gray'], levels=[-1, 0, 1], alpha=0.7,
                   linestyles=['--', '-', '--'])
        # Highlight support vectors
        ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=200,
                   linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

        plt.title(f'SVM Decision Boundary (C={C_param}, gamma={gamma_param})')
        plt.xlabel(X_train_data.columns[0])
        plt.ylabel(X_train_data.columns[1])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        # Display metrics in a structured way
        metrics_df = pd.DataFrame([train_metrics, test_metrics])
        metrics_df = metrics_df.T # Transpose for better readability
        metrics_df.columns = ['Value']
        display(HTML(metrics_df.to_html(classes='table table-striped')))
    ```

*   **Code Cell (Execution):**
    ```python
    # This function will be called by the interactive widget. No direct execution here.
    print("Function `plot_svm_decision_boundary` defined.")
    ```

*   **Markdown Cell (Explanation for Execution):**
    The `plot_svm_decision_boundary` function has been created. It encapsulates the core logic for training an SVM, visualizing its decision boundary (including support vectors and margin lines), and reporting the calculated credit risk metrics. This function is designed to be highly reusable and will be central to our interactive demonstration.

### Section 10: Interactive Overfitting/Underfitting Demonstration

*   **Markdown Cell:**
    Now we will create an interactive demonstration using `ipywidgets`. You will be able to adjust the SVM regularization parameter $C$ using a slider and immediately observe its impact on the decision boundary, model complexity, and the training and test set performance metrics (Accuracy, FPR, FNR). This interactive experience is key to understanding the trade-off between bias and variance.

    *   **Small C values:** Lead to a larger margin, simpler decision boundary, potentially underfitting.
    *   **Large C values:** Lead to a smaller margin, more complex decision boundary, potentially overfitting.

*   **Code Cell (Function Implementation/Setup - Not Applicable):**

*   **Code Cell (Execution):**
    ```python
    # Create the slider for C parameter
    c_slider = FloatSlider(
        min=0.01,
        max=100.0,
        step=0.1,
        value=1.0, # Default C value
        description='C Parameter:',
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    # Link the slider to the plotting function using interactive
    interactive_plot = interactive(plot_svm_decision_boundary, C_param=c_slider, gamma_param='scale')

    # Display the interactive widget
    display(interactive_plot)
    ```

*   **Markdown Cell (Explanation for Execution):**
    An interactive slider is now available, allowing you to dynamically adjust the SVM's $C$ parameter. As you move the slider, the SVM model is re-trained, the decision boundary on the plot updates in real-time, and the training and test metrics are re-calculated and displayed. Start by observing the default $C=1.0$ and then experiment with much smaller values (e.g., $C=0.1$) and much larger values (e.g., $C=50.0$ or $C=100.0$). Pay close attention to how the decision boundary changes and how the training vs. test metrics diverge or converge.

### Section 11: Interpreting the Interactive Results: Diagnosing Overfitting and Underfitting

*   **Markdown Cell:**
    As you interact with the $C$ parameter, observe the following patterns:

    *   **Underfitting (Small $C$ values, e.g., $C < 1$):**
        *   **Decision Boundary:** The boundary will appear very smooth and simple, potentially ignoring many data points or failing to separate the classes well.
        *   **Metrics:** Both training accuracy and test accuracy will likely be low. The model is too simple to capture the underlying patterns in the data, leading to high bias. FPR and FNR might both be high, indicating the model is not reliable for identifying defaults or non-defaults.

    *   **Optimal Fit (Moderate $C$ values, e.g., $C \approx 1$ to $10$ for this dataset):**
        *   **Decision Boundary:** The boundary will show a reasonable level of complexity, separating most of the training data while maintaining a good margin.
        *   **Metrics:** Training accuracy will be high, and crucially, test accuracy will also be high and close to the training accuracy. This indicates good generalization. Both FPR and FNR should be acceptably low, showing a balanced performance in risk identification.

    *   **Overfitting (Large $C$ values, e.g., $C > 20$):**
        *   **Decision Boundary:** The boundary will become highly complex and "wiggly," trying to perfectly separate every single training data point, even noise. It might tightly encircle individual points.
        *   **Metrics:** Training accuracy will be very high (possibly 100%), but test accuracy will drop significantly compared to training accuracy. This indicates poor generalization to unseen data. While training FPR/FNR might be low, the test FPR/FNR could be problematic, especially if the model becomes overly sensitive to minor fluctuations in training data, misclassifying new applicants.

    This visual and quantitative analysis directly demonstrates the "bias-variance trade-off": a simple model has high bias (underfitting), while a complex model has high variance (overfitting). The ideal model finds a balance.

### Section 12: Analyzing Performance Across a Range of C Values

*   **Markdown Cell:**
    While interactive sliders are great for intuitive understanding, it's also valuable to systematically evaluate model performance across a predefined range of $C$ values. This allows us to plot training and test accuracies against $C$ and precisely identify the region where the model achieves the best generalization.

*   **Code Cell (Function Implementation/Setup):**
    ```python
    def analyze_c_range(C_values, X_train_data=X_train, y_train_data=y_train, X_test_data=X_test, y_test_data=y_test, gamma_param='scale'):
        """
        Trains SVMs for a range of C values and collects training and test metrics.
        C_values: A list or array of C values to test.
        """
        train_accuracies = []
        test_accuracies = []
        train_fprs = []
        test_fprs = []
        train_fnrs = []
        test_fnrs = []

        for C_val in C_values:
            svm_model = SVC(kernel='rbf', C=C_val, gamma=gamma_param, random_state=42)
            svm_model.fit(X_train_data, y_train_data)

            y_train_pred = svm_model.predict(X_train_data)
            y_test_pred = svm_model.predict(X_test_data)

            train_metrics = calculate_credit_risk_metrics(y_train_data, y_train_pred)
            test_metrics = calculate_credit_risk_metrics(y_test_data, y_test_pred)

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
    ```

*   **Code Cell (Execution):**
    ```python
    C_range = np.logspace(-2, 2, 50) # From 0.01 to 100, 50 points
    c_analysis_results = analyze_c_range(C_range)
    print("First 5 rows of C-parameter analysis results:")
    print(c_analysis_results.head())
    ```

*   **Markdown Cell (Explanation for Execution):**
    We have executed a comprehensive sweep of the $C$ parameter across a logarithmic range, from very small (0.01) to very large (100). For each $C$ value, an SVM model was trained, and its performance metrics on both training and test sets were recorded. This tabular summary provides the foundation for our next visualization, which will highlight the generalization sweet spot.

### Section 13: Visualizing Training vs. Test Accuracy for Different C Values

*   **Markdown Cell:**
    This plot is a crucial diagnostic tool for identifying the optimal $C$ parameter. It directly shows the relationship between model complexity (driven by $C$) and the model's ability to generalize.

    *   The point where **Test Accuracy** is maximized, and **Train Accuracy** and **Test Accuracy** are close, indicates the best balance between bias and variance, and thus the optimal generalization.
    *   A significant gap where **Train Accuracy** is high but **Test Accuracy** is low indicates overfitting.
    *   Low **Train Accuracy** and low **Test Accuracy** suggest underfitting.

*   **Code Cell (Function Implementation/Setup):**
    ```python
    def plot_accuracy_vs_c(results_df):
        """Plots training and test accuracy against C values."""
        plt.figure(figsize=(12, 7))
        plt.plot(results_df['C'], results_df['Train Accuracy'], label='Training Accuracy', marker='o', linestyle='--', alpha=0.7)
        plt.plot(results_df['C'], results_df['Test Accuracy'], label='Test Accuracy', marker='x', linestyle='-', alpha=0.9)

        plt.xscale('log') # Use logarithmic scale for C
        plt.xlabel('C Parameter (log scale)')
        plt.ylabel('Accuracy')
        plt.title('SVM Training vs. Test Accuracy across different C values')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 7))
        plt.plot(results_df['C'], results_df['Train FPR'], label='Training FPR', marker='o', linestyle='--', alpha=0.7, color='red')
        plt.plot(results_df['C'], results_df['Test FPR'], label='Test FPR', marker='x', linestyle='-', alpha=0.9, color='darkred')
        plt.plot(results_df['C'], results_df['Train FNR'], label='Training FNR', marker='o', linestyle='--', alpha=0.7, color='blue')
        plt.plot(results_df['C'], results_df['Test FNR'], label='Test FNR', marker='x', linestyle='-', alpha=0.9, color='darkblue')

        plt.xscale('log')
        plt.xlabel('C Parameter (log scale)')
        plt.ylabel('Rate')
        plt.title('SVM Training vs. Test FPR and FNR across different C values')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
    ```

*   **Code Cell (Execution):**
    ```python
    plot_accuracy_vs_c(c_analysis_results)
    ```

*   **Markdown Cell (Explanation for Execution):**
    The generated plots clearly demonstrate the behavior of our SVM model as the $C$ parameter changes. You can visually identify:
    *   The region where both training and test accuracies are low (underfitting).
    *   The "sweet spot" where test accuracy peaks and is close to training accuracy (optimal generalization).
    *   The region where training accuracy continues to rise or stays high, but test accuracy declines, indicating overfitting.
    The FPR and FNR plots provide further insight into the specific types of errors as $C$ varies, which is invaluable for a Risk Manager evaluating the practical implications of model choices.

### Section 14: Conclusion: Generalization in Credit Risk Management

*   **Markdown Cell:**
    This notebook has provided a hands-on exploration of overfitting and underfitting using an SVM model for credit risk assessment. We've seen how the regularization parameter $C$ directly influences model complexity and, consequently, its ability to generalize to unseen data.

    For Risk Managers, the key takeaway is that a model's performance on historical (training) data alone is insufficient. **Generalization** – the model's ability to accurately predict outcomes for new, unseen loan applications – is paramount. Overfitting can lead to unreliable predictions, potentially increasing exposure to defaults (high FNR on test data) or unnecessarily rejecting creditworthy applicants (high FPR on test data), both carrying significant financial implications.

    By carefully tuning parameters like $C$ and systematically evaluating performance on an independent test set, we can develop robust credit scoring models that strike the right balance between fitting the available data and maintaining predictive power in real-world scenarios, thereby adhering to the **structural risk minimization principle**.

