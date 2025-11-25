id: 692602d63bc9bc6a310b833c_user_guide
summary: Support vector machines User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Credit Risk Overfit Identifier: A Practical Guide to Model Generalization

## 1. Welcome to the Credit Risk Overfit Identifier
Duration: 0:05:00

Welcome, Risk Managers, to this interactive lab designed to demystify the crucial concepts of **overfitting** and **underfitting** in the context of credit risk modeling. In financial institutions, accurately predicting loan defaults is vital. Our ability to assess borrower creditworthiness directly impacts lending decisions and portfolio management. However, even the most sophisticated predictive models can lead to costly errors if they fail to generalize well to new, unseen credit applications. This often happens due to overfitting or underfitting.

This application provides a hands-on experience using a **Support Vector Machine (SVM)** classifier. By interactively adjusting the SVM's key regularization parameter, $C$, you will gain an intuitive understanding of how model complexity influences its ability to learn from historical data and make reliable predictions on future data.

<aside class="positive">
<b>Why is this important for Risk Managers?</b>
Building models that generalize well is fundamental to robust credit risk management. A model that overfits might perform excellently on past data but drastically fail on new applicants, leading to unexpected losses. An underfit model, conversely, might be too simplistic to capture real risks. This lab helps you visually and quantitatively understand this critical balance.
</aside>

### Learning Objectives

By the end of this codelab, you will be able to:
*   Clearly differentiate between overfitting and underfitting in predictive models.
*   Understand the role of the SVM $C$ parameter in controlling model complexity.
*   Interpret key credit risk metrics (Accuracy, False Positive Rate, False Negative Rate) for diagnosing model performance issues.
*   Visually identify the impact of model complexity on decision boundaries.
*   Appreciate the importance of model generalization for stable and reliable credit risk models.

### Support Vector Machine (SVM) Fundamentals

Support Vector Machines are powerful algorithms used for classification. At their core, SVMs aim to find the "best" hyperplane that separates different classes of data points. The "best" hyperplane is one that has the largest distance (margin) to the nearest training data point of any class.

For cases where data isn't linearly separable, SVMs use a "kernel trick" to implicitly transform the data into a higher-dimensional space where a linear separation might be possible. We will use the **Radial Basis Function (RBF) kernel**, a popular choice for capturing non-linear relationships.

#### The Role of the Regularization Parameter $C$

The parameter $C$ is a critical component in SVMs. It governs the **trade-off between correctly classifying training points and maximizing the margin**.

*   **Small $C$ values** (e.g., $C \approx 0.01$): The model tolerates more misclassifications on the training data to achieve a larger margin. This results in a **simpler model** that can be prone to **underfitting** if it doesn't capture the underlying patterns sufficiently.
*   **Large $C$ values** (e.g., $C \approx 100$): The model places a high penalty on misclassifications, trying to classify almost all training points correctly. This leads to a **more complex model** with a narrower margin, which tends to **overfit** the training data by memorizing its specific nuances and noise.

The goal is to find an optimal $C$ value that allows the model to generalize well, performing robustly on both seen and unseen data.

To follow along, use the sidebar navigation in the Streamlit application. First, ensure you are on the "Introduction & SVM Fundamentals" page. Then, navigate to the next page as indicated in the following steps.

<aside class="positive">
<b>Tip:</b> Don't worry about the underlying code! This codelab focuses purely on how to use the application to understand the concepts.
</aside>

## 2. Preparing Your Data: Synthetic Credit Risk Dataset
Duration: 0:03:00

Before we can train an SVM, we need data! This section of the application (`Generating & Splitting Data` in the sidebar) will create a synthetic credit risk dataset and prepare it for model training and evaluation.

### Generating the Synthetic Credit Risk Dataset

The application generates a synthetic dataset designed to mimic real-world credit risk scenarios. It features two crucial financial indicators: **Debt-to-Income Ratio** and **Credit Score**. The dataset includes two classes: 'Non-Default' (label 0) and 'Default' (label 1), intentionally made imbalanced to reflect the rarity of defaults in real credit portfolios.

1.  **Navigate to "Generating & Splitting Data"** in the sidebar.
2.  Observe the generated data. The application displays the first few rows of the dataset and its class distribution. You'll notice the 'loan_default' column indicates whether a loan defaulted (1) or not (0).

<aside class="console">
First 5 rows of the synthetic credit data:
    Debt_to_Income_Ratio  Credit_Score  loan_default
0              0.370428    448.971550             0
1              0.551817    653.606990             1
2              0.311756    525.753896             0
3              0.198357    540.384214             0
4              0.334032    453.619054             0
...
Class distribution:
loan_default
0    850
1    150
Name: count, dtype: int64
</aside>

### Splitting Data into Training and Test Sets

A fundamental practice in machine learning is to divide your dataset into separate training and test sets.

*   The **training set** is used by the SVM model to learn patterns and define its decision boundary.
*   The **test set** consists of data the model has never seen before. It is crucial for evaluating how well the model generalizes and if it will perform reliably on new credit applications.

The application automatically performs a **stratified split**, ensuring that the proportion of 'Default' cases is preserved in both the training and test sets. This is especially important for imbalanced datasets like ours.

The application displays the size and class distribution of both the training and test sets.

<aside class="console">
Training set size: 800 samples
Test set size: 200 samples
Training set class distribution:

loan_default
0    0.85
1    0.15
Name: proportion, dtype: float64
Test set class distribution:

loan_default
0    0.85
1    0.15
Name: proportion, dtype: float64
</aside>

This setup ensures that our evaluation of the SVM model will be fair and indicative of its real-world performance.

## 3. Understanding Performance: Metrics and Initial Visuals
Duration: 0:04:00

Now that our data is ready, let's establish how we'll measure our SVM's performance and take a first look at the data distribution. This section (`Evaluation Metrics & Initial Data Viz` in the sidebar) introduces the specific metrics relevant to credit risk and visualizes the dataset.

1.  **Navigate to "Evaluation Metrics & Initial Data Viz"** in the sidebar.

### Defining Evaluation Metrics for Credit Risk

In credit risk, overall "accuracy" isn't always sufficient. We need to understand the *types* of errors a model makes, as different errors have different business impacts.

We will focus on:
*   **Accuracy:** The overall proportion of correctly classified instances (both defaults and non-defaults).
*   **False Positive Rate (FPR):** This tells us the proportion of actual **non-defaults** that were incorrectly predicted as **defaults**.
*   **False Negative Rate (FNR):** This tells us the proportion of actual **defaults** that were incorrectly predicted as **non-defaults**.

#### Mathematical Definitions
For clarity, let's define these metrics. We consider 'loan default' (label 1) as the 'Positive' class.

**False Positive Rate (FPR)**
$$
\text{False Positive Rate (FPR)} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}
$$
A **False Positive** occurs when a customer who **does not default** (True Negative) is predicted to **default**. This might lead to rejecting a creditworthy applicant.

**False Negative Rate (FNR)**
$$
\text{False Negative Rate (FNR)} = \frac{\text{False Negatives}}{\text{False Negatives} + \text{True Positives}}
$$
A **False Negative** occurs when a customer who **does default** (True Positive) is predicted to **not default**. This can result in significant financial losses for the institution.

<aside class="negative">
<b>Important:</b> In credit risk, minimizing the **False Negative Rate (FNR)** is often a primary objective. We want to avoid lending to customers who will default. High FNR implies significant financial exposure.
</aside>

### Visualizing the Initial Dataset

Before any model is trained, it's beneficial to visualize the data. This plot displays 'Debt-to-Income Ratio' against 'Credit Score', with points colored by their 'loan_default' status.

Observe the distribution of default (1) and non-default (0) cases. You'll likely see some overlap, indicating that a simple straight line might not perfectly separate them, highlighting the need for a more sophisticated model like an SVM with an RBF kernel. This visualization gives us a baseline understanding before we introduce the SVM's decision boundary.

<aside class="positive">
<b>Observe:</b> Look for areas where the two classes are distinct and areas where they overlap. This overlap is where the model will face challenges and where the decision boundary becomes critical.
</aside>

## 4. Interactive Exploration: SVM Regularization and Decision Boundaries
Duration: 0:08:00

This is the core interactive section of the codelab! Here, you will train an SVM model and directly observe the impact of the regularization parameter $C$ on its complexity and performance.

1.  **Navigate to "Interactive SVM Demonstration"** in the sidebar.

### Implementing the SVM Classifier and Decision Boundary Plotting

The application trains an SVM with an RBF kernel using the training data we prepared. It then visualizes the decision boundary, which is the line (or curve) that the SVM uses to separate the classes. Crucially, it also plots the **Support Vectors** – these are the data points that lie closest to the decision boundary and directly influence its position and shape.

Alongside the visual, the application calculates and displays the Accuracy, FPR, and FNR for both the training and test sets.

### Interactive Overfitting/Underfitting Demonstration

Now, let's get interactive!

<aside class="positive">
Use the slider labeled **"Select SVM Regularization Parameter $C$"** to adjust the $C$ value.
</aside>

As you move the slider, pay close attention to the following:

*   **The Decision Boundary:** How does its shape and smoothness change? Does it become more "wiggly" or smoother?
*   **The Margins:** How wide or narrow is the separation between the classes?
*   **Support Vectors:** Which data points are chosen as support vectors? Do more points become support vectors as $C$ changes?
*   **Training vs. Test Metrics:** Compare the Accuracy, FPR, and FNR between the training and test sets. This is the **most important indicator** for diagnosing overfitting or underfitting.

#### Interpreting the Interactive Results: Diagnosing Overfitting and Underfitting

Let's break down what you should observe for different $C$ values:

*   **Underfitting (Small $C$ values, e.g., $C \approx 0.01$ to $0.1$):**
    *   **Visuals:** The decision boundary will be very smooth and generalized, often failing to effectively separate the classes. The margin will be wide. You'll see many misclassified points, even within the training set.
    *   **Metrics:**
        *   **Low Training Accuracy:** The model struggles to fit the training data.
        *   **Low Test Accuracy:** It performs poorly on unseen data.
        *   **High FPR and FNR on both training and test sets:** The model makes many errors in both directions.
    *   **Diagnosis:** The model is too simple; it suffers from high **bias**. It hasn't learned enough from the training data.

*   **Optimal Fit (Moderate $C$ values, e.g., $C \approx 1.0$):**
    *   **Visuals:** The decision boundary effectively separates the classes, striking a good balance between complexity and smoothness. The margin is reasonable, and support vectors are strategically placed.
    *   **Metrics:**
        *   **High Training Accuracy:** The model fits the training data well.
        *   **High Test Accuracy (similar to Training Accuracy):** Crucially, it performs almost as well on unseen data.
        *   **Low FPR and FNR on both training and test sets, with values close to each other:** This indicates a good balance in correctly identifying positive and negative classes and strong generalization.
    *   **Diagnosis:** The model has found a good balance between bias and variance.

*   **Overfitting (Large $C$ values, e.g., $C \approx 10$ to $100$):**
    *   **Visuals:** The decision boundary becomes highly convoluted and "wiggly," trying to perfectly classify every single training point, even outliers. The margin will be very narrow, or even collapse around individual points.
    *   **Metrics:**
        *   **Very High Training Accuracy:** The model performs exceptionally well on the training data, often near 100%.
        *   **Significantly Lower Test Accuracy:** The model performs poorly on unseen data, showing it has essentially memorized the training set.
        *   **Low Training FPR/FNR but High Test FPR/FNR:** This is a classic sign of overfitting. The model looks great on what it's seen but fails dramatically on new data. For instance, it might have a very low training FNR (identifies almost all defaults in training) but a very high test FNR (misses many defaults in the test set).
    *   **Diagnosis:** The model is too complex; it suffers from high **variance**. It has learned the noise and specific idiosyncrasies of the training data rather than the general underlying relationships.

Experiment with the slider to see these effects firsthand. Try to find a $C$ value that gives you the best balance between training and test performance, particularly focusing on keeping the Test FNR low without significantly increasing the Test FPR.

## 5. Comprehensive Analysis: C-Parameter Performance Trends
Duration: 0:05:00

While the interactive demonstration is great for an intuitive feel, it's also valuable to systematically analyze model performance across a broader range of $C$ values. This section (accessible via `C-Parameter Range Analysis` in the sidebar) automates this process and presents the results in informative plots.

1.  **Navigate to "C-Parameter Range Analysis"** in the sidebar.

### Analyzing Performance Across a Range of C Values

The application performs an automated sweep of the $C$ parameter, from very small (0.01) to very large (100). For each $C$ value, it trains an SVM and calculates the Accuracy, FPR, and FNR for both the training and test sets. This systematic approach allows us to observe trends and identify optimal ranges for $C$.

The initial table displayed shows a snippet of these results.

### Visualizing Training vs. Test Accuracy for Different C Values

The plots generated in this section provide a powerful summary:

*   **Accuracy Plot:** This plot shows how both training and test accuracy change as the $C$ parameter increases (on a logarithmic scale).
    *   **Underfitting region:** On the left side (small $C$), both training and test accuracies will likely be low.
    *   **Overfitting region:** On the right side (large $C$), training accuracy will be very high, but test accuracy will drop significantly, creating a large gap between the two lines.
    *   **Optimal region:** Look for the point where training accuracy is high, and test accuracy is also high and very close to the training accuracy. This is where the model generalizes best.

*   **FPR and FNR Plot:** This plot is particularly insightful for credit risk, showing the trends of False Positive Rate and False Negative Rate for both training and test sets across different $C$ values.
    *   Notice how for small $C$, both FPR and FNR might be high.
    *   As $C$ increases, the training FNR will likely decrease dramatically (the model gets better at catching defaults in the training set). However, pay close attention to the **Test FNR**. If it starts to climb or remains high while training FNR is low, that's a clear sign of overfitting, where the model is failing to identify actual defaults in new data.
    *   Similarly, observe the trends for FPR.

These visualizations provide a comprehensive view of the bias-variance trade-off, helping you identify the optimal range for $C$ that balances model fit on training data with strong generalization capability on unseen data. The ideal $C$ value minimizes the gap between training and test performance for key metrics, especially FNR for credit risk.

## 6. Conclusion: Building Robust Credit Risk Models
Duration: 0:02:00

We have reached the end of this interactive codelab. You can find this conclusion on the main `QuLab` page of the application, below the navigation section.

This lab has demonstrated the paramount importance of **model generalization** in credit risk management. By interactively adjusting the SVM's regularization parameter $C$, you've seen how model complexity directly impacts its ability to make accurate and reliable predictions on new, unseen data.

Here’s a quick recap of the key insights:

*   **Underfitting:** Occurs with models that are too simple (low $C$), failing to capture essential patterns. This results in poor performance on both training and test datasets.
*   **Overfitting:** Happens with models that are too complex (high $C$), memorizing the training data, including its noise. This leads to excellent performance on training data but drastically poor performance on new data.
*   **Optimal Generalization:** Achieves a strong balance, performing well on both training and test sets, indicating a robust and reliable model for real-world application.

For Risk Managers, the true value of a predictive model lies not in its ability to perfectly explain historical data, but in its capacity to accurately forecast future events. Understanding and actively mitigating the risks of overfitting and underfitting is fundamental to building predictive models that:
*   Support sound lending decisions.
*   Minimize financial exposure.
*   Ensure the long-term stability of credit portfolios.

Effective credit risk management requires continuous monitoring of model performance on new data and recalibrating models as necessary. We hope this interactive experience has equipped you with a deeper, more intuitive understanding of these critical concepts.
