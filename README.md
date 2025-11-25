# Credit Risk Overfit Identifier: Understanding Model Complexity and Generalization (QuLab)

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

This interactive Streamlit application, **"Credit Risk Overfit Identifier: Understanding Model Complexity and Generalization"**, is a hands-on educational tool designed specifically for Risk Managers. It demystifies the critical concepts of **overfitting** and **underfitting** in predictive modeling, using a Support Vector Machine (SVM) classifier applied to a synthetic credit default prediction scenario.

Financial institutions depend on robust models to assess creditworthiness and manage risk. This application allows users to manipulate the SVM's key regularization parameter, **$C$**, to observe in real-time how model complexity impacts its ability to generalize from historical data to new, unseen credit applications. Through visual decision boundaries and quantitative performance metrics, users will gain an intuitive understanding of the bias-variance trade-off, a cornerstone for building reliable and stable credit risk models.

## Features

This application offers the following key functionalities:

*   **Introduction to Overfitting/Underfitting & SVM Fundamentals**: Comprehensive explanation of these core concepts, including the mathematical formulation of SVMs and the crucial role of the regularization parameter $C$.
*   **Synthetic Credit Risk Data Generation**: Generates a two-feature (Debt-to-Income Ratio, Credit Score) synthetic dataset mimicking real-world credit default scenarios, complete with an imbalanced class distribution.
*   **Data Splitting**: Demonstrates the essential practice of splitting data into training and test sets using stratified sampling to ensure robust model evaluation.
*   **Credit Risk Specific Evaluation Metrics**: Focuses on relevant metrics for credit risk, including Accuracy, False Positive Rate (FPR), and False Negative Rate (FNR), with clear mathematical definitions and business implications.
*   **Initial Data Visualization**: Provides a visual overview of the synthetic dataset distribution before model training, highlighting class separability.
*   **Interactive SVM Demonstration**:
    *   Allows users to interactively adjust the SVM regularization parameter $C$ via a slider.
    *   Real-time visualization of the SVM's decision boundary, margin, and support vectors as $C$ changes.
    *   Simultaneous display of classification metrics (Accuracy, FPR, FNR) for both training and test sets.
    *   Guided interpretation to diagnose underfitting, optimal fit, and overfitting based on visual and quantitative feedback.
*   **C-Parameter Range Analysis**: Conducts an automated sweep across a wide range of $C$ values, plotting training vs. test accuracy, FPR, and FNR to systematically identify optimal generalization points and illustrate the bias-variance trade-off trends.

## Getting Started

Follow these instructions to set up and run the application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/credit-risk-overfit-identifier.git
    cd credit-risk-overfit-identifier
    ```
    *(Note: Replace `your-username/credit-risk-overfit-identifier.git` with the actual repository URL if available.)*

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    Create a `requirements.txt` file in the root directory of your project with the following content:
    ```
    streamlit>=1.0
    numpy>=1.20
    pandas>=1.3
    scikit-learn>=1.0
    matplotlib>=3.4
    seaborn>=0.11
    ```
    Then install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application:

1.  Ensure your virtual environment is activated.
2.  Navigate to the root directory of the cloned repository in your terminal.
3.  Execute the following command:
    ```bash
    streamlit run app.py
    ```

This will open the application in your default web browser (usually at `http://localhost:8501`).

**Instructions for Use:**

*   **Navigation**: Use the sidebar on the left to navigate between different sections of the lab project.
*   **Interactive SVM Demonstration**: On the "Interactive SVM Demonstration" page, use the slider to adjust the SVM regularization parameter $C$. Observe how the decision boundary, support vectors, and performance metrics (for both training and test sets) change in real-time.
*   **Analysis**: Pay close attention to the visual differences and metric discrepancies between training and test sets to understand underfitting, optimal fitting, and overfitting.

## Project Structure

The project is organized into modular pages for clarity and maintainability:

```
credit-risk-overfit-identifier/
├── application_pages/
│   ├── page_1_intro_svm.py             # Introduction, learning objectives, and SVM fundamentals
│   ├── page_2_data_generation.py       # Synthetic data generation and train/test split
│   ├── page_3_metrics_and_initial_viz.py # Definition of evaluation metrics and initial data visualization
│   ├── page_4_interactive_svm.py       # Interactive SVM C-parameter tuning and decision boundary plotting
│   └── page_5_c_range_analysis.py      # Automated analysis of C-parameter across a range
├── app.py                              # Main Streamlit application entry point and navigation logic
└── requirements.txt                    # List of Python dependencies
```

## Technology Stack

The application is built using the following technologies:

*   **Streamlit**: For creating interactive web applications with Python.
*   **Python**: The primary programming language.
*   **NumPy**: Fundamental package for numerical computing.
*   **Pandas**: For data manipulation and analysis.
*   **Scikit-learn**: A robust machine learning library for SVM implementation, data generation (`make_classification`), data splitting (`train_test_split`), and performance metrics (`accuracy_score`, `confusion_matrix`).
*   **Matplotlib**: For creating static, animated, and interactive visualizations.
*   **Seaborn**: A data visualization library based on matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.

## Contributing

As a lab project, direct contributions might not be formally managed, but feedback and suggestions are always welcome.

If you wish to contribute to similar projects:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes and commit them (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: Create a `LICENSE` file in your repository if it doesn't exist, typically containing the MIT license text.)*

## Contact

For any questions, suggestions, or issues, please feel free to:

*   Open an issue in the GitHub repository.
*   Contact QuantUniversity at [info@quantuniversity.com](mailto:info@quantuniversity.com) or visit [www.quantuniversity.com](https://www.quantuniversity.com).

---
**QuantUniversity - Empowering Financial Professionals with AI & Data Science.**
