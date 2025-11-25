
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
    st.markdown("""
    To effectively demonstrate overfitting and underfitting in a controlled environment,
    we will use a synthetic dataset. This dataset is designed to mimic key characteristics
    of credit risk data, specifically featuring two primary drivers of loan default:
    **Debt-to-Income Ratio** and **Credit Score**. The dataset will contain two distinct
    classes: 'Non-Default' (label 0) and 'Default' (label 1), with an imbalanced distribution
    to reflect real-world credit scenarios.

    The features have been scaled to be within typical ranges for these financial indicators.
    """)

    st.session_state.credit_data = generate_credit_risk_data(n_samples=1000, random_state=42)
    st.write("First 5 rows of the synthetic credit data:")
    st.dataframe(st.session_state.credit_data.head())
    st.write("\nClass distribution:")
    st.dataframe(st.session_state.credit_data['loan_default'].value_counts())

    st.markdown("## 6. Splitting Data into Training and Test Sets")
    st.markdown("""
    Before training any machine learning model, it is crucial to split the dataset
    into distinct training and test sets. This practice is fundamental for evaluating
    a model's ability to generalize to unseen data and detect potential overfitting.

    *   **Training Set:** Used to train the SVM model. The model learns patterns and decision boundaries from this data.
    *   **Test Set:** Used to evaluate the trained model's performance on data it has never seen before.
        This provides an unbiased estimate of the model's generalization capability.

    We will perform a stratified split to ensure that the proportion of loan default
    cases is maintained in both the training and test sets, which is particularly
    important for imbalanced datasets.
    """)

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
