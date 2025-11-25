
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
    classes: 