
import streamlit as st

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

st.markdown("""
In this lab, **"Credit Risk Overfit Identifier: Understanding Model Complexity and Generalization"**,
we explore the critical concepts of overfitting and underfitting in the context of credit risk modeling.
Using an interactive Support Vector Machine (SVM) demonstration, Risk Managers can visually and quantitatively
understand how model complexity, controlled by the SVM regularization parameter $C$, impacts the model's
ability to generalize from training data to unseen credit applications.

Navigate through the sections using the sidebar to delve into SVM fundamentals, generate synthetic credit data,
visualize initial data distributions, and interactively adjust model complexity to observe its effects on decision boundaries
and key credit risk metrics (Accuracy, False Positive Rate, False Negative Rate) across both training and test datasets.
Finally, we will analyze model performance over a range of $C$ values to identify optimal generalization.
This hands-on experience aims to equip you with a deeper understanding of building robust and reliable credit risk models.
""")

page = st.sidebar.selectbox(
    label="Navigation",
    options=[
        "Introduction & SVM Fundamentals",
        "Generating & Splitting Data",
        "Evaluation Metrics & Initial Data Viz",
        "Interactive SVM Demonstration",
        "C-Parameter Range Analysis"
    ]
)

if page == "Introduction & SVM Fundamentals":
    from application_pages.page_1_intro_svm import main
    main()
elif page == "Generating & Splitting Data":
    from application_pages.page_2_data_generation import main
    main()
elif page == "Evaluation Metrics & Initial Data Viz":
    from application_pages.page_3_metrics_and_initial_viz import main
    main()
elif page == "Interactive SVM Demonstration":
    from application_pages.page_4_interactive_svm import main
    main()
elif page == "C-Parameter Range Analysis":
    from application_pages.page_5_c_range_analysis import main
    main()


# Conclusion for the overall application can be placed here or in a separate page if preferred
st.markdown("## 14. Conclusion: Generalization in Credit Risk Management")
st.markdown("""
This interactive lab has demonstrated the critical importance of model generalization in credit risk management.
Through the lens of Support Vector Machines and the regularization parameter $C$, we've observed how a model's
complexity directly influences its ability to perform accurately on new, unseen data.

*   **Underfitting** (too simple, low $C$) leads to models that fail to capture essential patterns, performing poorly on both training and test sets.
*   **Overfitting** (too complex, high $C$) results in models that memorize the training data, exhibiting excellent performance there but failing drastically on new data.
*   **Optimal Generalization** (balanced $C$) achieves a strong performance across both training and test sets, indicating a robust and reliable model.

For Risk Managers, the key takeaway is that a model's true value lies not in its ability to perfectly explain historical data,
but in its capacity to accurately predict future events. Understanding and mitigating the risks of overfitting and underfitting
is fundamental to building predictive models that support sound lending decisions, minimize financial exposure, and ensure the long-term stability of credit portfolios.
Continuously monitoring model performance on new data and recalibrating as necessary are essential practices for effective credit risk management.
""")
