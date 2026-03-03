import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# Using the simulation function from main.py to create demo model behavior
st.set_page_config(page_title="Credit Risk Decision Engine", layout="wide", initial_sidebar_state="expanded")

st.title("🏦 Algorithmic Underwriting: Fair Credit Risk Engine")
st.markdown("""
This **Decision Science Dashboard** visualizes the real-time probability of default (PD) for an Auto Loan applicant. 
Using an XGBoost model with **Monotonic Constraints**, it ensures regulatory fairness (e.g., higher FICO implies lower predictive risk). Adjust the borrower characteristics on the left to see the AI's transparent reasoning.
""")

@st.cache_resource
def load_synthetic_model_and_data():
    from main import simulate_sophisticated_auto_loan_data
    df = simulate_sophisticated_auto_loan_data(n_samples=5000)
    X = df.drop('Default', axis=1)
    y = df['Default']
    
    # Train constraint model
    monotone_constraints = {
        'FICO_Score': -1, 'Income': -1, 'Loan_Amount': 1, 'DTI_Ratio': 1, 
        'LTV_Ratio': 1, 'Unemployment_Rate': 1, 'Inflation_Rate': 1, 'Vehicle_Age_Years': 1
    }
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, 
                              monotone_constraints=monotone_constraints, use_label_encoder=False)
    model.fit(X, y, verbose=False)
    return model, X

model, X_train = load_synthetic_model_and_data()

st.sidebar.header("Borrower Profile")
fico = st.sidebar.slider("FICO Score", 300, 850, 680)
income = st.sidebar.number_input("Annual Income ($)", 20000, 300000, 60000)
loan_amt = st.sidebar.number_input("Loan Amount ($)", 5000, 100000, 25000)
dti = st.sidebar.slider("Debt-To-Income (DTI) %", 0.05, 0.70, 0.35)
ltv = st.sidebar.slider("Loan-To-Value (LTV)", 0.5, 1.5, 0.95)
term = st.sidebar.selectbox("Loan Term (Months)", [36, 48, 60, 72, 84], index=2)
v_age = st.sidebar.slider("Vehicle Age (Years)", 0, 15, 3)
unemp = st.sidebar.slider("Regional Unemployment Rate (%)", 3.0, 15.0, 5.0)
inf = st.sidebar.slider("Regional Inflation Rate (%)", 1.0, 10.0, 3.0)

v_type = st.sidebar.selectbox("Vehicle Type", ["Sedan", "SUV", "Truck", "Luxury"])
v_sedan = 1 if v_type == 'Sedan' else 0
v_suv = 1 if v_type == 'SUV' else 0
v_truck = 1 if v_type == 'Truck' else 0
v_luxury = 1 if v_type == 'Luxury' else 0

input_df = pd.DataFrame({
    'FICO_Score': [fico],
    'Income': [income],
    'Loan_Amount': [loan_amt],
    'DTI_Ratio': [dti],
    'LTV_Ratio': [ltv],
    'Loan_Term_Months': [term],
    'Vehicle_Age_Years': [v_age],
    'Unemployment_Rate': [unemp],
    'Inflation_Rate': [inf],
    'Vehicle_Luxury': [v_luxury],
    'Vehicle_SUV': [v_suv],
    'Vehicle_Sedan': [v_sedan],
    'Vehicle_Truck': [v_truck]
})

# Make prediction
prob_default = model.predict_proba(input_df)[0][1]

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Decision Output")
    st.metric("Probability of Default", f"{prob_default * 100:.2f}%")
    
    threshold = 0.15 # 15% cutoff
    if prob_default > threshold:
        st.error(f"🔴 **DECLINED**: Risk exceeds {threshold*100}% threshold.")
    else:
        st.success(f"🟢 **APPROVED**: Auto-Decision engine clear.")

with col2:
    st.subheader("Algorithmic Transparency (SHAP)")
    st.markdown("This force plot explains exactly how the model arrived at this decision by calculating the marginal mathematical contribution of each feature.")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)
    
    # SHAP Waterfall Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('dark_background')
    import shap.plots
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                         base_values=explainer.expected_value, 
                                         data=input_df.iloc[0], 
                                         feature_names=input_df.columns), show=False)
    # Dark mode theming for matplotlib output in streamlit
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    st.pyplot(fig)
