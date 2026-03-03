import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, brier_score_loss

def simulate_sophisticated_auto_loan_data(n_samples=25000):
    np.random.seed(42)
    
    # 1. Macro-economic & regional factors
    unemployment_rate = np.random.uniform(3.5, 9.0, n_samples)
    inflation_rate = np.random.uniform(1.5, 7.5, n_samples)
    
    # 2. Borrower characteristics (with correlated features using Cholesky decomposition)
    # Correlation between FICO and Income
    cov_matrix = np.array([[10000, 3000], [3000, 15000]])
    L = np.linalg.cholesky(cov_matrix)
    Z = np.random.normal(0, 1, size=(n_samples, 2))
    correlated_features = np.dot(Z, L.T)
    
    fico_scores = (correlated_features[:, 0] + 650).clip(300, 850)
    incomes = (correlated_features[:, 1] * 10 + 60000).clip(20000, 300000)
    
    # 3. Loan characteristics
    vehicle_types = np.random.choice(['Sedan', 'SUV', 'Truck', 'Luxury'], size=n_samples, p=[0.4, 0.4, 0.15, 0.05])
    vehicle_age = np.random.exponential(scale=3, size=n_samples).clip(0, 15)
    
    # Base loan amount depends on income and vehicle type
    base_loan = incomes * np.random.uniform(0.1, 0.5, n_samples)
    loan_amounts = np.where(vehicle_types == 'Luxury', base_loan * 1.5, base_loan).clip(5000, 100000)
    
    ltv_ratios = np.random.normal(0.95, 0.15, n_samples).clip(0.3, 1.4)
    dti_ratios = np.random.normal(0.35, 0.1, n_samples).clip(0.05, 0.65)
    loan_terms = np.random.choice([36, 48, 60, 72, 84], size=n_samples, p=[0.1, 0.15, 0.35, 0.3, 0.1])
    
    # Non-linear probability of default synthesis
    # Capturing interactions (e.g., High LTV + Low FICO is very bad)
    logit_p = (-6.5 
               - 0.015 * (fico_scores - 600) 
               + 4.0 * (ltv_ratios ** 2)      # Non-linear LTV impact
               + 6.5 * (dti_ratios ** 1.5)    # Non-linear DTI impact
               + 0.2 * unemployment_rate 
               + 0.1 * inflation_rate
               + 0.05 * vehicle_age
               + 0.00001 * loan_amounts
               - 0.00002 * incomes
               + 1.5 * ((ltv_ratios > 1.1) & (fico_scores < 620)).astype(float) # Interaction term
               + 0.5 * (loan_terms == 84).astype(float)) # 84-month loans are riskier
               
    p_default = 1 / (1 + np.exp(-logit_p))
    defaults = np.random.binomial(1, p_default)
    
    df = pd.DataFrame({
        'FICO_Score': fico_scores,
        'Income': incomes,
        'Loan_Amount': loan_amounts,
        'DTI_Ratio': dti_ratios,
        'LTV_Ratio': ltv_ratios,
        'Loan_Term_Months': loan_terms,
        'Vehicle_Age_Years': vehicle_age,
        'Unemployment_Rate': unemployment_rate,
        'Inflation_Rate': inflation_rate,
        'Default': defaults
    })
    
    # Convert 'Vehicle_Type' to dummy variables
    vehicle_dummies = pd.get_dummies(vehicle_types, prefix='Vehicle')
    df = pd.concat([df, vehicle_dummies], axis=1)
    
    return df

def generate_insights_report(X_train, X_test, y_train, y_test, model, brier):
    print("Generating comprehensive evaluation report...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Setup multiple plots
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 12))
    
    # Panel 1: ROC Curve
    ax1 = fig.add_subplot(221)
    ax1.plot(fpr, tpr, color='cyan', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', color='white')
    ax1.set_ylabel('True Positive Rate', color='white')
    ax1.set_title('Receiver Operating Characteristic (ROC)', color='white')
    ax1.legend(loc="lower right", facecolor='black', edgecolor='white')
    
    # Panel 2: Precision-Recall Curve
    ax2 = fig.add_subplot(222)
    ax2.plot(recall, precision, color='#FF00FF', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlabel('Recall', color='white')
    ax2.set_ylabel('Precision', color='white')
    ax2.set_title('Precision-Recall Curve (Highly Imbalanced Target)', color='white')
    ax2.legend(loc="lower left", facecolor='black', edgecolor='white')

    # Panel 3: SHAP Summary Plot
    ax3 = fig.add_subplot(223)
    plt.sca(ax3)
    shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False, plot_size=None, color_bar=False)
    ax3.set_title(f'SHAP Values: Explaining Individual Loan Decisions\n(Brier Calibration Score: {brier:.4f})', color='white', pad=20)
    
    # Panel 4: SHAP Dependence Plot (Interactive LTV vs FICO insight)
    ax4 = fig.add_subplot(224)
    plt.sca(ax4)
    # Recreate a dependence plot manually for better styling integration
    ltv_idx = list(X_test.columns).index('LTV_Ratio')
    fico_idx = list(X_test.columns).index('FICO_Score')
    scatter = ax4.scatter(X_test['LTV_Ratio'], shap_values[:, ltv_idx], 
                          c=X_test['FICO_Score'], cmap='plasma', alpha=0.6, s=10)
    ax4.set_xlabel('LTV Ratio (Loan-to-Value)', color='white')
    ax4.set_ylabel('SHAP Value for LTV (Impact on Default Risk)', color='white')
    ax4.set_title('SHAP Dependence: LTV Ratio interaction with FICO Score', color='white')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('FICO Score', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    # Styling enhancements
    for ax in [ax1, ax2, ax3, ax4]:
        ax.spines['bottom'].set_color('#333333')
        ax.spines['top'].set_color('#333333')
        ax.spines['right'].set_color('#333333')
        ax.spines['left'].set_color('#333333')
        ax.tick_params(colors='white')
        
    plt.tight_layout()
    plt.savefig('advanced_credit_risk_dash.png', dpi=300, facecolor='#050510')
    print("✅ Advanced Credit Risk Dashboard saved: advanced_credit_risk_dash.png")

def main():
    print("Initiating Algorithmic Underwriting Engine (Credit Risk)...")
    df = simulate_sophisticated_auto_loan_data()
    print(f"Data Generation Vol: {df.shape[0]} borrowers, features populated.")
    print(f"Base Default Rate: {df['Default'].mean() * 100:.2f}%")
    
    # Feature Engineering explicitly defined
    X = df.drop('Default', axis=1)
    y = df['Default']
    
    # Split out validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # In a real pipeline, we would run Optuna here for hyperparameter tuning.
    # For CI efficiency, we define an optimal complex parameter set directly.
    print("Training Monotonic Constrained XGBoost Model (Algorithmic Fairness)...")
    
    # Monotonic constraints ensure predictable regulatory behavior (e.g., higher FICO always reduces risk)
    # 1 indicates increasing default risk, -1 indicates decreasing default risk, 0 is unconstrained
    monotone_constraints = {
        'FICO_Score': -1, 
        'Income': -1, 
        'Loan_Amount': 1, 
        'DTI_Ratio': 1, 
        'LTV_Ratio': 1,
        'Unemployment_Rate': 1,
        'Inflation_Rate': 1,
        'Vehicle_Age_Years': 1
    }
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        monotone_constraints=monotone_constraints,
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Calibration Assessment
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, y_pred_proba)
    
    generate_insights_report(X_train, X_test, y_train, y_test, model, brier)

if __name__ == '__main__':
    main()
