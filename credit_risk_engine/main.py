import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.stats import norm

def simulate_auto_loan_data(n_samples=5000):
    np.random.seed(42)
    # Simulate auto loan application features
    fico_scores = np.random.normal(680, 50, n_samples).clip(300, 850)
    incomes = np.random.lognormal(mean=np.log(60000), sigma=0.5, size=n_samples).clip(20000, 250000)
    loan_amounts = np.random.normal(25000, 8000, n_samples).clip(5000, 80000)
    dti_ratios = np.random.beta(2, 5, n_samples) * 0.8
    ltv_ratios = np.random.normal(0.95, 0.15, n_samples).clip(0.5, 1.3)
    
    # Hidden true probability of default based on a logistic function
    logit_p = (-4.0 
               - 0.05 * (fico_scores - 600) 
               + 2.5 * ltv_ratios 
               + 4.0 * dti_ratios 
               - 0.00001 * incomes 
               + 0.00005 * loan_amounts)
    
    p_default = 1 / (1 + np.exp(-logit_p))
    defaults = np.random.binomial(1, p_default)
    
    return pd.DataFrame({
        'FICO_Score': fico_scores,
        'Income': incomes,
        'Loan_Amount': loan_amounts,
        'DTI_Ratio': dti_ratios,
        'LTV_Ratio': ltv_ratios,
        'Default': defaults
    })

def main():
    print("Generating Synthetic Auto Loan Data...")
    df = simulate_auto_loan_data()
    
    X = df.drop('Default', axis=1)
    y = df['Default']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Probability of Default Model...")
    model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, max_depth=4, learning_rate=0.05, n_estimators=100)
    model.fit(X_train, y_train)
    
    # Feature Importance Visualization
    importances = model.feature_importances_
    features = X.columns
    indices = np.argsort(importances)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color='cyan', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([features[i] for i in indices], color='white')
    ax.set_xlabel('Relative Importance (XGBoost Weight)', color='white')
    ax.set_title("Credit Risk Engine: Features driving Auto Loan Default", color='white', pad=20)
    
    # Add neon styling border and glow
    ax.spines['bottom'].set_color('#111111')
    ax.spines['top'].set_color('#111111')
    ax.spines['right'].set_color('#111111')
    ax.spines['left'].set_color('#111111')
    
    plt.tight_layout()
    plt.savefig('credit_risk_visualization.png', dpi=300, facecolor='#050510')
    print("Saved -> credit_risk_visualization.png")

if __name__ == '__main__':
    main()
