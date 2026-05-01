import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate monthly Auto Loan Default Rates for two states
months = np.arange(1, 25)

# Intervention at Month 12: State Y dropped 'Minimum FICO' from 650 to 600
intervention_month = 12

np.random.seed(1)
# State X (Control Group) - Default rate stays steady around 2%
control_defaults = 0.02 + np.sin(months * 0.5) * 0.002 + np.random.normal(0, 0.001, 24)

# State Y (Treatment Group) - Follows control before intervention, spikes after
treatment_defaults = control_defaults + 0.005 # Baseline difference

# Intervention Effect
true_causal_impact = 0.015
treatment_defaults[intervention_month:] += true_causal_impact + (months[intervention_month:] - intervention_month) * 0.001

# Counterfactual (What would State Y have looked like without the policy change?)
counterfactual = control_defaults + 0.005

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(months, control_defaults * 100, label='Control (State X)', color='#00FFFF', linewidth=2, marker='o')
ax.plot(months, treatment_defaults * 100, label='Treatment (State Y) - Lowered FICO Req', color='#FF00FF', linewidth=2, marker='o')
ax.plot(months[intervention_month:], counterfactual[intervention_month:] * 100, linestyle='--', color='gray', label='Counterfactual (If no policy change)', linewidth=2)

ax.axvline(x=intervention_month, color='white', linestyle='-.', alpha=0.5, label='Policy Intervention (Month 12)')

# Highlight the causal gap
ax.fill_between(months[intervention_month:], counterfactual[intervention_month:] * 100, treatment_defaults[intervention_month:] * 100, color='#FF00FF', alpha=0.2, label='Causal Impact')

ax.set_title("Causal Inference (Difference-in-Differences)\nImpact of lowering FICO requirements on Auto Loan Defaults", color='white', pad=20)
ax.set_ylabel("Default Rate (%)", color='white')
ax.set_xlabel("Months", color='white')
ax.grid(True, color='#333333', linestyle=':')
ax.legend(facecolor='black', edgecolor='none')

plt.tight_layout()
plt.savefig('causal_impact_plot.png', dpi=300, facecolor='#050510')
print("✅ Saved -> causal_impact_plot.png")
