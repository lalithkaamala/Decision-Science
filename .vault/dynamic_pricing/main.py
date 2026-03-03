import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def elasticity_of_demand(apr, base_accept=0.8, elasticity=7.5):
    """Logistic function: As APR increases, probability of accepting the loan drops."""
    return 1 / (1 + np.exp(elasticity * (apr - 0.05))) * base_accept

def profit_margin(apr, cost_of_funds=0.03, default_rate=0.015):
    """Profit calculation (simplified): Yield - Cost of Capital - Default Rate"""
    return apr - cost_of_funds - default_rate

# Create Grid of Interest Rates and FICO Scores
aprs = np.linspace(0.04, 0.12, 100) # 4% to 12% APR
ficos = np.linspace(600, 800, 100)
A, F = np.meshgrid(aprs, ficos)

# Base probability is higher for high FICO, elasticity is sharper (they shop around)
default_r = 0.1 - (F - 500) * 0.0003 # Defaults drop as FICO increases
base_P = 0.5 + (F - 600) * 0.002
elastic = 5 + (F - 600) * 0.05 # High FICO borrowers are highly sensitive to APR

P_accept = 1 / (1 + np.exp(elastic * (A - 0.07))) * base_P
Margins = A - 0.03 - default_r

# Expected Value
Expected_Profit = P_accept * Margins

fig = plt.figure(figsize=(12, 8))
plt.style.use('dark_background')
ax = fig.add_subplot(111, projection='3d')

# Plot the Expected Value Surface
surf = ax.plot_surface(F, A*100, Expected_Profit*10000, cmap='plasma', alpha=0.9, edgecolor='none')

ax.set_xlabel('FICO Score', color='cyan', labelpad=15)
ax.set_ylabel('Offered APR (%)', color='cyan', labelpad=15)
ax.set_zlabel('Expected Profit ($) per $10k Loan', color='cyan', labelpad=15)

ax.set_title("Optimal Loan Pricing: Elasticity vs. Margin", color='white', size=16, pad=20)

plt.tight_layout()
plt.savefig('optimization_surface.png', dpi=300, facecolor='#0a0a0a')
print("✅ Saved -> optimization_surface.png")
