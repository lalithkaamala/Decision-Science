import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Context: We have 3 Auto Loan Offers for our website visitors
# Offer A: 0% APR, but higher vehicle price
# Offer B: Zero Down Payment, standard APR
# Offer C: $1000 Cash Back, standard APR

true_conversion_rates = [0.12, 0.15, 0.08]  # Unknown to the algorithm
offers = ["Offer A (0% APR)", "Offer B (No Down)", "Offer C (Cash Back)"]
colors = ['#FF4E50', '#F9D423', '#00E1D9']

n_arms = len(true_conversion_rates)
iterations = 1000

# Arrays to keep track of Bandit state
pulls = np.zeros(n_arms)
successes = np.zeros(n_arms)
estimated_rates = np.zeros((iterations, n_arms))

def epsilon_greedy(epsilon, step, q_values):
    if np.random.rand() < epsilon:
        return np.random.randint(n_arms)
    else:
        return np.argmax(q_values)

# Setup figure
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

fig.suptitle('Multi-Armed Bandit: Auto-Lending Offer Optimization', color='white', fontsize=16)

def update(frame):
    # Simulate batch of 10 visitors per frame
    for _ in range(10):
        # Epsilon decays over time (Exploration -> Exploitation)
        epsilon = max(0.01, 0.5 * (1 - (frame * 10) / iterations))
        
        # Calculate current estimated success rates
        current_q = np.where(pulls > 0, successes / pulls, 0.5)  # Optimistic initial value
        
        # Choose action
        action = epsilon_greedy(epsilon, frame, current_q)
        
        # Observe reward (user converts or not based on true probability)
        reward = 1 if np.random.rand() < true_conversion_rates[action] else 0
        
        # Update trackers
        pulls[action] += 1
        successes[action] += reward
        
    current_q = np.where(pulls > 0, successes / pulls, 0)
    
    # Left subplot: Estimated Conversion Rates converging
    ax1.clear()
    ax1.bar(offers, current_q, color=colors, alpha=0.8)
    for idx, true_rate in enumerate(true_conversion_rates):
        ax1.axhline(y=true_rate, color=colors[idx], linestyle='--', xmax=(idx+0.8)/n_arms, xmin=(idx+0.2)/n_arms)
    ax1.set_ylim(0, 0.2)
    ax1.set_title(f"Estimated Conversion Rates (Visitors: {frame*10})")
    ax1.set_ylabel("Conversion Probability")
    
    # Right subplot: Traffic Distribution (Agent learning to exploit best offer)
    ax2.clear()
    ax2.pie(pulls + 1, labels=offers, colors=colors, autopct='%1.1f%%', startangle=90, 
            wedgeprops={'edgecolor': 'w', 'linewidth': 1, 'antialiased': True})
    ax2.set_title("Website Traffic Allocation")
    
    return [ax1, ax2]

print("Simulating Multi-Armed Bandit...")
anim = animation.FuncAnimation(fig, update, frames=100, interval=100, blit=False)

filename = 'mab_optimization.gif'
anim.save(filename, writer='pillow', fps=10)
print(f"✅ Saved animation -> {filename}")
