import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

st.set_page_config(page_title="Multi-Armed Bandit Optimizer", layout="wide")

st.title("🎯 Reinforcement Learning: Marketing Offer Optimization")
st.markdown("""
Instead of **A/B Testing** which wastes traffic on losing variants, this **Multi-Armed Bandit (MAB)** algorithm dynamically routes website traffic to the highest-converting Auto Loan offer in real-time. 
Click **"Simulate Traffic"** to watch the Epsilon-Greedy algorithm learn, explore, and eventually exploit the winning offer.
""")

offers = ["0% APR (High Cost)", "$0 Down Payment", "$1000 Cash Back"]
true_rates = [0.12, 0.15, 0.08] # Unknown to model

# Streamlit Session State Configuration
if 'pulls' not in st.session_state:
    st.session_state.pulls = np.zeros(3)
if 'successes' not in st.session_state:
    st.session_state.successes = np.zeros(3)
if 'step' not in st.session_state:
    st.session_state.step = 0

def epsilon_greedy(epsilon, q_values):
    if np.random.rand() < epsilon:
        return np.random.randint(3)
    else:
        return np.argmax(q_values)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Control Panel")
    visitors = st.slider("Visitors to Simulate per click", 10, 500, 100)
    eps_start = st.slider("Exploration Rate (Epsilon)", 0.01, 1.0, 0.20)
    
    if st.button("Simulate Website Traffic 🚀"):
        st.session_state.step += visitors
        eps = max(0.01, eps_start * (1 - st.session_state.step / 5000)) # Decay
        
        for _ in range(visitors):
            current_q = np.where(st.session_state.pulls > 0, st.session_state.successes / st.session_state.pulls, 1.0)
            action = epsilon_greedy(eps, current_q)
            reward = 1 if np.random.rand() < true_rates[action] else 0
            
            st.session_state.pulls[action] += 1
            st.session_state.successes[action] += reward

with col2:
    st.subheader("Algorithmic Performance")
    st.metric("Total Visitors Simulated", st.session_state.step)
    
    current_q = np.where(st.session_state.pulls > 0, st.session_state.successes / st.session_state.pulls, 0)
    
    # Traffic Allocation Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.style.use('dark_background')
    
    colors = ['#FF4E50', '#F9D423', '#00E1D9']
    
    ax1.bar(offers, current_q, color=colors, alpha=0.8)
    for idx, r in enumerate(true_rates):
        ax1.axhline(y=r, color=colors[idx], linestyle='--', alpha=0.5, label=f"True: {r*100}%")
    ax1.set_ylim(0, 0.2)
    ax1.set_title("Estimated Conversion Probability")
    
    # Pie Chart
    if sum(st.session_state.pulls) > 0:
        ax2.pie(st.session_state.pulls, labels=offers, colors=colors, autopct='%1.1f%%', startangle=90, 
                wedgeprops={'linewidth': 1, 'edgecolor': 'w'})
        ax2.set_title("RL Traffic Allocation (Exploitation vs Exploration)")
    
    fig.patch.set_facecolor('#0e1117')
    st.pyplot(fig)

    if st.button("Reset Simulation"):
        st.session_state.pulls = np.zeros(3)
        st.session_state.successes = np.zeros(3)
        st.session_state.step = 0
        st.rerun()
