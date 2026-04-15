import matplotlib.pyplot as plt
import networkx as nx

# Define State Space for an Auto-Loan Collections Process
# Each state represents how late the borrower is.
states = ["Current", "15_Days_Late", "30_Days_Late", "60_Days_Late", "90_Days_Late", "Repo_Order", "Resolved"]
colors = ['#00FF88', '#F9D423', '#FFA000', '#FF4E50', '#8B0000', '#4A0E4E', '#00FFFF']
actions = {
    "Current": [("Wait", "15_Days_Late", 0.05), ("Wait", "Current", 0.95)],
    "15_Days_Late": [("Email Reminder", "Current", 0.5), ("Email Reminder", "30_Days_Late", 0.5)],
    "30_Days_Late": [("Call Borrower", "Resolved", 0.4), ("Call Borrower", "60_Days_Late", 0.6)],
    "60_Days_Late": [("Offer Restructure", "Current", 0.3), ("Offer Restructure", "90_Days_Late", 0.7)],
    "90_Days_Late": [("Final Notice", "Resolved", 0.1), ("Final Notice", "Repo_Order", 0.9)],
    "Repo_Order": [("Execute Repo", "Resolved", 1.0)],
    "Resolved": []
}

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(14, 8))

G = nx.DiGraph()

# Add Nodes
for idx, s in enumerate(states):
    G.add_node(s, level=idx)

# Add Edges (Actions)
edge_labels = {}
for s, transitions in actions.items():
    for (action, next_state, prob) in transitions:
        if action != "Wait":
            G.add_edge(s, next_state, weight=prob, label=f"{action}\n(p={prob})")
            edge_labels[(s, next_state)] = f"{action}\n(p={prob})"

# Circular Layout with "Current" at top
pos = nx.spring_layout(G, k=2.5, iterations=100) # Slightly spaced out

# Draw Nodes
node_colors = [colors[states.index(n)] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, edgecolors='w', linewidths=2)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="black")

# Draw Edges
nx.draw_networkx_edges(G, pos, edge_color='cyan', arrowsize=25, arrowstyle='->', width=2.0, alpha=0.8, connectionstyle='arc3,rad=0.2')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='white', font_size=9, label_pos=0.4, rotate=False, bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

plt.title("Markov Decision Process: Auto Loan Collections Policy", color='white', size=16, pad=20)
plt.axis('off')
plt.savefig('mdp_policy_graph.png', dpi=300, facecolor='#050510')
print("✅ Saved -> mdp_policy_graph.png")
