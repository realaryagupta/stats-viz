import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

st.title("Bernoulli Distribution Visualizer")

# Sidebar control
p = st.sidebar.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01)

# Bernoulli outcomes
x = [0, 1]
pmf = bernoulli.pmf(x, p)

# Plot
fig, ax = plt.subplots()
bars = ax.bar(x, pmf, width=0.4, color=["red", "green"], tick_label=["0 (Failure)", "1 (Success)"])
ax.set_ylim(0, 1)
ax.set_title(f"Bernoulli PMF (p = {p:.2f})")
ax.set_ylabel("Probability")

# Annotate bars
for bar, prob in zip(bars, pmf):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{prob:.2f}', ha='center')

st.pyplot(fig)
