import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

st.title("Binomial Distribution Visualizer")

# Sidebar controls
n = st.sidebar.slider("Number of Trials (n)", 1, 100, 10, 1)
p = st.sidebar.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01)

# Generate data
x = np.arange(0, n + 1)
pmf = binom.pmf(x, n, p)

# Plot
fig, ax = plt.subplots()
bars = ax.bar(x, pmf, color="skyblue", edgecolor="black")
ax.set_title(f"Binomial PMF (n = {n}, p = {p:.2f})")
ax.set_xlabel("Number of Successes")
ax.set_ylabel("Probability")
ax.set_ylim(0, 1)

# Annotate bars
for i in range(len(x)):
    if pmf[i] > 0.01:
        ax.text(x[i], pmf[i] + 0.01, f"{pmf[i]:.2f}", ha="center")

st.pyplot(fig)
