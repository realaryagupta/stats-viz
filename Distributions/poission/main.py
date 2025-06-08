import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

st.title("Poisson Distribution Visualizer")

# Sidebar control
lam = st.sidebar.slider("Rate (λ - Expected Events)", 0.1, 30.0, 5.0, 0.1)

# Generate data
x_max = int(lam + 4 * np.sqrt(lam)) + 1  # reasonable max x based on lambda
x = np.arange(0, x_max)
pmf = poisson.pmf(x, lam)

# Plot
fig, ax = plt.subplots()
bars = ax.bar(x, pmf, color="orange", edgecolor="black")
ax.set_title(f"Poisson PMF (λ = {lam:.2f})")
ax.set_xlabel("Number of Events")
ax.set_ylabel("Probability")
ax.set_ylim(0, max(pmf) * 1.2)

# Annotate bars
for i in range(len(x)):
    if pmf[i] > 0.01:
        ax.text(x[i], pmf[i] + 0.01, f"{pmf[i]:.2f}", ha="center")

st.pyplot(fig)
