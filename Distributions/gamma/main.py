import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

st.title("Gamma Distribution Visualizer")

# Sidebar controls
k = st.sidebar.slider("Shape (k)", 0.5, 20.0, 2.0, 0.5)
lam = st.sidebar.slider("Rate (λ)", 0.1, 5.0, 1.0, 0.1)
scale = 1 / lam  # scipy uses scale = 1 / λ

# Generate data
x_max = gamma.ppf(0.99, k, scale=scale)
x = np.linspace(0, x_max, 1000)
pdf = gamma.pdf(x, k, scale=scale)

# Plot
fig, ax = plt.subplots()
ax.plot(x, pdf, color='green', label=f'k = {k:.2f}, λ = {lam:.2f}')
ax.fill_between(x, pdf, alpha=0.3, color='green')
ax.set_title("Gamma Distribution PDF")
ax.set_xlabel("x (Time)")
ax.set_ylabel("Density")
ax.set_xlim(0, x_max)
ax.set_ylim(0, max(pdf) * 1.1)
ax.legend()

st.pyplot(fig)
