import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

st.title("Exponential Distribution Visualizer")

# Sidebar control
lam = st.sidebar.slider("Rate (λ)", 0.1, 10.0, 1.0, 0.1)
scale = 1 / lam  # scipy's expon uses scale = 1/λ

# Generate data
x = np.linspace(0, 10, 1000)
pdf = expon.pdf(x, scale=scale)

# Plot
fig, ax = plt.subplots()
ax.plot(x, pdf, color='purple', label=f'λ = {lam:.2f}')
ax.fill_between(x, pdf, alpha=0.3, color='purple')
ax.set_title("Exponential PDF")
ax.set_xlabel("x (Time)")
ax.set_ylabel("Density")
ax.set_xlim(0, 10)
ax.set_ylim(0, max(pdf) * 1.1)
ax.legend()

st.pyplot(fig)
