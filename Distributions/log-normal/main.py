import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

st.title("Log-Normal Distribution Visualizer")

# Sidebar controls
mu = st.sidebar.slider("Mean (μ) of underlying normal", -2.0, 3.0, 0.0, 0.1)
sigma = st.sidebar.slider("Standard Deviation (σ)", 0.1, 2.0, 0.5, 0.1)

# Generate x values and log-normal PDF
x = np.linspace(0.001, 10, 1000)  # Avoid 0 to prevent log issues
pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

# Plot
fig, ax = plt.subplots()
ax.plot(x, pdf, color='darkgreen', label=f'μ = {mu:.2f}, σ = {sigma:.2f}')
ax.fill_between(x, pdf, alpha=0.3, color='green')

# Axis labels and limits
ax.set_title("Log-Normal Distribution PDF")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.set_xlim(0, 10)
ax.set_ylim(0, max(pdf) * 1.1)
ax.legend()

st.pyplot(fig)
