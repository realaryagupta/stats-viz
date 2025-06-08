import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.title("Gaussian Distribution Visualizer")

# Sidebar controls
mean = st.sidebar.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1)
sigma = st.sidebar.slider("Standard Deviation (σ)", 0.1, 3.0, 1.0, 0.1)

# Fixed x range
x = np.linspace(-10, 10, 1000)
y = norm.pdf(x, mean, sigma)

# Plot
fig, ax = plt.subplots()
ax.plot(x, y, label=f'N({mean:.2f}, {sigma**2:.2f})', color='blue')
ax.fill_between(x, y, alpha=0.2, color='blue')

# Lock axis limits
ax.set_xlim(-10, 10)
ax.set_ylim(0, 0.5)  

ax.set_title("Gaussian Distribution (Fixed Axes)")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()

st.pyplot(fig)
