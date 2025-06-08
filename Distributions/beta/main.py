import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

st.title("Beta Distribution Visualizer with Histogram")

# Sidebar controls
alpha = st.sidebar.slider("Alpha (α)", 0.1, 10.0, 2.0, 0.1)
beta_val = st.sidebar.slider("Beta (β)", 0.1, 10.0, 2.0, 0.1)
num_samples = st.sidebar.slider("Number of Samples", 100, 10000, 1000, 100)

# Generate theoretical PDF
x = np.linspace(0, 1, 1000)
pdf = beta.pdf(x, alpha, beta_val)

# Generate random samples
samples = beta.rvs(alpha, beta_val, size=num_samples)

# Plot
fig, ax = plt.subplots()

# Histogram of samples (normalized to match PDF scale)
ax.hist(samples, bins=30, density=True, alpha=0.5, color='orange', label='Sample Histogram')

# Plot PDF
ax.plot(x, pdf, color='blue', label=f'Beta PDF (α={alpha:.2f}, β={beta_val:.2f})')
ax.fill_between(x, pdf, alpha=0.2, color='blue')

# Axis and labels
ax.set_title("Beta Distribution PDF + Sample Histogram")
ax.set_xlabel("x (Proportion)")
ax.set_ylabel("Density")
ax.set_xlim(0, 1)
ax.set_ylim(0, max(pdf.max(), np.histogram(samples, bins=30, density=True)[0].max()) * 1.1)
ax.legend()

st.pyplot(fig)