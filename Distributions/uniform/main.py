import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform

st.title("Uniform Distribution Visualizer with Histogram")

# Sidebar controls
a = st.sidebar.slider("Lower Bound (a)", -10.0, 10.0, 0.0, 0.1)
b = st.sidebar.slider("Upper Bound (b)", -10.0, 20.0, 1.0, 0.1)
num_samples = st.sidebar.slider("Number of Samples", 100, 10000, 1000, 100)

# Ensure b > a
if b <= a:
    st.error("Upper bound (b) must be greater than lower bound (a).")
else:
    # Generate data
    x = np.linspace(a - (b - a) * 0.1, b + (b - a) * 0.1, 1000)
    pdf = uniform.pdf(x, loc=a, scale=b - a)
    samples = uniform.rvs(loc=a, scale=b - a, size=num_samples)

    # Plot
    fig, ax = plt.subplots()

    # Histogram
    ax.hist(samples, bins=30, density=True, alpha=0.5, color='orange', label='Sample Histogram')

    # PDF
    ax.plot(x, pdf, color='blue', label=f'Uniform PDF [{a:.2f}, {b:.2f}]')
    ax.fill_between(x, pdf, alpha=0.2, color='blue')

    # Axis settings
    ax.set_title("Uniform Distribution PDF + Sample Histogram")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_xlim(a - (b - a) * 0.2, b + (b - a) * 0.2)
    ax.set_ylim(0, max(pdf.max(), np.histogram(samples, bins=30, density=True)[0].max()) * 1.1)
    ax.legend()

    st.pyplot(fig)
