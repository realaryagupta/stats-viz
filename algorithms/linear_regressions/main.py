import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("🔍 Linear Regression Visualizer")
st.markdown("""
Use the sliders to adjust the **slope (m)** and **intercept (b)** of the regression line.  
See how they affect the fit to the data points.
""")

# Sidebar for slope and intercept
st.sidebar.header("Adjust Model Parameters")
slope = st.sidebar.slider("Slope (m)", -10.0, 10.0, 1.0, 0.1)
intercept = st.sidebar.slider("Intercept (b)", -20.0, 20.0, 0.0, 0.5)

# Generate synthetic data
np.random.seed(42)
x = np.linspace(0, 10, 50)
true_slope = 2
true_intercept = 5
noise = np.random.normal(0, 3, size=x.shape)
y = true_slope * x + true_intercept + noise

# Predicted line
y_pred = slope * x + intercept

# Plotting
fig, ax = plt.subplots()
ax.scatter(x, y, color="blue", label="Data")
ax.plot(x, y_pred, color="red", label=f"Line: y = {slope:.2f}x + {intercept:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Display current equation
st.markdown(f"### 📈 Current Regression Line:  \n**y = {slope:.2f}x + {intercept:.2f}**")
