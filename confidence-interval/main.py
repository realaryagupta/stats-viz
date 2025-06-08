import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Streamlit app title and layout
st.set_page_config(page_title="Confidence Interval Simulation", layout="wide")
st.title("🎯 Confidence Interval Simulation")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")
sample_size = st.sidebar.slider("Sample Size", 2, 100, 10)
population_mean = st.sidebar.slider("Population Mean", 0, 100, 50)
population_std_dev = st.sidebar.slider("Population Standard Deviation", 1, 100, 15)
num_simulations = st.sidebar.slider("Number of Simulations", 1, 1000, 100)
confidence_level = st.sidebar.slider("Confidence Level (%)", 50, 99, 95)
method = st.sidebar.selectbox("Method", ["Z with sigma", "Z with s", "T with s"])

# Run simulations
np.random.seed(42)
conf_int_captured = 0
lower_bounds = []
upper_bounds = []

for _ in range(num_simulations):
    sample = np.random.normal(loc=population_mean, scale=population_std_dev, size=sample_size)
    sample_mean = np.mean(sample)
    sample_std_dev = np.std(sample, ddof=1)

    # Calculate margin of error
    if method == "Z with sigma":
        critical_value = stats.norm.ppf((1 + confidence_level / 100) / 2)
        margin_of_error = critical_value * (population_std_dev / np.sqrt(sample_size))
    elif method == "Z with s":
        critical_value = stats.norm.ppf((1 + confidence_level / 100) / 2)
        margin_of_error = critical_value * (sample_std_dev / np.sqrt(sample_size))
    else:  # "T with s"
        critical_value = stats.t.ppf((1 + confidence_level / 100) / 2, df=sample_size - 1)
        margin_of_error = critical_value * (sample_std_dev / np.sqrt(sample_size))

    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    lower_bounds.append(lower_bound)
    upper_bounds.append(upper_bound)

    if lower_bound <= population_mean <= upper_bound:
        conf_int_captured += 1

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
for i in range(num_simulations):
    color = "skyblue" if lower_bounds[i] <= population_mean <= upper_bounds[i] else "tomato"
    ax.plot([i, i], [lower_bounds[i], upper_bounds[i]], color=color, linewidth=2)

# Population mean reference line
ax.axhline(population_mean, color="red", linestyle="--", label="Population Mean")

# Style and labels
ax.set_title("Confidence Intervals Across Simulations", fontsize=16)
ax.set_xlabel("Simulation Index")
ax.set_ylabel("Interval Value")
ax.set_xlim(-1, num_simulations)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="upper right")
st.pyplot(fig)

# Results
accuracy = 100 * conf_int_captured / num_simulations
st.markdown(f"""
### 📊 Results:
- **Captured Mean** in `{conf_int_captured}` out of `{num_simulations}` simulations
- **Capture Rate:** `{accuracy:.2f}%`
- **Method Used:** `{method}`
""")
