import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, bernoulli, binom, poisson, expon, gamma, beta, uniform, lognorm

st.set_page_config(page_title="Distribution Explorer", layout="centered")
st.title("Probability Distribution Visualizer")

# --- Sidebar ---
dist = st.sidebar.selectbox("Choose a Distribution", [
    "Gaussian (Normal)", "Bernoulli", "Binomial", "Poisson",
    "Exponential", "Gamma", "Beta", "Uniform", "Log-Normal"
])

num_samples = st.sidebar.slider("Sample Size (for histogram, if shown)", 100, 10000, 1000, 100)
show_hist = st.sidebar.checkbox("Overlay Histogram (where applicable)", value=False)

# --- Light Green Theme Colors ---
main_color = "#4CAF50"
hist_color = "#C8E6C9"

fig = go.Figure()

# --- Distribution Logic ---
if dist == "Gaussian (Normal)":
    mu = st.sidebar.slider("Mean (μ)", -10.0, 10.0, 0.0)
    sigma = st.sidebar.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    pdf = norm.pdf(x, mu, sigma)
    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF', line=dict(color=main_color)))
    if show_hist:
        samples = norm.rvs(mu, sigma, size=num_samples)
        fig.add_trace(go.Histogram(x=samples, histnorm='probability density', nbinsx=30,
                                   opacity=0.5, name='Histogram', marker_color=hist_color))

elif dist == "Bernoulli":
    p = st.sidebar.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01)
    x = [0, 1]
    pmf = bernoulli.pmf(x, p)
    fig.add_trace(go.Bar(x=x, y=pmf, name='PMF', marker_color=[main_color, hist_color],
                         text=[f"{v:.2f}" for v in pmf], textposition="outside"))

elif dist == "Binomial":
    n = st.sidebar.slider("Number of Trials (n)", 1, 100, 10)
    p = st.sidebar.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01)
    x = np.arange(0, n + 1)
    pmf = binom.pmf(x, n, p)
    fig.add_trace(go.Bar(x=x, y=pmf, name="PMF", marker_color=main_color))
    if show_hist:
        samples = binom.rvs(n, p, size=num_samples)
        fig.add_trace(go.Histogram(x=samples, histnorm='probability', nbinsx=n+1,
                                   opacity=0.5, name='Histogram', marker_color=hist_color))

elif dist == "Poisson":
    lam = st.sidebar.slider("Rate (λ)", 0.1, 30.0, 5.0, 0.1)
    x = np.arange(0, int(lam + 4*np.sqrt(lam)) + 1)
    pmf = poisson.pmf(x, lam)
    fig.add_trace(go.Bar(x=x, y=pmf, name="PMF", marker_color=main_color))
    if show_hist:
        samples = poisson.rvs(lam, size=num_samples)
        fig.add_trace(go.Histogram(x=samples, histnorm='probability', nbinsx=len(x),
                                   opacity=0.5, name='Histogram', marker_color=hist_color))

elif dist == "Exponential":
    lam = st.sidebar.slider("Rate (λ)", 0.1, 10.0, 1.0)
    scale = 1 / lam
    x = np.linspace(0, 10, 1000)
    pdf = expon.pdf(x, scale=scale)
    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF', line=dict(color=main_color)))
    if show_hist:
        samples = expon.rvs(scale=scale, size=num_samples)
        fig.add_trace(go.Histogram(x=samples, histnorm='probability density', nbinsx=30,
                                   opacity=0.5, name='Histogram', marker_color=hist_color))

elif dist == "Gamma":
    k = st.sidebar.slider("Shape (k)", 0.1, 20.0, 2.0)
    lam = st.sidebar.slider("Rate (λ)", 0.1, 5.0, 1.0)
    scale = 1 / lam
    x = np.linspace(0, gamma.ppf(0.99, k, scale=scale), 1000)
    pdf = gamma.pdf(x, k, scale=scale)
    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF', line=dict(color=main_color)))
    if show_hist:
        samples = gamma.rvs(k, scale=scale, size=num_samples)
        fig.add_trace(go.Histogram(x=samples, histnorm='probability density', nbinsx=30,
                                   opacity=0.5, name='Histogram', marker_color=hist_color))

elif dist == "Beta":
    alpha = st.sidebar.slider("Alpha (α)", 0.1, 10.0, 2.0)
    beta_val = st.sidebar.slider("Beta (β)", 0.1, 10.0, 2.0)
    x = np.linspace(0, 1, 1000)
    pdf = beta.pdf(x, alpha, beta_val)
    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF', line=dict(color=main_color)))
    if show_hist:
        samples = beta.rvs(alpha, beta_val, size=num_samples)
        fig.add_trace(go.Histogram(x=samples, histnorm='probability density', nbinsx=30,
                                   opacity=0.5, name='Histogram', marker_color=hist_color))

elif dist == "Uniform":
    a = st.sidebar.slider("Lower Bound (a)", -10.0, 10.0, 0.0)
    b = st.sidebar.slider("Upper Bound (b)", a + 0.1, a + 20.0, a + 1.0)
    x = np.linspace(a - (b-a)*0.2, b + (b-a)*0.2, 1000)
    pdf = uniform.pdf(x, loc=a, scale=b - a)
    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF', line=dict(color=main_color)))
    if show_hist:
        samples = uniform.rvs(loc=a, scale=b - a, size=num_samples)
        fig.add_trace(go.Histogram(x=samples, histnorm='probability density', nbinsx=30,
                                   opacity=0.5, name='Histogram', marker_color=hist_color))

elif dist == "Log-Normal":
    mu = st.sidebar.slider("Mean (μ)", -2.0, 3.0, 0.0)
    sigma = st.sidebar.slider("Standard Deviation (σ)", 0.1, 2.0, 0.5)
    x = np.linspace(0.001, 10, 1000)
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF', line=dict(color=main_color)))
    if show_hist:
        samples = lognorm.rvs(s=sigma, scale=np.exp(mu), size=num_samples)
        fig.add_trace(go.Histogram(x=samples, histnorm='probability density', nbinsx=30,
                                   opacity=0.5, name='Histogram', marker_color=hist_color))

# --- Final Plot Setup ---
fig.update_layout(
    title=f"{dist} Distribution",
    xaxis_title="x",
    yaxis_title="Probability Density / Mass",
    bargap=0.05,
    template="plotly_white",
    showlegend=True
)

st.plotly_chart(fig, use_container_width=True)
