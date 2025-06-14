import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm, bernoulli, binom, poisson, expon, gamma, beta, uniform, lognorm
import pandas as pd

# Page configuration with custom styling
st.set_page_config(
    page_title="Distribution Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------- Global ---------- */
html, body, [class*="st-"] {
    color: black !important;          /* all text black */
    font-family: 'Inter', sans-serif;
}

.main { padding-top: 2rem; }

.stApp { background: white; }

/* ---------- Tables ---------- */
.stDataFrame, .stTable {
    background-color: white !important;
    color: black !important;
}

.stDataFrame th, .stTable th {
    background-color: #f8f9fa !important;
    color: black !important;
}

.stDataFrame td, .stTable td {
    background-color: white !important;
    color: black !important;
    border-color: #e9ecef !important;
}

/* ---------- Header ---------- */
.main-header {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(0, 0, 0, 0.08);
    text-align: center;
}

.main-title {
    font-size: 3rem;
    font-weight: 700;
    color: black;                     /* was gradient */
    margin-bottom: 0.5rem;
}

.subtitle {
    font-size: 1.2rem;
    color: black;                     /* was #666 */
    font-weight: 400;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background-color: #f0f0f0;
    border-right: 1px solid #ccc;
    backdrop-filter: blur(20px);
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    color: black;
}

/* ---------- Cards & Info ---------- */
.metric-card,
.chart-container,
.info-box {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(0, 0, 0, 0.08);
    color: black;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
}

/* Statistics */
.stat-container { color: black; }
.stat-value     { font-size: 1.8rem; font-weight: 700; color: black; }
.stat-label     { font-size: 0.9rem;  color: black; }

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white !important;          /* changed back to white for better contrast */
    border: none;
    border-radius: 25px;
    padding: 0.5rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

/* ---------- Sliders & Selectboxes ---------- */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white; /* Added to make text white */
}

.stSelectbox > div > div {
    background: white;
    border-radius: 10px;
    border: 2px solid rgba(102, 126, 234, 0.2);
    color: black !important; /* Changed to black for better readability */
}

/* Hide Streamlit default bits */
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Main title
st.markdown("""
<div class="main-header">
    <h1 class="main-title">📊 Distribution Explorer</h1>
    <p class="subtitle">Interactive Probability Distribution Visualizer</p>
</div>
""", unsafe_allow_html=True)

# Create sidebar with enhanced styling
with st.sidebar:
    st.markdown("### 🎛️ Controls")
    
    # Distribution selection
    dist = st.selectbox(
        "Choose a Distribution",
        [
            "Gaussian (Normal)", "Bernoulli", "Binomial", "Poisson",
            "Exponential", "Gamma", "Beta", "Uniform", "Log-Normal"
        ],
        help="Select the probability distribution to visualize"
    )
    
    st.markdown("---")
    
    # Sample size
    num_samples = st.slider(
        "Sample Size",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        help="Number of samples for histogram overlay"
    )
    
    # Histogram toggle
    show_hist = st.checkbox(
        "Show Histogram Overlay",
        value=False,
        help="Overlay sample histogram on the theoretical distribution"
    )
    
    st.markdown("---")

# Distribution info dictionary
dist_info = {
    "Gaussian (Normal)": {
        "description": "Bell-shaped curve describing many natural phenomena. Symmetric around the mean.",
        "use_cases": "Heights, test scores, measurement errors",
        "parameters": "Mean (μ) and Standard Deviation (σ)"
    },
    "Bernoulli": {
        "description": "Models a single trial with two outcomes: success (1) or failure (0).",
        "use_cases": "Coin flips, pass/fail tests, yes/no questions",
        "parameters": "Probability of success (p)"
    },
    "Binomial": {
        "description": "Number of successes in n independent Bernoulli trials.",
        "use_cases": "Quality control, survey responses, clinical trials",
        "parameters": "Number of trials (n) and Success probability (p)"
    },
    "Poisson": {
        "description": "Models the number of events occurring in a fixed interval.",
        "use_cases": "Website visits, phone calls, defects per unit",
        "parameters": "Rate parameter (λ)"
    },
    "Exponential": {
        "description": "Models the time between events in a Poisson process.",
        "use_cases": "Customer service times, equipment failure",
        "parameters": "Rate parameter (λ)"
    },
    "Gamma": {
        "description": "Generalizes the exponential distribution. Models waiting times.",
        "use_cases": "Insurance claims, rainfall amounts",
        "parameters": "Shape (k) and Rate (λ)"
    },
    "Beta": {
        "description": "Defined on [0,1]. Often used to model proportions.",
        "use_cases": "Success rates, proportions, probabilities",
        "parameters": "Alpha (α) and Beta (β)"
    },
    "Uniform": {
        "description": "All values in the interval are equally likely.",
        "use_cases": "Random number generation, modeling uncertainty",
        "parameters": "Lower bound (a) and Upper bound (b)"
    },
    "Log-Normal": {
        "description": "If ln(X) is normally distributed, then X follows log-normal.",
        "use_cases": "Stock prices, income distribution, particle sizes",
        "parameters": "Mean (μ) and Standard Deviation (σ) of log"
    }
}

# Enhanced color scheme
colors = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#f093fb',
    'success': '#4CAF50',
    'warning': '#FF9800',
    'info': '#2196F3',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

# Create the main plot
fig = go.Figure()

# Parameters and plotting logic for each distribution
if dist == "Gaussian (Normal)":
    with st.sidebar:
        mu = st.slider("Mean (μ)", -10.0, 10.0, 0.0, 0.1)
        sigma = st.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0, 0.1)
    
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    pdf = norm.pdf(x, mu, sigma)
    
    fig.add_trace(go.Scatter(
        x=x, y=pdf, mode='lines', name='PDF',
        line=dict(color=colors['primary'], width=3),
        fill='tonexty', fillcolor=f"rgba(102, 126, 234, 0.1)"
    ))
    
    if show_hist:
        samples = norm.rvs(mu, sigma, size=num_samples)
        fig.add_trace(go.Histogram(
            x=samples, histnorm='probability density', nbinsx=50,
            opacity=0.6, name='Histogram', 
            marker_color=colors['accent']
        ))
    
    # Calculate statistics
    mean_val = mu
    var_val = sigma**2
    std_val = sigma
    skew_val = 0
    
elif dist == "Bernoulli":
    with st.sidebar:
        p = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01)
    
    x = [0, 1]
    pmf = bernoulli.pmf(x, p)
    
    fig.add_trace(go.Bar(
        x=x, y=pmf, name='PMF',
        marker_color=[colors['primary'], colors['secondary']],
        text=[f"{v:.3f}" for v in pmf],
        textposition="outside"
    ))
    
    mean_val = p
    var_val = p * (1 - p)
    std_val = np.sqrt(var_val)
    skew_val = (1 - 2*p) / np.sqrt(p * (1-p)) if p != 0 and p != 1 else 0

elif dist == "Binomial":
    with st.sidebar:
        n = st.slider("Number of Trials (n)", 1, 100, 10)
        p = st.slider("Probability of Success (p)", 0.0, 1.0, 0.5, 0.01)
    
    x = np.arange(0, n + 1)
    pmf = binom.pmf(x, n, p)
    
    fig.add_trace(go.Bar(
        x=x, y=pmf, name="PMF",
        marker_color=colors['primary']
    ))
    
    if show_hist:
        samples = binom.rvs(n, p, size=num_samples)
        fig.add_trace(go.Histogram(
            x=samples, histnorm='probability', nbinsx=n+1,
            opacity=0.6, name='Histogram',
            marker_color=colors['accent']
        ))
    
    mean_val = n * p
    var_val = n * p * (1 - p)
    std_val = np.sqrt(var_val)
    skew_val = (1 - 2*p) / np.sqrt(n * p * (1-p)) if p != 0 and p != 1 else 0

elif dist == "Poisson":
    with st.sidebar:
        lam = st.slider("Rate (λ)", 0.1, 30.0, 5.0, 0.1)
    
    x = np.arange(0, int(lam + 4*np.sqrt(lam)) + 1)
    pmf = poisson.pmf(x, lam)
    
    fig.add_trace(go.Bar(
        x=x, y=pmf, name="PMF",
        marker_color=colors['primary']
    ))
    
    if show_hist:
        samples = poisson.rvs(lam, size=num_samples)
        fig.add_trace(go.Histogram(
            x=samples, histnorm='probability', nbinsx=len(x),
            opacity=0.6, name='Histogram',
            marker_color=colors['accent']
        ))
    
    mean_val = lam
    var_val = lam
    std_val = np.sqrt(lam)
    skew_val = 1 / np.sqrt(lam)

elif dist == "Exponential":
    with st.sidebar:
        lam = st.slider("Rate (λ)", 0.1, 10.0, 1.0, 0.1)
    
    scale = 1 / lam
    x = np.linspace(0, 10/lam, 1000)
    pdf = expon.pdf(x, scale=scale)
    
    fig.add_trace(go.Scatter(
        x=x, y=pdf, mode='lines', name='PDF',
        line=dict(color=colors['primary'], width=3),
        fill='tonexty', fillcolor=f"rgba(102, 126, 234, 0.1)"
    ))
    
    if show_hist:
        samples = expon.rvs(scale=scale, size=num_samples)
        fig.add_trace(go.Histogram(
            x=samples, histnorm='probability density', nbinsx=50,
            opacity=0.6, name='Histogram',
            marker_color=colors['accent']
        ))
    
    mean_val = 1/lam
    var_val = 1/(lam**2)
    std_val = 1/lam
    skew_val = 2

elif dist == "Gamma":
    with st.sidebar:
        k = st.slider("Shape (k)", 0.1, 20.0, 2.0, 0.1)
        lam = st.slider("Rate (λ)", 0.1, 5.0, 1.0, 0.1)
    
    scale = 1 / lam
    x = np.linspace(0, gamma.ppf(0.99, k, scale=scale), 1000)
    pdf = gamma.pdf(x, k, scale=scale)
    
    fig.add_trace(go.Scatter(
        x=x, y=pdf, mode='lines', name='PDF',
        line=dict(color=colors['primary'], width=3),
        fill='tonexty', fillcolor=f"rgba(102, 126, 234, 0.1)"
    ))
    
    if show_hist:
        samples = gamma.rvs(k, scale=scale, size=num_samples)
        fig.add_trace(go.Histogram(
            x=samples, histnorm='probability density', nbinsx=50,
            opacity=0.6, name='Histogram',
            marker_color=colors['accent']
        ))
    
    mean_val = k / lam
    var_val = k / (lam**2)
    std_val = np.sqrt(k) / lam
    skew_val = 2 / np.sqrt(k)

elif dist == "Beta":
    with st.sidebar:
        alpha = st.slider("Alpha (α)", 0.1, 10.0, 2.0, 0.1)
        beta_val = st.slider("Beta (β)", 0.1, 10.0, 2.0, 0.1)
    
    x = np.linspace(0, 1, 1000)
    pdf = beta.pdf(x, alpha, beta_val)
    
    fig.add_trace(go.Scatter(
        x=x, y=pdf, mode='lines', name='PDF',
        line=dict(color=colors['primary'], width=3),
        fill='tonexty', fillcolor=f"rgba(102, 126, 234, 0.1)"
    ))
    
    if show_hist:
        samples = beta.rvs(alpha, beta_val, size=num_samples)
        fig.add_trace(go.Histogram(
            x=samples, histnorm='probability density', nbinsx=50,
            opacity=0.6, name='Histogram',
            marker_color=colors['accent']
        ))
    
    mean_val = alpha / (alpha + beta_val)
    var_val = (alpha * beta_val) / ((alpha + beta_val)**2 * (alpha + beta_val + 1))
    std_val = np.sqrt(var_val)
    skew_val = (2 * (beta_val - alpha) * np.sqrt(alpha + beta_val + 1)) / ((alpha + beta_val + 2) * np.sqrt(alpha * beta_val))

elif dist == "Uniform":
    with st.sidebar:
        a = st.slider("Lower Bound (a)", -10.0, 10.0, 0.0, 0.1)
        b = st.slider("Upper Bound (b)", a + 0.1, a + 20.0, a + 1.0, 0.1)
    
    x = np.linspace(a - (b-a)*0.2, b + (b-a)*0.2, 1000)
    pdf = uniform.pdf(x, loc=a, scale=b - a)
    
    fig.add_trace(go.Scatter(
        x=x, y=pdf, mode='lines', name='PDF',
        line=dict(color=colors['primary'], width=3),
        fill='tonexty', fillcolor=f"rgba(102, 126, 234, 0.1)"
    ))
    
    if show_hist:
        samples = uniform.rvs(loc=a, scale=b - a, size=num_samples)
        fig.add_trace(go.Histogram(
            x=samples, histnorm='probability density', nbinsx=50,
            opacity=0.6, name='Histogram',
            marker_color=colors['accent']
        ))
    
    mean_val = (a + b) / 2
    var_val = (b - a)**2 / 12
    std_val = np.sqrt(var_val)
    skew_val = 0

elif dist == "Log-Normal":
    with st.sidebar:
        mu = st.slider("Mean (μ)", -2.0, 3.0, 0.0, 0.1)
        sigma = st.slider("Standard Deviation (σ)", 0.1, 2.0, 0.5, 0.1)
    
    x = np.linspace(0.001, 20, 1000)
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    
    fig.add_trace(go.Scatter(
        x=x, y=pdf, mode='lines', name='PDF',
        line=dict(color=colors['primary'], width=3),
        fill='tonexty', fillcolor=f"rgba(102, 126, 234, 0.1)"
    ))
    
    if show_hist:
        samples = lognorm.rvs(s=sigma, scale=np.exp(mu), size=num_samples)
        fig.add_trace(go.Histogram(
            x=samples, histnorm='probability density', nbinsx=50,
            opacity=0.6, name='Histogram',
            marker_color=colors['accent']
        ))
    
    mean_val = np.exp(mu + sigma**2/2)
    var_val = (np.exp(sigma**2) - 1) * np.exp(2*mu + sigma**2)
    std_val = np.sqrt(var_val)
    skew_val = (np.exp(sigma**2) + 2) * np.sqrt(np.exp(sigma**2) - 1)

# Enhanced plot layout
fig.update_layout(
    title={
        'text': f"{dist} Distribution",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24, 'color': 'black', 'family': 'Inter'}  # Changed to black
    },
    xaxis_title="Value",
    yaxis_title="Probability Density/Mass",
    template="plotly_white",
    showlegend=True,
    legend=dict(
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="rgba(102, 126, 234, 0.2)",
        borderwidth=1,
        font=dict(family="Inter", size=12, color='black')  # Changed to black
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Inter", size=12, color='black'),  # Changed to black (affects axis titles and labels)
    hovermode='x unified',
    xaxis=dict(
        title_font=dict(color='black'),  # Ensures x-axis title is black
        tickfont=dict(color='black')     # Ensures x-axis ticks are black
    ),
    yaxis=dict(
        title_font=dict(color='black'),  # Ensures y-axis title is black
        tickfont=dict(color='black')     # Ensures y-axis ticks are black
    )
)

# Display the plot in a container
st.markdown('<div class="chart-container">', unsafe_allow_html=True)
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
st.markdown('</div>', unsafe_allow_html=True)

# Statistics section
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="stat-container">
        <div class="stat-value">{mean_val:.4f}</div>
        <div class="stat-label">Mean</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-container">
        <div class="stat-value">{var_val:.4f}</div>
        <div class="stat-label">Variance</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-container">
        <div class="stat-value">{std_val:.4f}</div>
        <div class="stat-label">Std Dev</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="stat-container">
        <div class="stat-value">{skew_val:.4f}</div>
        <div class="stat-label">Skewness</div>
    </div>
    """, unsafe_allow_html=True)

# Information section
st.markdown("---")
info = dist_info[dist]
st.markdown(f"""
<div class="info-box">
    <h3>📚 About {dist}</h3>
    <p><strong>Description:</strong> {info['description']}</p>
    <p><strong>Common Use Cases:</strong> {info['use_cases']}</p>
    <p><strong>Parameters:</strong> {info['parameters']}</p>
</div>
""", unsafe_allow_html=True)

# Sample data section (if histogram is shown)
if show_hist:
    st.markdown("### 📈 Sample Statistics")
    
    # Generate samples based on distribution
    if dist == "Gaussian (Normal)":
        samples = norm.rvs(mu, sigma, size=num_samples)
    elif dist == "Binomial":
        samples = binom.rvs(n, p, size=num_samples)
    elif dist == "Poisson":
        samples = poisson.rvs(lam, size=num_samples)
    elif dist == "Exponential":
        samples = expon.rvs(scale=1/lam, size=num_samples)
    elif dist == "Gamma":
        samples = gamma.rvs(k, scale=1/lam, size=num_samples)
    elif dist == "Beta":
        samples = beta.rvs(alpha, beta_val, size=num_samples)
    elif dist == "Uniform":
        samples = uniform.rvs(loc=a, scale=b-a, size=num_samples)
    elif dist == "Log-Normal":
        samples = lognorm.rvs(s=sigma, scale=np.exp(mu), size=num_samples)
    else:
        samples = np.array([])
    
    if len(samples) > 0:
        sample_stats = pd.DataFrame({
            'Statistic': ['Sample Mean', 'Sample Std', 'Sample Min', 'Sample Max', 'Sample Median'],
            'Value': [np.mean(samples), np.std(samples), np.min(samples), np.max(samples), np.median(samples)]
        })

        # Display with white background and black text
        st.dataframe(
            sample_stats.style
                .set_properties(**{
                    'background-color': 'white',
                    'color': 'black',
                    'border-color': '#e0e0e0'
                })
                .set_table_styles([{
                    'selector': 'th',
                    'props': [('background-color', '#f8f9fa'), 
                            ('color', 'black'),
                            ('font-weight', 'bold')]
                }]),
            hide_index=True,
            use_container_width=True)
        
        # Create comparison table
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Theoretical vs Sample Statistics**")
            comparison_df = pd.DataFrame({
                'Metric': ['Mean', 'Standard Deviation'],
                'Theoretical': [f"{mean_val:.4f}", f"{std_val:.4f}"],
                'Sample': [f"{np.mean(samples):.4f}", f"{np.std(samples):.4f}"]
            })
            st.dataframe(comparison_df, hide_index=True)
        
        with col2:
            st.markdown("**Sample Summary**")
            st.dataframe(sample_stats, hide_index=True)

# Advanced Analysis Section
st.markdown("---")
st.markdown("### 🔬 Advanced Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #667eea; margin-bottom: 1rem;">📊 Distribution Properties</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Distribution properties based on type
    if dist in ["Gaussian (Normal)", "Exponential", "Gamma", "Beta", "Uniform", "Log-Normal"]:
        dist_type = "Continuous"
        support_info = {
            "Gaussian (Normal)": "(-∞, +∞)",
            "Exponential": "[0, +∞)",
            "Gamma": "[0, +∞)",
            "Beta": "[0, 1]",
            "Uniform": f"[{a}, {b}]" if dist == "Uniform" else "[a, b]",
            "Log-Normal": "(0, +∞)"
        }
    else:
        dist_type = "Discrete"
        support_info = {
            "Bernoulli": "{0, 1}",
            "Binomial": f"{{0, 1, 2, ..., {n}}}" if dist == "Binomial" else "{0, 1, 2, ..., n}",
            "Poisson": "{0, 1, 2, 3, ...}"
        }
    
    properties_df = pd.DataFrame({
        'Property': ['Type', 'Support', 'Symmetry'],
        'Value': [
            dist_type,
            support_info.get(dist, "Variable"),
            'Symmetric' if dist in ["Gaussian (Normal)", "Uniform"] or (dist == "Beta" and alpha == beta_val) else 'Asymmetric'
        ]
    })
    st.dataframe(properties_df, hide_index=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #667eea; margin-bottom: 1rem;">🎯 Applications</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Real-world applications
    applications = {
        "Gaussian (Normal)": ["Quality Control", "Test Scores", "Physical Measurements", "Financial Modeling"],
        "Bernoulli": ["A/B Testing", "Medical Diagnosis", "Quality Inspection", "Survey Response"],
        "Binomial": ["Clinical Trials", "Marketing Campaigns", "Defect Counting", "Election Polling"],
        "Poisson": ["Website Traffic", "Call Center Calls", "Equipment Failures", "Natural Disasters"],
        "Exponential": ["Service Times", "Component Lifetimes", "Inter-arrival Times", "Radioactive Decay"],
        "Gamma": ["Insurance Claims", "Rainfall Modeling", "Queueing Theory", "Reliability Analysis"],
        "Beta": ["Project Management", "Bayesian Analysis", "Risk Assessment", "Success Rates"],
        "Uniform": ["Random Sampling", "Monte Carlo", "Simulation", "Fair Games"],
        "Log-Normal": ["Stock Prices", "Income Distribution", "Particle Sizes", "Network Traffic"]
    }
    
    apps = applications.get(dist, ["General Statistics"])
    for i, app in enumerate(apps[:4], 1):
        st.markdown(f"**{i}.** {app}")

# Probability Calculator Section
st.markdown("---")
st.markdown("### 🧮 Probability Calculator")

calc_col1, calc_col2 = st.columns(2)

with calc_col1:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #667eea; margin-bottom: 1rem;">Calculate Probabilities</h4>
    </div>
    """, unsafe_allow_html=True)
    
    if dist_type == "Continuous":
        calc_type = st.selectbox("Calculation Type", ["P(X ≤ x)", "P(X ≥ x)", "P(a ≤ X ≤ b)"])
        
        if calc_type == "P(a ≤ X ≤ b)":
            lower_bound = st.number_input("Lower bound (a):", value=0.0)
            upper_bound = st.number_input("Upper bound (b):", value=1.0)
        else:
            x_value = st.number_input("x value:", value=0.0)
        
        if st.button("Calculate Probability", type="primary"):
            try:
                if dist == "Gaussian (Normal)":
                    dist_obj = norm(mu, sigma)
                elif dist == "Exponential":
                    dist_obj = expon(scale=1/lam)
                elif dist == "Gamma":
                    dist_obj = gamma(k, scale=1/lam)
                elif dist == "Beta":
                    dist_obj = beta(alpha, beta_val)
                elif dist == "Uniform":
                    dist_obj = uniform(a, b-a)
                elif dist == "Log-Normal":
                    dist_obj = lognorm(s=sigma, scale=np.exp(mu))
                
                if calc_type == "P(X ≤ x)":
                    prob = dist_obj.cdf(x_value)
                    st.success(f"P(X ≤ {x_value}) = {prob:.6f}")
                elif calc_type == "P(X ≥ x)":
                    prob = 1 - dist_obj.cdf(x_value)
                    st.success(f"P(X ≥ {x_value}) = {prob:.6f}")
                else:
                    prob = dist_obj.cdf(upper_bound) - dist_obj.cdf(lower_bound)
                    st.success(f"P({lower_bound} ≤ X ≤ {upper_bound}) = {prob:.6f}")
            except Exception as e:
                st.error("Error calculating probability. Check your parameters.")
    
    else:  # Discrete distributions
        calc_type = st.selectbox("Calculation Type", ["P(X = k)", "P(X ≤ k)", "P(X ≥ k)"])
        k_value = st.number_input("k value:", value=1, step=1, min_value=0)
        
        if st.button("Calculate Probability", type="primary"):
            try:
                if dist == "Bernoulli":
                    if calc_type == "P(X = k)":
                        prob = bernoulli.pmf(k_value, p)
                    elif calc_type == "P(X ≤ k)":
                        prob = bernoulli.cdf(k_value, p)
                    else:
                        prob = 1 - bernoulli.cdf(k_value-1, p) if k_value > 0 else 1
                elif dist == "Binomial":
                    if calc_type == "P(X = k)":
                        prob = binom.pmf(k_value, n, p)
                    elif calc_type == "P(X ≤ k)":
                        prob = binom.cdf(k_value, n, p)
                    else:
                        prob = 1 - binom.cdf(k_value-1, n, p) if k_value > 0 else 1
                elif dist == "Poisson":
                    if calc_type == "P(X = k)":
                        prob = poisson.pmf(k_value, lam)
                    elif calc_type == "P(X ≤ k)":
                        prob = poisson.cdf(k_value, lam)
                    else:
                        prob = 1 - poisson.cdf(k_value-1, lam) if k_value > 0 else 1
                
                st.success(f"{calc_type.replace('k', str(k_value))} = {prob:.6f}")
            except Exception as e:
                st.error("Error calculating probability. Check your parameters.")

with calc_col2:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #667eea; margin-bottom: 1rem;">Quantile Calculator</h4>
    </div>
    """, unsafe_allow_html=True)
    
    percentile = st.slider("Percentile (%)", 1, 99, 50, 1)
    quantile = percentile / 100
    
    if st.button("Calculate Quantile", type="primary"):
        try:
            if dist == "Gaussian (Normal)":
                result = norm.ppf(quantile, mu, sigma)
            elif dist == "Exponential":
                result = expon.ppf(quantile, scale=1/lam)
            elif dist == "Gamma":
                result = gamma.ppf(quantile, k, scale=1/lam)
            elif dist == "Beta":
                result = beta.ppf(quantile, alpha, beta_val)
            elif dist == "Uniform":
                result = uniform.ppf(quantile, a, b-a)
            elif dist == "Log-Normal":
                result = lognorm.ppf(quantile, s=sigma, scale=np.exp(mu))
            elif dist == "Binomial":
                result = binom.ppf(quantile, n, p)
            elif dist == "Poisson":
                result = poisson.ppf(quantile, lam)
            else:  # Bernoulli
                result = 1 if quantile > (1-p) else 0
            
            st.success(f"The {percentile}th percentile is: {result:.6f}")
            
            # Add interpretation
            if dist_type == "Continuous":
                st.info(f"💡 {percentile}% of values fall below {result:.4f}")
            else:
                st.info(f"💡 {percentile}% of values fall below or equal to {int(result)}")
                
        except Exception as e:
            st.error("Error calculating quantile. Check your parameters.")

# Export and Download Section
st.markdown("---")
st.markdown("### 💾 Export Data")

export_col1, export_col2 = st.columns(2)

with export_col1:
    if st.button("📊 Generate Sample Data", type="secondary"):
        # Generate sample data
        np.random.seed(42)  # For reproducibility
        
        if dist == "Gaussian (Normal)":
            sample_data = norm.rvs(mu, sigma, size=1000)
        elif dist == "Bernoulli":
            sample_data = bernoulli.rvs(p, size=1000)
        elif dist == "Binomial":
            sample_data = binom.rvs(n, p, size=1000)
        elif dist == "Poisson":
            sample_data = poisson.rvs(lam, size=1000)
        elif dist == "Exponential":
            sample_data = expon.rvs(scale=1/lam, size=1000)
        elif dist == "Gamma":
            sample_data = gamma.rvs(k, scale=1/lam, size=1000)
        elif dist == "Beta":
            sample_data = beta.rvs(alpha, beta_val, size=1000)
        elif dist == "Uniform":
            sample_data = uniform.rvs(loc=a, scale=b-a, size=1000)
        else:  # Log-Normal
            sample_data = lognorm.rvs(s=sigma, scale=np.exp(mu), size=1000)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Sample_Index': range(1, 1001),
            'Value': sample_data,
            'Distribution': dist,
            'Parameters': str({k: v for k, v in locals().items() if k in ['mu', 'sigma', 'p', 'n', 'lam', 'k', 'alpha', 'beta_val', 'a', 'b'] and k in locals()})
        })
        
        # Display sample
        st.markdown("**Sample Data Preview (first 10 rows):**")
        st.dataframe(df.head(10))
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{dist.lower().replace(' ', '_')}_sample_data.csv",
            mime="text/csv"
        )

with export_col2:
    if st.button("📈 Export Plot Data", type="secondary"):
        if dist_type == "Continuous":
            if dist == "Gaussian (Normal)":
                x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
                y_vals = norm.pdf(x_vals, mu, sigma)
            elif dist == "Exponential":
                x_vals = np.linspace(0, 10/lam, 1000)
                y_vals = expon.pdf(x_vals, scale=1/lam)
            elif dist == "Gamma":
                x_vals = np.linspace(0, gamma.ppf(0.99, k, scale=1/lam), 1000)
                y_vals = gamma.pdf(x_vals, k, scale=1/lam)
            elif dist == "Beta":
                x_vals = np.linspace(0, 1, 1000)
                y_vals = beta.pdf(x_vals, alpha, beta_val)
            elif dist == "Uniform":
                x_vals = np.linspace(a - (b-a)*0.2, b + (b-a)*0.2, 1000)
                y_vals = uniform.pdf(x_vals, loc=a, scale=b-a)
            else:  # Log-Normal
                x_vals = np.linspace(0.001, 20, 1000)
                y_vals = lognorm.pdf(x_vals, s=sigma, scale=np.exp(mu))
            
            plot_df = pd.DataFrame({
                'x': x_vals,
                'pdf': y_vals,
                'distribution': dist
            })
        else:
            if dist == "Bernoulli":
                x_vals = [0, 1]
                y_vals = bernoulli.pmf(x_vals, p)
            elif dist == "Binomial":
                x_vals = list(range(0, n + 1))
                y_vals = binom.pmf(x_vals, n, p)
            else:  # Poisson
                x_vals = list(range(0, int(lam + 4*np.sqrt(lam)) + 1))
                y_vals = poisson.pmf(x_vals, lam)
            
            plot_df = pd.DataFrame({
                'x': x_vals,
                'pmf': y_vals,
                'distribution': dist
            })
        
        st.markdown("**Plot Data Preview:**")
        st.dataframe(plot_df.head(10))
        
        csv = plot_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Plot Data",
            data=csv,
            file_name=f"{dist.lower().replace(' ', '_')}_plot_data.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255, 255, 255, 0.1); border-radius: 15px; margin-top: 2rem;">
    <h4 style="color: #667eea; margin-bottom: 1rem;">📚 Learn More</h4>
    <p style="color: #666; margin-bottom: 1rem;">
        This interactive tool helps you explore probability distributions and their properties. 
        Experiment with different parameters to see how they affect the shape and characteristics of each distribution.
    </p>
    <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
        <span style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 8px 16px; border-radius: 20px; font-size: 0.9rem;">
            🎯 Interactive Learning
        </span>
        <span style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 8px 16px; border-radius: 20px; font-size: 0.9rem;">
            📊 Statistical Analysis
        </span>
        <span style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 8px 16px; border-radius: 20px; font-size: 0.9rem;">
            💾 Data Export
        </span>
    </div>
</div>
""", unsafe_allow_html=True)