import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import scipy.stats as stats

# Page configuration
st.set_page_config(page_title="Regression Assumptions Visualizer", layout="wide")

# Title
st.title("📊 Linear Regression Assumptions Visualizer")
st.markdown("This app demonstrates the five key assumptions of linear regression using **synthetic data**.")

# Generate synthetic data
np.random.seed(42)
n = 100
X1 = np.random.normal(0, 1, n)
X2 = 0.8 * X1 + np.random.normal(0, 0.2, n)  # correlated with X1 for multicollinearity
X3 = np.random.normal(0, 1, n)
noise = np.random.normal(0, 1, n)
y = 3 + 2 * X1 - 1.5 * X2 + 0.5 * X3 + noise

# Create DataFrame
df = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'y': y
})

# Fit regression model
X = df[['X1', 'X2', 'X3']]
y = df['y']
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
residuals = y - y_pred
fitted = y_pred

# Sidebar info
st.sidebar.title("🧪 Model Summary")
st.sidebar.write("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    st.sidebar.write(f"{feature}: {coef:.2f}")
st.sidebar.write(f"Intercept: {model.intercept_:.2f}")

# === 1. Linearity ===
st.subheader("1️⃣ Linearity: Scatter Plots with Regression Line")
cols = st.columns(len(X.columns))
for i, col in enumerate(X.columns):
    fig = px.scatter(df, x=col, y='y', opacity=0.6, title=f"{col} vs y")
    lr = LinearRegression().fit(df[[col]], df['y'])
    fig.add_traces(go.Scatter(x=df[col], y=lr.predict(df[[col]]), mode='lines', name='Linear Fit'))
    cols[i].plotly_chart(fig, use_container_width=True)

# === 2. Multicollinearity ===
st.subheader("2️⃣ Multicollinearity: Correlation Matrix & VIF")
corr_fig = px.imshow(df[['X1', 'X2', 'X3']].corr(), text_auto=True, title="Correlation Heatmap")
st.plotly_chart(corr_fig, use_container_width=True)

# VIF calculation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
vif_bar = px.bar(vif_data, x='Feature', y='VIF', title='Variance Inflation Factor (VIF)')
st.plotly_chart(vif_bar, use_container_width=True)

# === 3. Normality of Residuals ===
st.subheader("3️⃣ Normality of Residuals")
resid_df = pd.DataFrame({'Residuals': residuals})
hist_fig = px.histogram(resid_df, x="Residuals", nbins=30, marginal="rug", title="Histogram of Residuals")
qq_fig = go.Figure()
sm_resid = sm.OLS(residuals, np.ones(len(residuals))).fit()
qq = sm.ProbPlot(sm_resid.resid, fit=True)
theoretical, sample = qq.theoretical_quantiles, qq.sample_quantiles
qq_fig.add_trace(go.Scatter(x=theoretical, y=sample, mode='markers', name='Q-Q Plot'))
qq_fig.add_trace(go.Scatter(x=theoretical, y=theoretical, mode='lines', name='45-degree line'))
qq_fig.update_layout(title="Q-Q Plot of Residuals", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")

col1, col2 = st.columns(2)
col1.plotly_chart(hist_fig, use_container_width=True)
col2.plotly_chart(qq_fig, use_container_width=True)

# === 4. Homoscedasticity ===
st.subheader("4️⃣ Homoscedasticity: Residuals vs. Fitted")
homo_fig = px.scatter(x=fitted, y=residuals, labels={"x": "Fitted values", "y": "Residuals"}, title="Residuals vs. Fitted Values")
homo_fig.add_hline(y=0, line_dash="dash")
st.plotly_chart(homo_fig, use_container_width=True)

# === 5. Autocorrelation ===
st.subheader("5️⃣ Autocorrelation: Residual Lag Plot")
resid_lag = pd.DataFrame({'residuals': residuals, 'lag1': pd.Series(residuals).shift(1)})
auto_fig = px.scatter(resid_lag, x='lag1', y='residuals', title="Lag Plot of Residuals")
auto_fig.update_layout(xaxis_title="Lag 1 Residual", yaxis_title="Residual")
st.plotly_chart(auto_fig, use_container_width=True)
