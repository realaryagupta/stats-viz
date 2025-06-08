import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Define function and gradient
def func(x):
    return x ** 2

def grad(x):
    return 2 * x

# Gradient descent steps
def gradient_descent(start, lr, steps):
    x_vals = [start]
    y_vals = [func(start)]
    x = start
    for _ in range(steps):
        x = x - lr * grad(x)
        x_vals.append(x)
        y_vals.append(func(x))
    return x_vals, y_vals

# Streamlit UI
st.title("🏀 Gradient Descent with Rolling Ball")

# Sidebar controls
st.sidebar.header("Parameters")
start = st.sidebar.slider("Initial x", -10.0, 10.0, -6.0)
lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1)
steps = st.sidebar.slider("Steps", 1, 100, 30)

# Compute trajectory
x_path, y_path = gradient_descent(start, lr, steps)

# Static curve
x_curve = np.linspace(-10, 10, 500)
y_curve = func(x_curve)

# Initial plot elements
curve = go.Scatter(x=x_curve, y=y_curve, mode="lines", name="f(x) = x²", line=dict(color="blue"))
ball = go.Scatter(x=[x_path[0]], y=[y_path[0]], mode="markers", name="Ball",
                  marker=dict(size=14, color="red"))

# Animation frames
frames = []
for i in range(1, len(x_path)):
    frames.append(go.Frame(
        data=[
            go.Scatter(x=x_curve, y=y_curve, mode="lines", line=dict(color="blue")),
            go.Scatter(x=[x_path[i]], y=[y_path[i]], mode="markers", marker=dict(color="red", size=14))
        ],
        name=str(i)
    ))

# Layout with play button
layout = go.Layout(
    title="Gradient Descent Animation (Ball Rolling on Curve)",
    xaxis=dict(title="x", range=[-10, 10]),
    yaxis=dict(title="f(x)", range=[0, max(y_path) + 10]),
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        buttons=[
            dict(label="▶ Play",
                 method="animate",
                 args=[None, {"frame": {"duration": 300, "redraw": True},
                              "fromcurrent": True, "transition": {"duration": 0}}])
        ]
    )]
)

# Build figure
fig = go.Figure(data=[curve, ball], layout=layout, frames=frames)

# Show in Streamlit
st.plotly_chart(fig)
