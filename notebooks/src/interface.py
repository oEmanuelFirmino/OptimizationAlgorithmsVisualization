import streamlit as st
import numpy as np
import plotly.graph_objects as go
from optimizers import SGD, Adam, RMSprop, Adagrad
from loss_functions import (
    quadratic_loss,
    rastrigin_loss,
    rosenbrock_loss,
    cross_entropy_loss,
    hinge_loss,
    quadratic_loss_gradient,
    rastrigin_loss_gradient,
    rosenbrock_loss_gradient,
    cross_entropy_loss_gradient,
    hinge_loss_gradient,
)

st.title("Optimization Visualization in Machine Learning")

optimizer_name = st.selectbox(
    "Choose the optimizer:", ["SGD", "Adam", "RMSprop", "Adagrad"]
)
learning_rate = st.slider("Learning rate", 0.001, 1.0, 0.1)
num_iterations = st.slider("Number of iterations", 10, 100, 50)

loss_function_name = st.selectbox(
    "Choose the loss function:",
    ["Quadratic", "Rastrigin", "Rosenbrock", "Cross-Entropy", "Hinge"],
)

if loss_function_name == "Quadratic":
    loss_function = quadratic_loss
    loss_gradient = quadratic_loss_gradient
elif loss_function_name == "Rastrigin":
    loss_function = rastrigin_loss
    loss_gradient = rastrigin_loss_gradient
elif loss_function_name == "Rosenbrock":
    loss_function = rosenbrock_loss
    loss_gradient = rosenbrock_loss_gradient
elif loss_function_name == "Cross-Entropy":
    loss_function = cross_entropy_loss
    loss_gradient = cross_entropy_loss_gradient
elif loss_function_name == "Hinge":
    loss_function = hinge_loss
    loss_gradient = hinge_loss_gradient

np.random.seed(42)
weights = np.random.randn(2)

if optimizer_name == "SGD":
    optimizer = SGD(lr=learning_rate)
elif optimizer_name == "Adam":
    optimizer = Adam(lr=learning_rate)
elif optimizer_name == "RMSprop":
    optimizer = RMSprop(lr=learning_rate)
elif optimizer_name == "Adagrad":
    optimizer = Adagrad(lr=learning_rate)
history = []

for _ in range(num_iterations):
    gradients = loss_gradient(weights)
    weights = optimizer.update(weights, gradients)
    history.append(weights.copy())

fig = go.Figure(
    layout=go.Layout(
        title="Weight Optimization",
        xaxis=dict(range=[-2, 2], title="Weight 1"),
        yaxis=dict(range=[-2, 2], title="Weight 2"),
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=200, redraw=True), fromcurrent=True
                            ),
                        ],
                    )
                ],
            )
        ],
    )
)

frames = [
    go.Frame(
        data=[
            go.Scatter(
                x=[p[0] for p in history[:k]],
                y=[p[1] for p in history[:k]],
                mode="markers",
                marker=dict(color="red", size=10),
            )
        ],
        name=f"Frame {k}",
    )
    for k in range(1, len(history) + 1)
]

fig.frames = frames
fig.add_trace(
    go.Scatter(
        x=[history[0][0]],
        y=[history[0][1]],
        mode="markers",
        marker=dict(color="red", size=10),
    )
)

st.plotly_chart(fig, use_container_width=True)
