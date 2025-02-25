import streamlit as st
import numpy as np
import plotly.graph_objects as go
from optimizers import SGD, Adam
from loss_functions import (
    quadratic_loss,
    rastrigin_loss,
    quadratic_loss_gradient,
    rastrigin_loss_gradient,
)

st.title("Visualização da Otimização em Machine Learning")

optimizer_name = st.selectbox("Escolha o otimizador:", ["SGD", "Adam"])
learning_rate = st.slider("Taxa de aprendizado", 0.001, 1.0, 0.1)
num_iterations = st.slider("Número de iterações", 10, 100, 50)

loss_function_name = st.selectbox(
    "Escolha a função de perda:", ["Quadrática", "Rastrigin"]
)

if loss_function_name == "Quadrática":
    loss_function = quadratic_loss
    loss_gradient = quadratic_loss_gradient
else:
    loss_function = rastrigin_loss
    loss_gradient = rastrigin_loss_gradient

np.random.seed(42)
weights = np.random.randn(2)

optimizer = SGD(lr=learning_rate) if optimizer_name == "SGD" else Adam(lr=learning_rate)
history = []

for _ in range(num_iterations):
    gradients = loss_gradient(weights)
    weights = optimizer.update(weights, gradients)
    history.append(weights.copy())

fig = go.Figure(
    layout=go.Layout(
        title="Otimização dos Pesos",
        xaxis=dict(range=[-2, 2], title="Peso 1"),
        yaxis=dict(range=[-2, 2], title="Peso 2"),
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
