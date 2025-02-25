import matplotlib.pyplot as plt
import numpy as np


def animate_optimization(history):
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    for i in range(len(history)):
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.scatter(*zip(*history[: i + 1]), c="red", label="Pesos")
        ax.legend()
        plt.pause(0.01)

    return fig
