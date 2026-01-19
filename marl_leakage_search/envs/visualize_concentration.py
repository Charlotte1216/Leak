"""
浓度场可视化
"""
import numpy as np
import matplotlib.pyplot as plt

from plume_model import GaussianPlume2D
from vortex_model import KarmanVortexStreet
from concentration_field import ConcentrationField2D


def visualize_field(t=0.0):
    # -----------------------
    # Grid setup
    # -----------------------
    x_range = (0, 120)
    y_range = (-40, 40)
    resolution = 350

    x = np.linspace(*x_range, resolution)
    y = np.linspace(*y_range, resolution)
    X, Y = np.meshgrid(x, y)

    # -----------------------
    # Plume model
    # -----------------------
    plume = GaussianPlume2D(
        Q=1.0,
        U=2.0,
        sigma_y0=0.6,
        alpha=0.04,
        decay_length=70.0
    )

    # -----------------------
    # Vortex model (convected)
    # -----------------------
    vortex = KarmanVortexStreet(
        A=0.35,
        omega=0.2,
        kx=0.25,
        ky=1.5,
        decay=0.05
    )

    # -----------------------
    # Sources and obstacles
    # -----------------------
    sources = [
        {"x": 5.0, "y": 0.0, "Q": 1.0},
        {"x": 20.0, "y": 12.0, "Q": 0.8}
    ]

    obstacles = [
        {"x": 35.0, "y": 0.0, "radius": 2.5},
        {"x": 65.0, "y": -15.0, "radius": 2.5}
    ]

    field = ConcentrationField2D(
        plume_model=plume,
        vortex_model=vortex,
        sources=sources,
        obstacles=obstacles
    )

    # -----------------------
    # Compute field
    # -----------------------
    C = field.concentration(X, Y, t=t)

    # -----------------------
    # Plot
    # -----------------------
    plt.figure(figsize=(11, 4))
    contour = plt.contourf(X, Y, C, levels=60, cmap="viridis")
    plt.colorbar(contour, label="Concentration")

    # Sources
    for src in sources:
        plt.scatter(
            src["x"], src["y"],
            c="red", marker="x", s=90,
            label="Emission source"
        )

    # Obstacles
    for obs in obstacles:
        circle = plt.Circle(
            (obs["x"], obs["y"]),
            obs["radius"],
            color="white",
            fill=False,
            linewidth=1.5,
            linestyle="--"
        )
        plt.gca().add_patch(circle)

    plt.xlabel("x (downwind direction)")
    plt.ylabel("y (crosswind direction)")
    plt.title("Concentration Field with Convected Kármán Vortex Disturbance")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_field(t=0.0)
