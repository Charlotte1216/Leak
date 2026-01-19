"""
plume_model.py
高斯烟羽模型
"""
import numpy as np


class GaussianPlume2D:
    """
    2D Gaussian Plume Model (steady-state, constant wind)

    Assumptions:
    - Wind direction along +x
    - Ground-level release
    - Time-averaged concentration field
    """

    def __init__(
        self,
        Q=1.0,               # emission rate
        U=1.0,               # wind speed
        sigma_y0=0.1,        # base lateral dispersion
        alpha=0.1,           # dispersion growth rate
        decay_length=50.0    # effective decay length
    ):
        self.Q = Q
        self.U = U
        self.sigma_y0 = sigma_y0
        self.alpha = alpha
        self.decay_length = decay_length

    def sigma_y(self, x):
        """
        Lateral dispersion coefficient
        """
        return self.sigma_y0 + self.alpha * np.maximum(x, 0.0)

    def concentration(self, x, y, source):
        """
        Compute plume concentration from a single source.

        Parameters
        ----------
        x, y : ndarray or float
            Spatial coordinates
        source : dict
            {'x': float, 'y': float, 'Q': optional}

        Returns
        -------
        C : ndarray or float
            Concentration value
        """
        xs, ys = source["x"], source["y"]
        Q = source.get("Q", self.Q)

        # Shift coordinates
        dx = x - xs
        dy = y - ys

        # Upwind region has zero concentration
        mask = dx > 0
        C = np.zeros_like(dx, dtype=float)

        if not np.any(mask):
            return C

        sigma_y = self.sigma_y(dx[mask])

        C[mask] = (
            Q
            / (np.sqrt(2 * np.pi) * self.U * sigma_y)
            * np.exp(-0.5 * (dy[mask] / sigma_y) ** 2)
            * np.exp(-dx[mask] / self.decay_length)
        )

        return C

    def multi_source_concentration(self, x, y, sources):
        """
        Superposition of multiple emission sources.
        """
        C_total = np.zeros_like(x, dtype=float)
        for src in sources:
            C_total += self.concentration(x, y, src)
        return C_total


