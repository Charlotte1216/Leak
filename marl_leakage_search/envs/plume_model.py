"""
plume_model.py
高斯烟羽模型
"""
import numpy as np

class GaussianPlumeModel2D:
    """
    2D Gaussian plume model (simplified).
    Concentration from multiple sources:
        C_plume(x,y) = sum_i Q_i/(2*pi*sigma_y*u) * exp(-(y-y_i)^2/(2*sigma_y^2)) * exp(-(x-x_i)/L)
    """

    def __init__(self, u=1.0, L=100.0, sigma_y=5.0, wind_dir=0.0):
        """
        Parameters:
        - u: wind speed (assumed constant along x direction)
        - L: downstream attenuation length scale
        - sigma_y: lateral dispersion coefficient (can be constant or a function)
        - wind_dir: wind direction angle in radians (0 means +x direction)
        """
        self.u = u
        self.L = L
        self.sigma_y = sigma_y
        self.wind_dir = float(wind_dir)
        self._cos = np.cos(self.wind_dir)
        self._sin = np.sin(self.wind_dir)

    def sigma_y_func(self, dx):
        """
        Dispersion parameter.
        You can extend to be distance-dependent later.
        """
        return self.sigma_y

    def concentration_from_source(self, x, y, source):
        """
        Calculate concentration field from a single source.

        source: dict with keys {"x": , "y": , "Q": }
        """
        dx = x - source["x"]
        dy = y - source["y"]

        # Rotate into wind-aligned coordinates
        xw = dx * self._cos + dy * self._sin
        yw = -dx * self._sin + dy * self._cos

        # Only consider downstream
        downstream = xw > 0
        C = np.zeros_like(x, dtype=float)

        if np.any(downstream):
            sigma_y = self.sigma_y_func(xw[downstream])
            C[downstream] = (
                source["Q"]
                / (2.0 * np.pi * sigma_y * self.u)
                * np.exp(-0.5 * (yw[downstream] / sigma_y) ** 2)
                * np.exp(-xw[downstream] / self.L)
            )
        return C

    def calculate_concentration(self, x, y, sources):
        """
        Calculate total concentration from all sources.
        """
        C_total = np.zeros_like(x, dtype=float)
        for s in sources:
            C_total += self.concentration_from_source(x, y, s)
        return C_total
