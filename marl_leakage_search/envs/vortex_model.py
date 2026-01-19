"""
vortex_model.py
卡门涡街模型
"""
import numpy as np


class KarmanVortexStreet:
    """
    Convected Kármán vortex street model.
    """

    def __init__(
        self,
        A=0.3,        # disturbance amplitude
        omega=0.2,    # shedding frequency
        kx=0.3,       # streamwise wave number
        ky=2.0,       # cross-stream wave number
        decay=0.05,   # downstream attenuation
        phase=0.0
    ):
        self.A = A
        self.omega = omega
        self.kx = kx
        self.ky = ky
        self.decay = decay
        self.phase = phase

    def perturbation(self, x, y, t=0.0, obstacle=None):
        if obstacle is None:
            xo, yo = 0.0, 0.0
            r0 = 0.0
        else:
            xo = obstacle["x"]
            yo = obstacle["y"]
            r0 = obstacle.get("radius", 0.0)

        dx = x - xo
        dy = y - yo

        downstream = dx > r0
        delta_C = np.zeros_like(dx, dtype=float)

        if not np.any(downstream):
            return delta_C

        attenuation = np.exp(-self.decay * dx[downstream])

        phase = (
            self.omega * t
            - self.kx * dx[downstream]
            + self.ky * dy[downstream]
            + self.phase
        )

        delta_C[downstream] = self.A * attenuation * np.sin(phase)

        return delta_C

