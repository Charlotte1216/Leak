"""
vortex_model.py
卡门涡街模型
"""
import numpy as np


class KarmanVortexStreet:
    """
    Convected Karman vortex street perturbation model.
    Output is a dimensionless perturbation term f_vortex(x,y,t) in [-1, 1].
    """

    def __init__(
            self,
            A=0.5,  # perturbation amplitude (dimensionless)
            omega=0.2,  # shedding frequency
            kx=0.3,  # streamwise wave number
            ky=2.0,  # cross-stream wave number
            decay=0.05,  # downstream attenuation
            sigma_w0=2.0,  # initial wake half-width
            beta=0.05,  # wake spreading rate
            phase=0.0
    ):
        self.A = A
        self.omega = omega
        self.kx = kx
        self.ky = ky
        self.decay = decay
        self.sigma_w0 = sigma_w0
        self.beta = beta
        self.phase = phase

    def perturbation(self, x, y, t=0.0, obstacle=None, wind_dir=0.0):
        """
        Calculate the vortex perturbation term for points (x,y) at time t.

        obstacle: dict with keys {"x": , "y": , "radius": }
        """
        if obstacle is None:
            xo, yo, r0 = 0.0, 0.0, 0.0
        else:
            xo = obstacle.get("x", 0.0)
            yo = obstacle.get("y", 0.0)
            r0 = obstacle.get("radius", 0.0)

        dx = x - xo
        dy = y - yo

        wind_dir = float(wind_dir)
        cos_w = np.cos(wind_dir)
        sin_w = np.sin(wind_dir)
        xw = dx * cos_w + dy * sin_w
        yw = -dx * sin_w + dy * cos_w

        downstream = (xw > r0) & (np.abs(yw) <= r0)
        f = np.zeros_like(xw, dtype=float)

        if np.any(downstream):
            # downstream attenuation
            attenuation = np.exp(-self.decay * xw[downstream])

            # wake width grows with downstream distance
            sigma_w = self.sigma_w0 + self.beta * xw[downstream]

            # crosswind envelope
            envelope = np.exp(-0.5 * (yw[downstream] / sigma_w) ** 2)

            # oscillatory phase
            phase = (
                    self.omega * t
                    - self.kx * xw[downstream]
                    + self.ky * yw[downstream]
                    + self.phase
            )

            f[downstream] = self.A * attenuation * envelope * np.sin(phase)

        return f

