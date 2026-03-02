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
            phase=0.0,
            use_strouhal=False,
            strouhal=0.2,
            min_wind_speed=1e-3,
            min_diameter=1e-3,
    ):
        self.A = A
        self.omega = omega
        self.kx = kx
        self.ky = ky
        self.decay = decay
        self.sigma_w0 = sigma_w0
        self.beta = beta
        self.phase = phase
        self.use_strouhal = bool(use_strouhal)
        self.strouhal = float(strouhal)
        self.min_wind_speed = float(min_wind_speed)
        self.min_diameter = float(min_diameter)

    def _wave_params(self, obstacle_radius, wind_speed):
        if not self.use_strouhal:
            return float(self.omega), float(self.kx)
        diameter = max(2.0 * max(float(obstacle_radius), 0.0), self.min_diameter)
        u = max(abs(float(wind_speed)), self.min_wind_speed)
        omega = 2.0 * np.pi * self.strouhal * u / diameter
        kx = omega / u
        return omega, kx

    def perturbation(self, x, y, t=0.0, obstacle=None, wind_dir=0.0, wind_speed=1.0):
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
            omega, kx = self._wave_params(r0, wind_speed)

            # downstream attenuation
            attenuation = np.exp(-self.decay * xw[downstream])

            # wake width grows with downstream distance
            sigma_w = self.sigma_w0 + self.beta * xw[downstream]

            # crosswind envelope
            envelope = np.exp(-0.5 * (yw[downstream] / sigma_w) ** 2)

            # oscillatory phase
            phase = (
                    omega * t
                    - kx * xw[downstream]
                    + self.ky * yw[downstream]
                    + self.phase
            )

            f[downstream] = self.A * attenuation * envelope * np.sin(phase)

        return f

