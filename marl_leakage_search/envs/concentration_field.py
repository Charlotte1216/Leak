"""
concentration_field.py
浓度场接口
"""
import numpy as np
from plume_model import GaussianPlumeModel2D
from vortex_model import KarmanVortexStreet


class ConcentrationField:
    """
    Combined concentration field:
        C_total = sum(plume_i) * (1 + vortex)   [optionally only downstream]
    """

    def __init__(
        self,
        sources,
        obstacles=None,
        wind_speed=1.0,
        plume_params=None,
        vortex_params=None,
        keep_plume_behind_obstacle=True,
        noise_std=0.0,
    ):
        """
        sources: list of dicts {"x": , "y": , "Q": }
        obstacles: list of dicts {"x": , "y": , "radius": }
        keep_plume_behind_obstacle:
            if True: plume still exists behind obstacle (realistic if wind can go around)
            if False: plume is blocked behind obstacle (strong blockage)
        noise_std: standard deviation of Gaussian noise added to concentration
        """
        self.sources = sources
        self.obstacles = obstacles if obstacles is not None else []

        self.plume = GaussianPlumeModel2D(
            u=wind_speed,
            L=plume_params.get("L", 100.0),
            sigma_y=plume_params.get("sigma_y", 5.0)
        )

        self.vortex = KarmanVortexStreet(**(vortex_params or {}))

        self.keep_plume_behind_obstacle = keep_plume_behind_obstacle
        self.noise_std = noise_std

    def _is_downstream(self, x, obstacle):
        dx = x - obstacle["x"]
        return dx > obstacle.get("radius", 0.0)

    def concentration(self, x, y, t=0.0):
        """
        Compute the total concentration at grid points (x,y) at time t.
        """
        # 1) base plume from all sources
        C_plume = self.plume.calculate_concentration(x, y, self.sources)

        # 2) apply obstacle blocking (optional)
        if not self.keep_plume_behind_obstacle:
            for obs in self.obstacles:
                downstream = self._is_downstream(x, obs)
                C_plume[downstream] = 0.0

        # 3) add vortex perturbation (only downstream of obstacles)
        C_total = C_plume.copy()
        for obs in self.obstacles:
            downstream = self._is_downstream(x, obs)
            if np.any(downstream):
                f_vortex = self.vortex.perturbation(x, y, t=t, obstacle=obs)
                # only add perturbation downstream
                C_total[downstream] = C_total[downstream] * (1 + f_vortex[downstream])

        # 4) add sensor noise
        if self.noise_std > 0:
            C_total += np.random.normal(0, self.noise_std, size=C_total.shape)

        # ensure non-negative
        C_total = np.clip(C_total, 0, None)

        return C_total

