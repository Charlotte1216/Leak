"""
concentration_field.py
Concentration field interface.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from .plume_model import GaussianPlumeModel2D
    from .vortex_model import KarmanVortexStreet
except ImportError:
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
        wind_dir=0.0,
        plume_params=None,
        vortex_params=None,
        keep_plume_behind_obstacle=True,
        noise_std=0.0,
        wind_time_params: Dict | None = None,
    ):
        """
        sources: list of dicts {"x": , "y": , "Q": }
        obstacles: list of dicts {"x": , "y": , "radius": }
        keep_plume_behind_obstacle:
            if True: plume still exists behind obstacle (realistic if wind can go around)
            if False: plume is blocked behind obstacle (strong blockage)
        noise_std: standard deviation of Gaussian noise added to concentration
        wind_time_params:
            Optional time-varying wind configuration:
            {
                "enabled": bool,
                "speed_amplitude": float,
                "speed_frequency": float,
                "speed_phase": float,
                "dir_amplitude": float,
                "dir_frequency": float,
                "dir_phase": float,
            }
        """
        plume_params = plume_params or {}
        vortex_params = vortex_params or {}

        self.sources = sources
        self.obstacles = obstacles if obstacles is not None else []
        self.base_wind_speed = max(float(wind_speed), 1e-6)
        self.base_wind_dir = float(wind_dir)
        self.wind_time_params = dict(wind_time_params or {})

        self.wind_speed = self.base_wind_speed
        self.wind_dir = self.base_wind_dir
        self._cos = np.cos(self.wind_dir)
        self._sin = np.sin(self.wind_dir)

        self.plume = GaussianPlumeModel2D(
            u=self.wind_speed,
            L=plume_params.get("L", 100.0),
            sigma_y=plume_params.get("sigma_y", 5.0),
            wind_dir=self.wind_dir,
        )

        self.vortex = KarmanVortexStreet(**vortex_params)
        self.keep_plume_behind_obstacle = bool(keep_plume_behind_obstacle)
        self.noise_std = float(noise_std)

    def set_wind(self, wind_speed: float, wind_dir: float) -> None:
        self.base_wind_speed = max(float(wind_speed), 1e-6)
        self.base_wind_dir = float(wind_dir)
        self._set_current_wind(self.base_wind_speed, self.base_wind_dir)

    def _set_current_wind(self, wind_speed: float, wind_dir: float) -> None:
        self.wind_speed = max(float(wind_speed), 1e-6)
        self.wind_dir = float(wind_dir)
        self._cos = np.cos(self.wind_dir)
        self._sin = np.sin(self.wind_dir)
        self.plume.set_wind(self.wind_speed, self.wind_dir)

    def _wind_at_time(self, t: float) -> tuple[float, float]:
        cfg = self.wind_time_params
        if not bool(cfg.get("enabled", False)):
            return self.base_wind_speed, self.base_wind_dir

        tt = float(t)
        speed_amplitude = float(cfg.get("speed_amplitude", 0.0))
        speed_frequency = float(cfg.get("speed_frequency", 0.0))
        speed_phase = float(cfg.get("speed_phase", 0.0))
        dir_amplitude = float(cfg.get("dir_amplitude", 0.0))
        dir_frequency = float(cfg.get("dir_frequency", 0.0))
        dir_phase = float(cfg.get("dir_phase", 0.0))

        wind_speed = self.base_wind_speed + speed_amplitude * np.sin(
            2.0 * np.pi * speed_frequency * tt + speed_phase
        )
        wind_dir = self.base_wind_dir + dir_amplitude * np.sin(
            2.0 * np.pi * dir_frequency * tt + dir_phase
        )
        return max(float(wind_speed), 1e-6), float(wind_dir)

    def _is_downstream(self, x, y, obstacle):
        dx = x - obstacle["x"]
        dy = y - obstacle["y"]
        downstream_dist = dx * self._cos + dy * self._sin
        return downstream_dist > obstacle.get("radius", 0.0)

    def concentration(self, x, y, t=0.0):
        """
        Compute the total concentration at grid points (x,y) at time t.
        """
        wind_speed_t, wind_dir_t = self._wind_at_time(t)
        self._set_current_wind(wind_speed_t, wind_dir_t)

        # 1) base plume from all sources
        C_plume = self.plume.calculate_concentration(x, y, self.sources)

        # 2) apply obstacle blocking (optional)
        if not self.keep_plume_behind_obstacle:
            for obs in self.obstacles:
                downstream = self._is_downstream(x, y, obs)
                C_plume[downstream] = 0.0

        # 3) add vortex perturbation (only downstream of obstacles)
        C_total = C_plume.copy()
        for obs in self.obstacles:
            downstream = self._is_downstream(x, y, obs)
            if np.any(downstream):
                f_vortex = self.vortex.perturbation(
                    x,
                    y,
                    t=t,
                    obstacle=obs,
                    wind_dir=self.wind_dir,
                    wind_speed=self.wind_speed,
                )
                # only add perturbation downstream
                C_total[downstream] = C_total[downstream] * (1 + f_vortex[downstream])

        # 4) add sensor noise
        if self.noise_std > 0:
            C_total += np.random.normal(0, self.noise_std, size=C_total.shape)

        # ensure non-negative
        C_total = np.clip(C_total, 0, None)

        return C_total

