"""
marl_env.py
"""

import os
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from .uav_dynamics import UAVDynamics
except ImportError:
    from uav_dynamics import UAVDynamics


class PlumeEnv:
    """
    Gym-style multi-agent environment for gas leak source localization.
    Each reset loads a random concentration field from .npz files.
    """
    def __init__(
        self,
        field_dir: str,
        num_agents: int = 2,
        source_find_radius: float = 5.0, # 发现源的半径
        collision_penalty: float = 0.2, # 碰撞惩罚
        battery_penalty: float = 0.00001, # 电池惩罚
        found_source_bonus: float = 20.0, # 发现源奖励
        done_bonus: float = 0.0, # 完成任务奖励
        enable_collision: bool = True, # 是否启用与障碍物的碰撞检测/惩罚
        stop_on_collision: bool = True, # 碰撞时是否停止无人机
        observe_wind: bool = False, # 观测中是否包含风速/风向
        init_pos_mode: str = "random",
        seed: int = None,
        uav_params: Dict = None,
    ):
        self.field_dir = Path(field_dir)
        self.num_agents = num_agents
        self.source_find_radius = float(source_find_radius)
        self.collision_penalty = float(collision_penalty)
        self.battery_penalty = float(battery_penalty)
        self.found_source_bonus = float(found_source_bonus)
        self.done_bonus = float(done_bonus)
        self.enable_collision = bool(enable_collision)
        self.stop_on_collision = bool(stop_on_collision)
        self.observe_wind = bool(observe_wind)
        self.init_pos_mode = init_pos_mode
        self.rng = random.Random(seed)

        self.uav_params = uav_params or {}
        self.uavs: List[UAVDynamics] = []

        self.concentration = None
        self.sources = None
        self.obstacles = None
        self.wind_speed = None
        self.wind_dir = None
        self.x_vals = None
        self.y_vals = None
        self.found_sources = None

    def reset(self):
        """
        Load a random concentration field and reset UAVs.
        Returns initial observations for all agents.
        """
        data = self.random_select_field()

        # load field data
        if "concentration_field" in data:
            self.concentration = data["concentration_field"]
        elif "concentration" in data:
            self.concentration = data["concentration"]
        else:
            raise KeyError("Missing concentration field in .npz file")

        self.sources = data.get("sources", np.zeros((0, 3), dtype=float))
        self.obstacles = data.get("obstacles", np.zeros((0, 3), dtype=float))
        self.wind_speed = float(data.get("wind_speed", [1.0])[0])
        if "wind_dir" in data:
            self.wind_dir = float(data.get("wind_dir", [0.0])[0])
        elif "wind_direction" in data:
            self.wind_dir = float(data.get("wind_direction", [0.0])[0])
        else:
            self.wind_dir = 0.0
        self.x_vals = data.get("x")
        self.y_vals = data.get("y")

        # initialize UAVs
        self.uavs = []
        for _ in range(self.num_agents):
            init_pos = self._sample_init_pos()
            params = dict(self.uav_params)
            params["init_pos"] = init_pos
            self.uavs.append(UAVDynamics(**params))

        self.found_sources = np.zeros(len(self.sources), dtype=bool)
        return self._get_observations()

    def step(self, action):
        """
        Execute actions for all UAVs.
        Returns next_state, rewards, done, info.
        """
        if not isinstance(action, (list, tuple)) or len(action) != self.num_agents:
            raise ValueError("Action must be a list/tuple with length num_agents")

        rewards = []
        prev_batteries = [uav.battery for uav in self.uavs]
        collision_terms = [0.0] * self.num_agents
        conc_terms = [0.0] * self.num_agents
        found_terms = [0.0] * self.num_agents
        battery_terms = [0.0] * self.num_agents
        done_terms = [0.0] * self.num_agents

        for idx, (uav, act) in enumerate(zip(self.uavs, action)):
            if not uav.is_battery_empty():
                uav.move(int(act))

            # keep within bounds
            uav.x, uav.y = self._clip_position(uav.x, uav.y)

            # collision check
            collided = self.enable_collision and self._check_collision(uav.x, uav.y)
            if collided:
                collision_terms[idx] = -self.collision_penalty
                rewards.append(collision_terms[idx])
                if self.stop_on_collision:
                    # stop UAV on collision
                    uav.vx = 0.0
                    uav.vy = 0.0
            else:
                rewards.append(0.0)

        # compute concentration-based rewards and found sources
        for i, uav in enumerate(self.uavs):
            conc = self._sample_concentration(uav.x, uav.y)
            conc_terms[i] = conc
            rewards[i] += conc_terms[i]

            newly_found = self._update_found_sources(uav.x, uav.y)
            if newly_found > 0:
                found_terms[i] = self.found_source_bonus * newly_found
                rewards[i] += found_terms[i]

            # battery penalty based on consumption
            battery_terms[i] = -(prev_batteries[i] - uav.battery) * self.battery_penalty
            rewards[i] += battery_terms[i]

        done = self.is_done()
        if done:
            done_terms = [self.done_bonus for _ in range(self.num_agents)]
            rewards = [r + self.done_bonus for r in rewards]

        next_state = self._get_observations()
        info = {
            "found_sources": int(self.found_sources.sum()),
            "total_sources": int(len(self.sources)),
            "reward_components": {
                "collision": collision_terms,
                "concentration": conc_terms,
                "found_bonus": found_terms,
                "battery": battery_terms,
                "done_bonus": done_terms,
            },
        }
        return next_state, rewards, done, info

    def observe(self):
        """
        Return concentration values at each UAV position.
        """
        return [self._sample_concentration(uav.x, uav.y) for uav in self.uavs]

    def render(self, save_path: str = None, show: bool = False):
        """
        Render the current concentration field with UAVs, sources, and obstacles.
        If save_path is provided, saves the figure to disk.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, ax = plt.subplots(figsize=(6, 5))
        if self.x_vals is not None and self.y_vals is not None:
            extent = [self.x_vals.min(), self.x_vals.max(), self.y_vals.min(), self.y_vals.max()]
        else:
            extent = None

        ax.imshow(self.concentration, origin="lower", extent=extent, cmap="plasma", aspect="auto")

        # Sources
        if self.sources is not None and len(self.sources) > 0:
            ax.scatter(self.sources[:, 0], self.sources[:, 1], c="red", s=40, label="Sources", edgecolor="black")

        # Obstacles
        if self.obstacles is not None and len(self.obstacles) > 0:
            for ox, oy, r in self.obstacles:
                circle = patches.Circle((ox, oy), r, color="blue", alpha=0.3)
                ax.add_patch(circle)

        # UAVs
        for i, uav in enumerate(self.uavs):
            ax.scatter(uav.x, uav.y, c="white", s=40, edgecolor="black")
            ax.text(uav.x, uav.y, f"{i}", color="black", fontsize=8, ha="center", va="center")

        ax.set_title("Concentration Field")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper right", frameon=True)

        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        return fig

    def is_done(self):
        """
        Done when all sources found or all UAVs run out of battery.
        """
        if self.found_sources is not None and self.found_sources.all():
            return True
        if all(uav.is_battery_empty() for uav in self.uavs):
            return True
        return False

    def random_select_field(self) -> Dict[str, np.ndarray]:
        """
        Randomly select a .npz field file and load its data.
        """
        if not self.field_dir.exists():
            raise FileNotFoundError(f"Field directory not found: {self.field_dir}")

        files = [p for p in self.field_dir.iterdir() if p.suffix == ".npz"]
        if not files:
            raise FileNotFoundError(f"No .npz files found in {self.field_dir}")

        file_path = self.rng.choice(files)
        print(f"Randomly selected file: {file_path}")  # Debugging line to ensure randomness
        return dict(np.load(file_path, allow_pickle=True))

    def _get_observations(self) -> List[np.ndarray]:
        obs = []
        for uav in self.uavs:
            conc = self._sample_concentration(uav.x, uav.y)
            payload = [uav.x, uav.y, conc, uav.battery]
            if self.observe_wind:
                payload.extend([self.wind_speed, math.cos(self.wind_dir), math.sin(self.wind_dir)])
            obs.append(np.array(payload, dtype=float))
        return obs

    def _sample_init_pos(self) -> Tuple[float, float]:
        if self.init_pos_mode == "random":
            x_min, x_max, y_min, y_max = self._get_bounds()
            return (
                self.rng.uniform(x_min, x_max),
                self.rng.uniform(y_min, y_max),
            )
        return (0.0, 0.0)

    def _get_bounds(self) -> Tuple[float, float, float, float]:
        if self.x_vals is not None and self.y_vals is not None:
            return float(self.x_vals.min()), float(self.x_vals.max()), float(self.y_vals.min()), float(self.y_vals.max())
        h, w = self.concentration.shape
        return 0.0, float(w - 1), 0.0, float(h - 1)

    def _clip_position(self, x: float, y: float) -> Tuple[float, float]:
        x_min, x_max, y_min, y_max = self._get_bounds()
        return max(x_min, min(x_max, x)), max(y_min, min(y_max, y))

    def _sample_concentration(self, x: float, y: float) -> float:
        if self.concentration is None:
            return 0.0

        if self.x_vals is None or self.y_vals is None:
            ix = int(round(x))
            iy = int(round(y))
            ix = max(0, min(self.concentration.shape[1] - 1, ix))
            iy = max(0, min(self.concentration.shape[0] - 1, iy))
            return float(self.concentration[iy, ix])

        ix = int(np.argmin(np.abs(self.x_vals - x)))
        iy = int(np.argmin(np.abs(self.y_vals - y)))
        return float(self.concentration[iy, ix])

    def _check_collision(self, x: float, y: float) -> bool:
        if self.obstacles is None or len(self.obstacles) == 0:
            return False
        for ox, oy, r in self.obstacles:
            if (x - ox) ** 2 + (y - oy) ** 2 <= r ** 2:
                return True
        return False

    def _update_found_sources(self, x: float, y: float) -> int:
        if self.sources is None or len(self.sources) == 0:
            return 0
        newly_found = 0
        for i, (sx, sy, _q) in enumerate(self.sources):
            if self.found_sources[i]:
                continue
            if (x - sx) ** 2 + (y - sy) ** 2 <= self.source_find_radius ** 2:
                self.found_sources[i] = True
                newly_found += 1
        return newly_found
