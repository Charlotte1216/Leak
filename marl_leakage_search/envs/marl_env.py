"""
marl_env.py
"""

import os
import random
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from .uav_dynamics import UAVDynamics
    from .concentration_field import ConcentrationField
except ImportError:
    from uav_dynamics import UAVDynamics
    from concentration_field import ConcentrationField


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
        success_done_bonus: float | None = None, # 成功完成奖励（找到全部源）
        failure_done_penalty: float = 0.0, # 失败完成惩罚（电池耗尽）
        enable_collision: bool = True, # 是否启用与障碍物的碰撞检测/惩罚
        stop_on_collision: bool = True, # 碰撞时是否停止无人机
        observe_wind: bool = False, # 观测中是否包含风速/风向
        observe_velocity: bool = False, # 观测中是否包含速度 (vx, vy)
        distance_reward_scale: float = 0.0, # 距离源的密集奖励系数
        found_source_concentration_scale: float = 1.0,
        found_source_stay_penalty: float = 0.0,
        found_source_stay_radius_scale: float = 1.0,
        init_pos_mode: str = "random",
        seed: int = None,
        uav_params: Dict = None,
        dynamic_field_config: Dict | None = None,
        communication_config: Dict | None = None,
    ):
        self.field_dir = Path(field_dir)
        self.num_agents = num_agents
        self.source_find_radius = float(source_find_radius)
        self.collision_penalty = float(collision_penalty)
        self.battery_penalty = float(battery_penalty)
        self.found_source_bonus = float(found_source_bonus)
        self.done_bonus = float(done_bonus)
        self.success_done_bonus = float(done_bonus if success_done_bonus is None else success_done_bonus)
        self.failure_done_penalty = float(failure_done_penalty)
        self.enable_collision = bool(enable_collision)
        self.stop_on_collision = bool(stop_on_collision)
        self.observe_wind = bool(observe_wind)
        self.observe_velocity = bool(observe_velocity)
        self.distance_reward_scale = float(distance_reward_scale)
        self.found_source_concentration_scale = float(found_source_concentration_scale)
        self.found_source_stay_penalty = float(found_source_stay_penalty)
        self.found_source_stay_radius_scale = float(found_source_stay_radius_scale)
        self.init_pos_mode = init_pos_mode
        self.rng = random.Random(seed)
        self.dynamic_field_config = dict(dynamic_field_config or {})
        self.dynamic_field_enabled = bool(self.dynamic_field_config.get("enabled", False))
        self.field_dt = float(self.dynamic_field_config.get("dt", 1.0))
        self.field_time = 0.0
        self.communication_config = dict(communication_config or {})
        self.communication_enabled = bool(self.communication_config.get("enabled", False))
        self.communication_top_k = max(0, int(self.communication_config.get("top_k", 2)))
        self.communication_radius = float(self.communication_config.get("radius", 0.0))
        self.communication_include_concentration = bool(
            self.communication_config.get("include_concentration", True)
        )
        self.communication_include_battery = bool(
            self.communication_config.get("include_battery", True)
        )
        self.communication_include_velocity = bool(
            self.communication_config.get("include_velocity", False)
        )
        self.communication_normalize_relative = bool(
            self.communication_config.get("normalize_relative", True)
        )
        channel_cfg = self.communication_config.get("channel", {})
        if not isinstance(channel_cfg, dict):
            channel_cfg = {}
        self.communication_channel_enabled = bool(channel_cfg.get("enabled", False))
        self.communication_channel_mode = str(channel_cfg.get("mode", "none")).strip().lower()
        self.communication_nlos_gain = float(channel_cfg.get("nlos_gain", 0.35))
        self.communication_nlos_noise_std = max(0.0, float(channel_cfg.get("nlos_noise_std", 0.0)))
        self.communication_los_drop_prob = min(
            1.0, max(0.0, float(channel_cfg.get("los_drop_prob", 0.0)))
        )
        self.communication_nlos_drop_prob = min(
            1.0, max(0.0, float(channel_cfg.get("nlos_drop_prob", 0.0)))
        )
        self.communication_add_link_flag = bool(channel_cfg.get("add_link_flag", False))
        self.communication_feature_dim_per_neighbor = (
            2
            + (1 if self.communication_include_concentration else 0)
            + (1 if self.communication_include_battery else 0)
            + (2 if self.communication_include_velocity else 0)
            + (2 if self.communication_add_link_flag else 0)
        )

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
        self._max_distance = None
        self._concentration_model: ConcentrationField | None = None

    def reset(self):
        """
        Load a random concentration field and reset UAVs.
        Returns initial observations for all agents.
        """
        data = self.random_select_field()

        # load field data
        if "concentration_field" in data:
            self.concentration = np.asarray(data["concentration_field"], dtype=float)
        elif "concentration" in data:
            self.concentration = np.asarray(data["concentration"], dtype=float)
        else:
            if self.dynamic_field_enabled and "x" in data and "y" in data:
                x_vals = np.asarray(data["x"], dtype=float)
                y_vals = np.asarray(data["y"], dtype=float)
                self.concentration = np.zeros((len(y_vals), len(x_vals)), dtype=float)
            else:
                raise KeyError("Missing concentration field in .npz file")

        self.sources = self._coerce_matrix(data.get("sources"), cols=3)
        self.obstacles = self._coerce_matrix(data.get("obstacles"), cols=3)
        self.wind_speed = self._extract_scalar(data, "wind_speed", 1.0)
        if "wind_dir" in data:
            self.wind_dir = self._extract_scalar(data, "wind_dir", 0.0)
        elif "wind_direction" in data:
            self.wind_dir = self._extract_scalar(data, "wind_direction", 0.0)
        else:
            self.wind_dir = 0.0
        self.field_time = self._extract_scalar(data, "t", 0.0)
        self.x_vals = np.asarray(data.get("x"), dtype=float) if data.get("x") is not None else None
        self.y_vals = np.asarray(data.get("y"), dtype=float) if data.get("y") is not None else None
        x_min, x_max, y_min, y_max = self._get_bounds()
        self._max_distance = math.hypot(x_max - x_min, y_max - y_min)
        self._concentration_model = self._build_dynamic_field_model(data)

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
        found_mask_snapshot = None if self.found_sources is None else self.found_sources.copy()
        prev_nearest_distances = [
            self._nearest_unfound_distance(uav.x, uav.y, found_mask=found_mask_snapshot)
            for uav in self.uavs
        ]
        collision_terms = [0.0] * self.num_agents
        conc_terms = [0.0] * self.num_agents
        found_terms = [0.0] * self.num_agents
        battery_terms = [0.0] * self.num_agents
        success_done_terms = [0.0] * self.num_agents
        failure_done_terms = [0.0] * self.num_agents
        distance_terms = [0.0] * self.num_agents
        found_source_stay_terms = [0.0] * self.num_agents

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

        if self.dynamic_field_enabled:
            self.field_time += self.field_dt

        # compute concentration-based rewards and found sources
        for i, uav in enumerate(self.uavs):
            conc = self._sample_concentration(uav.x, uav.y)
            near_previously_found_source = self._is_near_found_source(
                uav.x,
                uav.y,
                found_mask=found_mask_snapshot,
            )
            conc_reward = conc
            if near_previously_found_source:
                conc_reward *= self.found_source_concentration_scale
            conc_terms[i] = conc_reward
            rewards[i] += conc_terms[i]

            newly_found = self._update_found_sources(uav.x, uav.y)
            if newly_found > 0:
                found_terms[i] = self.found_source_bonus * newly_found
                rewards[i] += found_terms[i]

            # battery penalty based on consumption
            battery_terms[i] = -(prev_batteries[i] - uav.battery) * self.battery_penalty
            rewards[i] += battery_terms[i]

            # Dense shaping reward: reward reduction in nearest-source distance.
            if self.distance_reward_scale > 0.0 and self._max_distance and self._max_distance > 0:
                prev_d = prev_nearest_distances[i]
                curr_d = self._nearest_unfound_distance(uav.x, uav.y, found_mask=found_mask_snapshot)
                if prev_d is not None and curr_d is not None:
                    delta = (prev_d - curr_d) / self._max_distance
                    distance_terms[i] = self.distance_reward_scale * float(delta)
                    rewards[i] += distance_terms[i]

            if (
                near_previously_found_source
                and self.found_source_stay_penalty > 0.0
                and self.found_sources is not None
                and np.any(~self.found_sources)
            ):
                found_source_stay_terms[i] = -self.found_source_stay_penalty
                rewards[i] += found_source_stay_terms[i]

        success_done = self.found_sources is not None and self.found_sources.all()
        failure_done = all(uav.is_battery_empty() for uav in self.uavs) and not success_done
        done = bool(success_done or failure_done)
        if done and success_done:
            success_done_terms = [self.success_done_bonus for _ in range(self.num_agents)]
            rewards = [r + self.success_done_bonus for r in rewards]
        elif done and failure_done:
            failure_done_terms = [-self.failure_done_penalty for _ in range(self.num_agents)]
            rewards = [r - self.failure_done_penalty for r in rewards]

        next_state = self._get_observations()
        legacy_done_terms = [s + f for s, f in zip(success_done_terms, failure_done_terms)]
        info = {
            "found_sources": int(self.found_sources.sum()),
            "total_sources": int(len(self.sources)),
            "success_done": bool(success_done),
            "failure_done": bool(failure_done),
            "field_time": float(self.field_time),
            "wind_speed": float(self.wind_speed),
            "wind_dir": float(self.wind_dir),
            "reward_components": {
                "collision": collision_terms,
                "concentration": conc_terms,
                "found_bonus": found_terms,
                "battery": battery_terms,
                "distance": distance_terms,
                "found_source_stay": found_source_stay_terms,
                "success_done": success_done_terms,
                "failure_done": failure_done_terms,
                "done_bonus": legacy_done_terms,
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

        concentration_to_plot = self.concentration
        if self.dynamic_field_enabled and self._concentration_model is not None:
            if self.x_vals is not None and self.y_vals is not None:
                X, Y = np.meshgrid(self.x_vals, self.y_vals, indexing="xy")
            else:
                h, w = self.concentration.shape
                X, Y = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float), indexing="xy")
            concentration_to_plot = self._concentration_model.concentration(X, Y, t=self.field_time)

        ax.imshow(concentration_to_plot, origin="lower", extent=extent, cmap="plasma", aspect="auto")

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
        return dict(np.load(file_path, allow_pickle=True))

    def _extract_scalar(self, data: Dict[str, Any], key: str, default: float) -> float:
        value = data.get(key, None)
        if value is None:
            return float(default)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return float(default)
            return float(np.asarray(value).reshape(-1)[0])
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _extract_dict(self, data: Dict[str, Any], key: str, default: Dict | None = None) -> Dict[str, Any]:
        value = data.get(key, None)
        if value is None:
            return dict(default or {})
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return dict(default or {})
            maybe_dict = np.asarray(value).reshape(-1)[0]
            if isinstance(maybe_dict, dict):
                return dict(maybe_dict)
        return dict(default or {})

    def _coerce_matrix(self, value: Any, cols: int) -> np.ndarray:
        if value is None:
            return np.zeros((0, cols), dtype=float)
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return np.zeros((0, cols), dtype=float)
        if arr.size == 0:
            return np.zeros((0, cols), dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] % cols != 0:
                return np.zeros((0, cols), dtype=float)
            arr = arr.reshape(-1, cols)
        else:
            arr = arr.reshape(-1, arr.shape[-1])
            if arr.shape[1] != cols:
                return np.zeros((0, cols), dtype=float)
        return arr

    def _as_source_list(self) -> List[Dict[str, float]]:
        if self.sources is None or len(self.sources) == 0:
            return []
        return [
            {"x": float(sx), "y": float(sy), "Q": float(q)}
            for sx, sy, q in self.sources
        ]

    def _as_obstacle_list(self) -> List[Dict[str, float]]:
        if self.obstacles is None or len(self.obstacles) == 0:
            return []
        return [
            {"x": float(ox), "y": float(oy), "radius": float(r)}
            for ox, oy, r in self.obstacles
        ]

    def _build_dynamic_field_model(self, data: Dict[str, Any]) -> ConcentrationField | None:
        if not self.dynamic_field_enabled:
            return None

        plume_params = self._extract_dict(
            data,
            "plume_params",
            default={"L": 100.0, "sigma_y": 5.0},
        )
        vortex_params = self._extract_dict(data, "vortex_params", default={})
        noise_std = self._extract_scalar(data, "noise_std", 0.0)

        dynamic_cfg = self.dynamic_field_config
        keep_plume_behind_obstacle = bool(dynamic_cfg.get("keep_plume_behind_obstacle", True))

        wind_time_params: Dict[str, Any] = {}
        wind_cfg = dynamic_cfg.get("wind", {})
        if isinstance(wind_cfg, dict):
            wind_time_params.update(wind_cfg)
        wind_time_params.setdefault("enabled", False)

        vortex_cfg = dynamic_cfg.get("vortex", {})
        if isinstance(vortex_cfg, dict):
            for key in (
                "use_strouhal",
                "strouhal",
                "min_wind_speed",
                "min_diameter",
            ):
                if key in vortex_cfg:
                    vortex_params[key] = vortex_cfg[key]

        return ConcentrationField(
            sources=self._as_source_list(),
            obstacles=self._as_obstacle_list(),
            wind_speed=self.wind_speed,
            wind_dir=self.wind_dir,
            plume_params=plume_params,
            vortex_params=vortex_params,
            keep_plume_behind_obstacle=keep_plume_behind_obstacle,
            noise_std=noise_std,
            wind_time_params=wind_time_params,
        )

    def _get_observations(self) -> List[np.ndarray]:
        obs = []
        concentrations = [self._sample_concentration(uav.x, uav.y) for uav in self.uavs]
        for idx, uav in enumerate(self.uavs):
            conc = concentrations[idx]
            payload = [uav.x, uav.y, conc, uav.battery]
            if self.observe_velocity:
                payload.extend([uav.vx, uav.vy])
            if self.observe_wind:
                payload.extend([self.wind_speed, math.cos(self.wind_dir), math.sin(self.wind_dir)])
            if self.communication_enabled and self.communication_top_k > 0:
                payload.extend(self._build_communication_features(idx, concentrations))
            obs.append(np.array(payload, dtype=float))
        return obs

    def _build_communication_features(
        self,
        agent_idx: int,
        concentrations: List[float],
    ) -> List[float]:
        """Build fixed-size Top-K neighbor communication features for one agent."""
        self_uav = self.uavs[agent_idx]
        neighbors: List[Tuple[float, int]] = []
        for other_idx, other_uav in enumerate(self.uavs):
            if other_idx == agent_idx:
                continue
            dist = math.hypot(other_uav.x - self_uav.x, other_uav.y - self_uav.y)
            if self.communication_radius > 0.0 and dist > self.communication_radius:
                continue
            neighbors.append((dist, other_idx))

        neighbors.sort(key=lambda item: item[0])
        features: List[float] = []
        normalizer = self._max_distance if (self.communication_normalize_relative and self._max_distance) else 1.0
        if normalizer is None or normalizer <= 0.0:
            normalizer = 1.0

        for _, other_idx in neighbors[: self.communication_top_k]:
            other_uav = self.uavs[other_idx]
            is_los = self._has_line_of_sight(self_uav.x, self_uav.y, other_uav.x, other_uav.y)
            link_gain = 1.0
            drop_prob = 0.0
            noise_std = 0.0
            if self.communication_channel_enabled and self.communication_channel_mode == "los_nlos":
                if not is_los:
                    link_gain = self.communication_nlos_gain
                    noise_std = self.communication_nlos_noise_std
                drop_prob = self.communication_los_drop_prob if is_los else self.communication_nlos_drop_prob

            if drop_prob > 0.0 and self.rng.random() < drop_prob:
                features.extend([0.0] * self.communication_feature_dim_per_neighbor)
                continue

            dx = (other_uav.x - self_uav.x) / normalizer
            dy = (other_uav.y - self_uav.y) / normalizer
            neighbor_features: List[float] = [float(dx), float(dy)]
            if self.communication_include_concentration:
                neighbor_features.append(float(concentrations[other_idx]))
            if self.communication_include_battery:
                neighbor_features.append(float(other_uav.battery))
            if self.communication_include_velocity:
                neighbor_features.extend([float(other_uav.vx), float(other_uav.vy)])

            if self.communication_channel_enabled and self.communication_channel_mode == "los_nlos":
                processed = []
                for value in neighbor_features:
                    noisy = value * link_gain
                    if noise_std > 0.0:
                        noisy += self.rng.gauss(0.0, noise_std)
                    processed.append(float(noisy))
                neighbor_features = processed

            if self.communication_add_link_flag:
                neighbor_features.extend([1.0 if is_los else 0.0, float(link_gain)])

            features.extend(neighbor_features)

        selected = min(len(neighbors), self.communication_top_k)
        missing = self.communication_top_k - selected
        if missing > 0:
            features.extend([0.0] * (missing * self.communication_feature_dim_per_neighbor))

        return features

    def _has_line_of_sight(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """
        LOS is blocked if the segment between two UAVs intersects any obstacle circle.
        """
        if self.obstacles is None or len(self.obstacles) == 0:
            return True
        for ox, oy, r in self.obstacles:
            if self._segment_intersects_circle(x1, y1, x2, y2, float(ox), float(oy), float(r)):
                return False
        return True

    @staticmethod
    def _segment_intersects_circle(
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        cx: float,
        cy: float,
        radius: float,
    ) -> bool:
        dx = x2 - x1
        dy = y2 - y1
        rr = max(0.0, radius) ** 2
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return (x1 - cx) ** 2 + (y1 - cy) ** 2 <= rr
        t = ((cx - x1) * dx + (cy - y1) * dy) / denom
        t = max(0.0, min(1.0, t))
        px = x1 + t * dx
        py = y1 + t * dy
        return (px - cx) ** 2 + (py - cy) ** 2 <= rr

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
        if self.dynamic_field_enabled and self._concentration_model is not None:
            sample_x = np.array([float(x)], dtype=float)
            sample_y = np.array([float(y)], dtype=float)
            c = self._concentration_model.concentration(sample_x, sample_y, t=self.field_time)
            # Keep observation wind terms aligned with the dynamic field state.
            self.wind_speed = float(self._concentration_model.wind_speed)
            self.wind_dir = float(self._concentration_model.wind_dir)
            return float(np.asarray(c).reshape(-1)[0])

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

    def _nearest_unfound_distance(
        self,
        x: float,
        y: float,
        found_mask: np.ndarray | None = None,
    ) -> float | None:
        if self.sources is None or len(self.sources) == 0:
            return None
        if found_mask is None:
            found_mask = self.found_sources
        if found_mask is None:
            return None
        d_min = None
        for i, (sx, sy, _q) in enumerate(self.sources):
            if found_mask[i]:
                continue
            d = math.hypot(x - sx, y - sy)
            if d_min is None or d < d_min:
                d_min = d
        return d_min

    def _is_near_found_source(
        self,
        x: float,
        y: float,
        found_mask: np.ndarray | None = None,
    ) -> bool:
        if self.sources is None or len(self.sources) == 0:
            return False
        if found_mask is None:
            found_mask = self.found_sources
        if found_mask is None or not np.any(found_mask):
            return False

        radius = self.source_find_radius * self.found_source_stay_radius_scale
        if radius <= 0.0:
            return False
        radius_sq = radius * radius

        for i, (sx, sy, _q) in enumerate(self.sources):
            if not found_mask[i]:
                continue
            if (x - sx) ** 2 + (y - sy) ** 2 <= radius_sq:
                return True
        return False
