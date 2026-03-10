"""
uav_dynamics.py
"""

import math
from typing import Dict, Tuple


class UAVDynamics:
    """
    Simple 2D UAV dynamics and energy consumption model.
    State: position (x, y), velocity (vx, vy), battery.
    """
    # Action encoding: 0 up, 1 down, 2 left, 3 right, 4 up-left, 5 up-right, 6 down-left, 7 down-right
    _ACTION_DIRS = {
        0: (0.0, 1.0),
        1: (0.0, -1.0),
        2: (-1.0, 0.0),
        3: (1.0, 0.0),
        4: (-1.0, 1.0),
        5: (1.0, 1.0),
        6: (-1.0, -1.0),
        7: (1.0, -1.0),
    }

    def __init__(
        self,
        max_speed: float = 5.0,
        max_acceleration: float = 1.0,
        max_battery: float = 100.0,
        energy_consumption_rate: float = 0.1,
        energy_model: str = "linear",
        # Rotorcraft power model params
        w: float = 40.0,
        k: float = 0.1,
        R: float = 0.4,
        deta: float = 0.012,
        Utip: float = 120.0,
        d0: float = 0.6,
        ro: float = 1.225,
        s: float = 0.05,
        A: float = 0.503,
        omiga: float = 300.0,
        rotorcraft_energy_scale: float = 0.001,
        rotorcraft_speed_epsilon: float = 1e-3,
        dt: float = 1.0,
        init_pos: Tuple[float, float] = (0.0, 0.0),
        init_vel: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Args:
            max_speed: Maximum speed (m/s).
            max_acceleration: Maximum acceleration (m/s^2) applied per step.
            max_battery: Maximum battery capacity (percentage units).
            energy_consumption_rate: Battery consumption per unit distance (linear model).
            energy_model: "linear" or "rotorcraft".
            w,k,R,deta,Utip,d0,ro,s,A,omiga: Rotorcraft power model params.
            rotorcraft_energy_scale: Convert physical energy to battery units.
            rotorcraft_speed_epsilon: Min speed used to avoid divide-by-zero.
            dt: Time step per action.
            init_pos: Initial position (x, y).
            init_vel: Initial velocity (vx, vy).
        """
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self.max_battery = float(max_battery)
        self.energy_consumption_rate = float(energy_consumption_rate)
        self.energy_model = str(energy_model).strip().lower()
        self.dt = float(dt)

        # Rotorcraft model parameters
        self.w = float(w)
        self.k = float(k)
        self.R = float(R)
        self.deta = float(deta)
        self.Utip = float(Utip)
        self.d0 = float(d0)
        self.ro = float(ro)
        self.s = float(s)
        self.A = float(A)
        self.omiga = float(omiga)
        self.rotorcraft_energy_scale = float(rotorcraft_energy_scale)
        self.rotorcraft_speed_epsilon = max(1e-8, float(rotorcraft_speed_epsilon))

        # Derived constants from the provided formulas.
        denom = max(2.0 * self.ro * self.A, 1e-8)
        self.v0 = math.sqrt(max(self.w / denom, 0.0))
        self.P0 = (
            (self.deta / 8.0)
            * self.ro
            * self.s
            * self.A
            * (self.omiga ** 3)
            * (self.R ** 3)
        )
        self.Pi = (1.0 + self.k) * (self.w ** 1.5) / math.sqrt(denom)

        self._init_pos = (float(init_pos[0]), float(init_pos[1]))
        self._init_vel = (float(init_vel[0]), float(init_vel[1]))

        self.reset()

    def _normalize(self, dx: float, dy: float) -> Tuple[float, float]:
        norm = math.hypot(dx, dy)
        if norm == 0.0:
            return 0.0, 0.0
        return dx / norm, dy / norm

    def _clamp_speed(self, vx: float, vy: float) -> Tuple[float, float]:
        speed = math.hypot(vx, vy)
        if speed <= self.max_speed or speed == 0.0:
            return vx, vy
        scale = self.max_speed / speed
        return vx * scale, vy * scale

    def move(self, action: int) -> None:
        """
        Apply an action to update position, velocity, and battery.
        """
        if action not in self._ACTION_DIRS:
            raise ValueError(f"Invalid action {action}, must be 0-7")

        dx, dy = self._ACTION_DIRS[action]
        ax_dir, ay_dir = self._normalize(dx, dy)

        # acceleration
        ax = ax_dir * self.max_acceleration
        ay = ay_dir * self.max_acceleration

        # update velocity
        self.vx += ax * self.dt
        self.vy += ay * self.dt
        self.vx, self.vy = self._clamp_speed(self.vx, self.vy)

        # update position
        new_x = self.x + self.vx * self.dt
        new_y = self.y + self.vy * self.dt

        # distance traveled in this step
        distance = math.hypot(new_x - self.x, new_y - self.y)
        self.update_energy(distance)

        self.x = new_x
        self.y = new_y

    def update_energy(self, distance_traveled: float) -> None:
        """
        Reduce battery based on traveled distance.
        """
        distance = max(0.0, float(distance_traveled))
        if self.energy_model == "rotorcraft":
            consumption = self._rotorcraft_consumption(distance)
        else:
            consumption = distance * self.energy_consumption_rate
        self.battery = max(0.0, min(self.max_battery, self.battery - consumption))

    def _rotorcraft_consumption(self, distance_traveled: float) -> float:
        """
        Rotorcraft energy model:
        E_step = distance * (P(x)/x) * scale, where x is step speed.
        """
        if distance_traveled <= 0.0:
            return 0.0

        dt = max(self.dt, 1e-8)
        x = max(distance_traveled / dt, self.rotorcraft_speed_epsilon)
        v0 = max(self.v0, self.rotorcraft_speed_epsilon)
        Utip = max(self.Utip, self.rotorcraft_speed_epsilon)

        term1 = self.P0 * (1.0 + 3.0 * (x ** 2) / (Utip ** 2))
        inner = math.sqrt(1.0 + (x ** 4) / (4.0 * (v0 ** 4))) - (x ** 2) / (2.0 * (v0 ** 2))
        term2 = self.Pi * math.sqrt(max(inner, 0.0))
        term3 = self.d0 * self.ro * self.s * self.A * (x ** 3) / 2.0

        energy_per_distance = (term1 + term2 + term3) / x
        return distance_traveled * energy_per_distance * self.rotorcraft_energy_scale

    def get_state(self) -> Dict[str, float]:
        """
        Return current state as a dict.
        """
        return {
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "battery": self.battery,
        }

    def reset(self) -> None:
        """
        Reset UAV state to initial values.
        """
        self.x, self.y = self._init_pos
        self.vx, self.vy = self._init_vel
        self.battery = self.max_battery

    def is_battery_empty(self) -> bool:
        """
        Check if battery is depleted.
        """
        return self.battery <= 0.0

