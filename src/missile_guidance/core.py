"""Core missile guidance classes."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class TargetState:
    """Target kinematic state."""

    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([100.0, 0.0, 0.0]))
    acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))


@dataclass
class MissileState:
    """Missile kinematic state."""

    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 10000.0, 0.0]))
    velocity: np.ndarray = field(default_factory=lambda: np.array([200.0, 0.0, 0.0]))
    acceleration: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))


class PNGuidance:
    """Proportional Navigation guidance law."""

    def __init__(
        self,
        navigation_constant: float = 4.0,
        max_g: float = 12.0,
        dt: float = 0.01,
    ) -> None:
        self.N = navigation_constant
        self.g = 9.81
        self.max_accel = max_g * self.g
        self.dt = dt

    def compute_acceleration(
        self,
        missile: MissileState,
        target: TargetState,
        method: str = "pure",
    ) -> np.ndarray:
        """Compute missile acceleration command."""
        r = target.position - missile.position
        v = target.velocity - missile.velocity
        r_norm = np.linalg.norm(r)

        if r_norm < 1e-6:
            return np.zeros(3)

        if method == "pure":
            # 2D-style Pure PN projected into 3D (XY plane engagement)
            # LOS rate as scalar in XY plane
            sigma_dot = (r[0] * v[1] - r[1] * v[0]) / (r[0] ** 2 + r[1] ** 2)
            V_m = np.linalg.norm(missile.velocity)
            if V_m > 1e-6:
                e_Vm = missile.velocity / V_m
                # Perpendicular to velocity in XY plane (90° CCW)
                m = np.array([-e_Vm[1], e_Vm[0], 0.0])
                a_m = self.N * V_m * sigma_dot * m
            else:
                a_m = np.zeros(3)

        elif method == "true":
            V_c = -np.dot(r, v) / r_norm
            n = np.cross(r, np.cross(r, v))
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-6:
                n = n / n_norm
                a_m = self.N * V_c * n
            else:
                a_m = np.zeros(3)

        elif method == "augmented":
            V_c = -np.dot(r, v) / r_norm
            n = np.cross(r, np.cross(r, v))
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-6:
                n = n / n_norm
                a_m = self.N * V_c * n + (self.N / 2) * target.acceleration
            else:
                a_m = np.zeros(3)

        elif method == "generalized":
            V_r = np.linalg.norm(v)
            if V_r > 1e-6:
                e_Vr = v / V_r
                n = np.cross(e_Vr, np.cross(r, v))
                n_norm = np.linalg.norm(n)
                if n_norm > 1e-6:
                    n = n / n_norm
                    a_m = self.N * V_r * n
                else:
                    a_m = np.zeros(3)
            else:
                a_m = np.zeros(3)
        else:
            raise ValueError(f"Unknown PN method: {method}")

        # Apply acceleration limit
        a_norm = np.linalg.norm(a_m)
        if a_norm > self.max_accel:
            a_m = a_m * (self.max_accel / a_norm)

        return a_m


class EngagementSimulation:
    """Run a missile-target engagement."""

    def __init__(
        self,
        missile: MissileState | None = None,
        target: TargetState | None = None,
        guidance: PNGuidance | None = None,
        max_time: float = 100.0,
        intercept_dist: float = 1.0,
    ) -> None:
        self.missile = missile or MissileState()
        self.target = target or TargetState()
        self.guidance = guidance or PNGuidance()
        self.max_time = max_time
        self.intercept_dist = intercept_dist
        self.trajectory_m: list[np.ndarray] = []
        self.trajectory_t: list[np.ndarray] = []

    def step(self) -> bool:
        """Run one simulation step. Returns True if intercept occurred."""
        r = self.target.position - self.missile.position
        if np.linalg.norm(r) < self.intercept_dist:
            return True

        a_m = self.guidance.compute_acceleration(
            self.missile, self.target, method="pure"
        )

        # Euler integration
        self.missile.velocity += a_m * self.guidance.dt
        self.missile.position += self.missile.velocity * self.guidance.dt
        self.target.velocity += self.target.acceleration * self.guidance.dt
        self.target.position += self.target.velocity * self.guidance.dt

        self.trajectory_m.append(self.missile.position.copy())
        self.trajectory_t.append(self.target.position.copy())
        return False

    def run(self) -> dict:
        """Run full engagement and return results."""
        t = 0.0
        while t < self.max_time:
            if self.step():
                return {
                    "intercept": True,
                    "time": t,
                    "missile_trajectory": np.array(self.trajectory_m),
                    "target_trajectory": np.array(self.trajectory_t),
                }
            t += self.guidance.dt

        return {
            "intercept": False,
            "time": t,
            "missile_trajectory": np.array(self.trajectory_m),
            "target_trajectory": np.array(self.trajectory_t),
        }
