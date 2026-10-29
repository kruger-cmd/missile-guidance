"""6-DOF missile dynamics with mass properties and aerodynamic forces."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MissileGeometry:
    """Simple missile geometry and mass properties."""

    length: float = 2.0
    diameter: float = 0.15
    mass: float = 100.0
    inertia: np.ndarray = None  # type: ignore

    def __post_init__(self) -> None:
        if self.inertia is None:
            # Approximate rod inertia about center
            I_long = self.mass * self.length ** 2 / 12.0
            I_rad = self.mass * self.diameter ** 2 / 8.0
            self.inertia = np.diag([I_rad, I_rad, I_long])


@dataclass
class AerodynamicModel:
    """Simple aerodynamic force/moment model."""

    reference_area: float = 0.02
    cd: float = 0.3
    cl: float = 3.0
    rho: float = 1.225
    cg_offset: np.ndarray = None  # type: ignore

    def __post_init__(self) -> None:
        if self.cg_offset is None:
            self.cg_offset = np.zeros(3)

    def forces(self, velocity: np.ndarray, alpha: float, control: np.ndarray) -> np.ndarray:
        """Return body-frame aerodynamic force."""
        v = np.linalg.norm(velocity)
        if v < 1e-6:
            return np.zeros(3)

        q = 0.5 * self.rho * v ** 2
        drag = q * self.reference_area * self.cd * np.array([-1.0, 0.0, 0.0])
        lift = q * self.reference_area * self.cl * np.array([0.0, 0.0, 1.0]) * np.sin(alpha)
        return drag + lift

    def moments(self, velocity: np.ndarray, omega: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Return body-frame aerodynamic moment."""
        return control * 50.0 - 0.1 * omega


class SixDOFMissile:
    """6-DOF rigid-body missile model."""

    def __init__(self, geom: MissileGeometry | None = None, aero: AerodynamicModel | None = None) -> None:
        self.geom = geom or MissileGeometry()
        self.aero = aero or AerodynamicModel()

    def equations_of_motion(self, state: np.ndarray, thrust: float, control: np.ndarray) -> np.ndarray:
        """State derivative for 12-state [pos, vel, quat, omega]."""
        if state.shape[0] != 12:
            raise ValueError(f"Expected 12-state, got {state.shape[0]}")

        pos = state[0:3]
        vel = state[3:6]
        quat = state[6:10]
        omega = state[10:13] if state.shape[0] == 13 else np.zeros(3)

        # Normalize quaternion if present
        if np.linalg.norm(quat) > 1e-6:
            quat = quat / np.linalg.norm(quat)

        # Body-axis velocity
        v_body = vel

        # Angle of attack approx
        v_norm = np.linalg.norm(v_body)
        if v_norm > 1e-6:
            alpha = np.arctan2(v_body[2], v_body[0])
        else:
            alpha = 0.0

        # Forces: thrust along body X, gravity in Z, aero in body frame
        R = self._quaternion_to_rotation(quat)
        thrust_vec = thrust * R[:, 0]
        gravity = np.array([0.0, 0.0, -self.geom.mass * 9.81])
        aero_force_body = self.aero.forces(v_body, alpha, control)
        aero_force_world = R @ aero_force_body

        acc = (thrust_vec + gravity + aero_force_world) / self.geom.mass

        # Quaternion derivative
        q_dot = 0.5 * self._quaternion_multiply(quat, np.array([0.0, *omega]))

        # Angular acceleration
        I = self.geom.inertia
        angular_acc = np.linalg.inv(I) @ (self.aero.moments(v_body, omega, control) - np.cross(omega, I @ omega))

        return np.concatenate([vel, acc, q_dot, angular_acc])

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def _quaternion_to_rotation(self, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)],
        ])


@dataclass
class GuidanceOutput:
    """Output from guidance law."""

    acceleration_command: np.ndarray
    desired_pitch: float = 0.0
    desired_yaw: float = 0.0
