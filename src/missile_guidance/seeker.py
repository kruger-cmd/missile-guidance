"""Seeker and tracking models for missile guidance."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class NoisyMeasurement:
    """Single seeker measurement with noise."""
    position: np.ndarray
    timestamp: float
    noise_std: float


class SeekerModel:
    """Simple seeker with Gaussian noise."""

    def __init__(self, noise_std: float = 10.0, fov_deg: float = 30.0) -> None:
        self.noise_std = noise_std
        self.fov_rad = np.deg2rad(fov_deg)

    def measure(self, true_position: np.ndarray, los_vector: np.ndarray) -> NoisyMeasurement | None:
        """Return noisy position measurement if target is within FOV."""
        los_norm = np.linalg.norm(los_vector)
        if los_norm < 1e-6:
            return None

        los_unit = los_vector / los_norm
        # Assume seeker looks along missile velocity
        # Simple FOV check: angle between LOS and forward vector
        forward = np.array([1.0, 0.0, 0.0])
        cos_angle = np.dot(los_unit, forward)
        if cos_angle < np.cos(self.fov_rad / 2):
            return None

        noise = np.random.normal(0, self.noise_std, size=3)
        return NoisyMeasurement(
            position=true_position + noise,
            timestamp=0.0,
            noise_std=self.noise_std,
        )


class KalmanTracker:
    """Simple Kalman filter for target tracking."""

    def __init__(self, process_noise: float = 0.1, measurement_noise: float = 10.0) -> None:
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 1000.0
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)
        self.Q = np.eye(6) * process_noise
        self.R = np.eye(3) * measurement_noise

    def predict(self, dt: float) -> None:
        """Predict state forward by dt."""
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.Q

    def update(self, measurement: NoisyMeasurement) -> None:
        """Update with new measurement."""
        z = measurement.position
        y = z - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ self.H) @ self.covariance

    def estimate(self) -> np.ndarray:
        """Return estimated position."""
        return self.state[:3]
