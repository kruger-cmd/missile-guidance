"""Tests for missile guidance seeker module."""

import numpy as np

from missile_guidance.seeker import SeekerModel, KalmanTracker, NoisyMeasurement


def test_seeker_within_fov():
    seeker = SeekerModel(noise_std=5.0, fov_deg=30.0)
    true_pos = np.array([100.0, 200.0, 300.0])
    los = np.array([1.0, 0.0, 0.0])  # target directly ahead
    measurement = seeker.measure(true_pos, los)
    assert measurement is not None
    assert np.linalg.norm(measurement.position - true_pos) < 20.0


def test_seeker_outside_fov():
    seeker = SeekerModel(noise_std=5.0, fov_deg=30.0)
    true_pos = np.array([0.0, 100.0, 0.0])  # target to the side
    los = np.array([0.0, 1.0, 0.0])
    measurement = seeker.measure(true_pos, los)
    # Outside FOV, should return None
    assert measurement is None


def test_kalman_tracker():
    tracker = KalmanTracker(process_noise=0.1, measurement_noise=5.0)
    true_pos = np.array([100.0, 200.0, 300.0])
    measurement = NoisyMeasurement(position=true_pos + np.random.normal(0, 5, 3), timestamp=0.0, noise_std=5.0)
    tracker.predict(dt=0.1)
    tracker.update(measurement)
    estimate = tracker.estimate()
    # Estimate should be close to true position
    assert np.linalg.norm(estimate - true_pos) < 20.0
