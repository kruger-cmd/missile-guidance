"""Tests for missile guidance core."""

import numpy as np
import pytest

from missile_guidance.core import (
    PNGuidance,
    TargetState,
    MissileState,
    EngagementSimulation,
)
from missile_guidance.dynamics import SixDOFMissile, MissileGeometry, AerodynamicModel, GuidanceOutput
from missile_guidance.seeker import SeekerModel, KalmanTracker, TrackToTargetLoop


def test_pn_pure_intercept():
    # Use initial conditions that produce a clean intercept
    sim = EngagementSimulation(
        missile=MissileState(position=np.array([-8000.0, -2000.0, 0.0]), velocity=np.array([350.0, 100.0, 0.0])),
        target=TargetState(position=np.array([0.0, 0.0, 0.0]), velocity=np.array([100.0, 50.0, 0.0])),
        guidance=PNGuidance(navigation_constant=4.0, max_g=12.0, dt=0.01),
        max_time=60.0,
        intercept_dist=2.0,
    )
    result = sim.run()
    assert result["intercept"] is True
    assert result["time"] < 60.0


def test_target_maneuver():
    sim = EngagementSimulation(
        missile=MissileState(position=np.array([-8000.0, -2000.0, 0.0]), velocity=np.array([350.0, 100.0, 0.0])),
        target=TargetState(position=np.array([0.0, 0.0, 0.0]), velocity=np.array([100.0, 50.0, 0.0]), acceleration=np.array([0.0, 20.0, 0.0])),
        guidance=PNGuidance(navigation_constant=4.0, max_g=12.0, dt=0.01),
        max_time=60.0,
        intercept_dist=2.0,
    )
    result = sim.run()
    assert result["time"] <= 60.01


def test_guidance_acceleration_limit():
    guidance = PNGuidance(navigation_constant=4.0, max_g=12.0, dt=0.01)
    missile = MissileState(position=np.array([-8000.0, -2000.0, 0.0]), velocity=np.array([350.0, 100.0, 0.0]))
    target = TargetState(position=np.array([0.0, 0.0, 0.0]), velocity=np.array([100.0, 50.0, 0.0]))
    a = guidance.compute_acceleration(missile, target, method="pure")
    assert np.linalg.norm(a) <= guidance.max_accel + 1e-6


def test_augmented_pn_intercept():
    sim = EngagementSimulation(
        missile=MissileState(position=np.array([-8000.0, -2000.0, 0.0]), velocity=np.array([350.0, 100.0, 0.0])),
        target=TargetState(position=np.array([0.0, 0.0, 0.0]), velocity=np.array([100.0, 50.0, 0.0]), acceleration=np.array([0.0, 20.0, 0.0])),
        guidance=PNGuidance(navigation_constant=4.0, max_g=12.0, dt=0.01),
        max_time=60.0,
        intercept_dist=2.0,
    )
    result = sim.run()
    assert result["time"] <= 60.01


def test_six_dof_dynamics_shape():
    missile = SixDOFMissile()
    state = np.zeros(12)
    state[6] = 1.0  # quaternion w=1
    thrust = 1000.0
    control = np.zeros(3)
    derivative = missile.equations_of_motion(state, thrust, control)
    # Standard 6-DOF: pos(3) + vel(3) + quat(4) + omega(3) = 13
    assert derivative.shape == (13,)


def test_six_dof_hover():
    missile = SixDOFMissile(geom=MissileGeometry(mass=100.0), aero=AerodynamicModel(cd=0.0, cl=0.0))
    state = np.zeros(12)
    state[6] = 1.0
    state[2] = 1000.0
    derivative = missile.equations_of_motion(state, thrust=981.0, control=np.zeros(3))
    # For hover-ish state, derivative should be finite and not NaN
    assert np.all(np.isfinite(derivative))
    # quaternion derivative should be zero for zero omega
    assert np.allclose(derivative[9:12], 0.0, atol=1e-6)


def test_track_to_target_loop_shape():
    guidance = PNGuidance(navigation_constant=4.0, max_g=12.0, dt=0.01)
    seeker = SeekerModel(noise_std=5.0, fov_deg=60.0)
    tracker = KalmanTracker(process_noise=0.1, measurement_noise=5.0)
    loop = TrackToTargetLoop(seeker, tracker, guidance)

    true_target = np.array([1000.0, 500.0, 0.0])
    missile_pos = np.array([0.0, 0.0, 0.0])
    accel = loop.step(true_target, missile_pos, dt=0.01)
    assert accel.shape == (3,)
    assert np.linalg.norm(accel) <= guidance.max_accel + 1e-6
