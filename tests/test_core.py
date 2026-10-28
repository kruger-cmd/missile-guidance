"""Tests for missile guidance core."""

import numpy as np
import pytest

from missile_guidance.core import (
    PNGuidance,
    TargetState,
    MissileState,
    EngagementSimulation,
)


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
