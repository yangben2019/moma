"""Tests for the optional slope-range clipping."""

import numpy as np

from moma.mobility import MobilityTensor

PARAMETERS = [np.array([0.4, 0.6]), np.array([0.0, 20.0])]
VALUES = np.array([[10.0, 0.0], [10.0, 0.0]])
# call order follows MobilityTensor.parameter_mask:
# moisture_content (unused), friction_coefficient, slope_angle
CALL = np.array([0.0, 0.5, 30.0])


def _tensor(clip):
    return MobilityTensor(
        "HT",
        PARAMETERS,
        ["friction_coefficient"],
        ["slope_angle"],
        VALUES,
        "test",
        clip_mobility_to_range=clip,
    )


def test_extrapolates_by_default():
    value = float(np.asarray(_tensor(False)(CALL)))
    assert value < 0.0


def test_clips_out_of_range():
    value = float(np.asarray(_tensor(True)(CALL)))
    assert value == 0.0
