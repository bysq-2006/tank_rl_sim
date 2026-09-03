import pytest

from tools.video_speed_analyzer import (
    DirectionSample,
    PointSample,
    calculate_distance,
    calculate_angular_speed,
    calculate_linear_speed,
)


def test_linear_speed_uses_video_frames_and_unit_scale():
    first = PointSample(frame=0, x=10, y=10)
    second = PointSample(frame=24, x=40, y=50)
    distance, speed = calculate_linear_speed(first, second, fps=24, unit_pixels=50)
    assert distance == pytest.approx(1.0)
    assert speed == pytest.approx(1.0)


def test_angular_speed_uses_shortest_signed_angle():
    first = DirectionSample(frame=0, x0=0, y0=0, x1=1, y1=0)
    second = DirectionSample(frame=12, x0=0, y0=0, x1=0, y1=1)
    angle, speed = calculate_angular_speed(first, second, fps=24)
    assert angle == pytest.approx(90.0)
    assert speed == pytest.approx(180.0)


def test_samples_at_same_frame_are_rejected():
    first = PointSample(frame=5, x=0, y=0)
    second = PointSample(frame=5, x=1, y=1)
    with pytest.raises(ValueError):
        calculate_linear_speed(first, second, fps=24, unit_pixels=10)


def test_distance_is_converted_to_unit_cells():
    pixels, units = calculate_distance((10, 20), (40, 60), unit_pixels=25)
    assert pixels == pytest.approx(50.0)
    assert units == pytest.approx(2.0)
