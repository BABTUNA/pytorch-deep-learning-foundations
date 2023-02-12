import pytest

from src.autograd.scalars import slope_at


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, -1.0), (1.0, 6.0), (2.0, 19.0)],
)
def test_polynomial_slope(value, expected):
    assert slope_at(value) == pytest.approx(expected)
