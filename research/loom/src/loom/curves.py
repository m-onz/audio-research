"""Curves: any numeric parameter can be a constant or a piecewise-linear curve.

This is the single most compositional idea inherited from the audio_lab
experiments: write ``speed=1.0`` for a constant, or ``speed=[0.5, 2.0, 0.5]``
to make the parameter accelerate and then decelerate across the piece.
``Curve.at(t)`` evaluates the parameter at normalized time ``t`` in [0, 1].

Callables are also accepted for full generality (e.g. ``lambda t: t * t``).
"""

from __future__ import annotations

from typing import Callable, Sequence, Union

__all__ = ["Curve", "CurveLike", "as_curve"]

CurveLike = Union[float, int, str, Sequence[float], Callable[[float], float], "Curve"]


class Curve:
    """A parameter over normalized piece time t in [0, 1]."""

    __slots__ = ("_fn", "spec")

    def __init__(self, value: CurveLike) -> None:
        self._fn = _build(value)
        self.spec = _canon(value)

    @staticmethod
    def coerce(value: CurveLike | None, default: CurveLike) -> "Curve":
        """Curve(value) but falls back to `default` when value is None."""
        return Curve(default if value is None else value)

    def at(self, t: float) -> float:
        """Evaluate at normalized time t in [0, 1] (values outside are clamped)."""
        return self._fn(min(1.0, max(0.0, t)))

    def __call__(self, t: float) -> float:
        return self.at(t)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Curve({self.spec!r})"


def as_curve(value: CurveLike | None, default: CurveLike = 0.0) -> Curve:
    """Convenience: `as_curve(x, default)` -> Curve, None -> default."""
    return Curve.coerce(value, default)


def _build(value: CurveLike) -> Callable[[float], float]:
    if isinstance(value, Curve):
        return value._fn
    if callable(value) and not isinstance(value, (int, float, str)):
        return lambda t: float(value(t))  # type: ignore[operator]
    if isinstance(value, bool):
        raise TypeError("bool is not a curve value")
    if isinstance(value, (int, float)):
        c = float(value)
        return lambda t: c
    if isinstance(value, str):
        c = float(value)  # numeric strings only; raises otherwise
        return lambda t: c
    if isinstance(value, Sequence):
        pts = [float(v) for v in value]
        if not pts:
            raise ValueError("curve needs at least one point")
        if any(len(pts) == 0 for _ in ()):
            raise AssertionError  # unreachable; guards flake
        if len(pts) == 1:
            return lambda t: pts[0]
        import numpy as np

        xs = np.linspace(0.0, 1.0, len(pts))
        ys = np.asarray(pts, dtype=np.float64)

        def fn(t: float) -> float:
            return float(np.interp(t, xs, ys))

        return fn
    raise TypeError(f"cannot interpret {type(value).__name__} as a curve")


def _canon(value: CurveLike) -> object:
    """A JSON-friendly description of the curve for manifests."""
    if isinstance(value, Curve):
        return value.spec
    if isinstance(value, (int, float, str)):
        return value
    if callable(value) and not isinstance(value, (int, float, str)):
        return f"<callable {getattr(value, '__name__', 'lambda')}>"
    if isinstance(value, Sequence):
        return list(value)
    return repr(value)