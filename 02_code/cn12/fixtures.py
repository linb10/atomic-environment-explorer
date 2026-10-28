"""Independently specified ideal coordinate fixtures and expected signatures."""

from __future__ import annotations

from math import cos, pi, sin, sqrt

import numpy as np


def cuboctahedron() -> np.ndarray:
    points = []
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            points.append((x, y, 0.0))
            points.append((x, 0.0, y))
            points.append((0.0, x, y))
    return np.unique(np.asarray(points, dtype=float), axis=0)


def anticuboctahedron() -> np.ndarray:
    points = [(cos(k*pi/3), sin(k*pi/3), 0.0) for k in range(6)]
    for sign in (-1.0, 1.0):
        points.extend(
            (cos(pi/6 + 2*k*pi/3) / sqrt(3),
             sin(pi/6 + 2*k*pi/3) / sqrt(3),
             sign * sqrt(2/3))
            for k in range(3)
        )
    return np.asarray(points, dtype=float)


def icosahedron() -> np.ndarray:
    phi = (1.0 + sqrt(5.0)) / 2.0
    points = []
    for a in (-1.0, 1.0):
        for b in (-1.0, 1.0):
            points.append((0.0, a, b*phi))
            points.append((a, b*phi, 0.0))
            points.append((b*phi, 0.0, a))
    return np.unique(np.asarray(points, dtype=float), axis=0)


def hexagonal_prism(half_height: float = 0.5) -> np.ndarray:
    return np.asarray(
        [(cos(k*pi/3), sin(k*pi/3), sign*half_height)
         for sign in (-1.0, 1.0) for k in range(6)],
        dtype=float,
    )


def deterministic_non_target() -> np.ndarray:
    """A seeded generic convex 12-point set, not a target reference geometry."""
    rng = np.random.default_rng(20261004)
    points = rng.normal(size=(12, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


EXPECTED_IDEAL_SIGNATURES = {
    "cuboctahedron": {
        "bisecting_planes": 4, "faces": {3: 8, 4: 6}, "edges": 24,
        "degree": 4, "face_adjacency": {"quadrilateral-triangle": 24},
    },
    "anticuboctahedron": {
        "bisecting_planes": 1, "faces": {3: 8, 4: 6}, "edges": 24,
        "degree": 4, "face_adjacency": {
            "quadrilateral-triangle": 18,
            "triangle-triangle": 3,
            "quadrilateral-quadrilateral": 3,
        },
    },
    "icosahedron": {
        "bisecting_planes": 0, "faces": {3: 20}, "edges": 30,
        "degree": 5, "face_adjacency": {"triangle-triangle": 30},
    },
    "hexagonal_prism": {
        "bisecting_planes": 0, "faces": {4: 6, 6: 2}, "edges": 18,
        "degree": 3, "face_adjacency": {
            "hexagon-quadrilateral": 12,
            "quadrilateral-quadrilateral": 6,
        },
    },
}
