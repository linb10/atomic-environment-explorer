"""Topology-first four-family classification with metric distortion diagnostics."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import minimize_scalar

from .fixtures import (
    EXPECTED_IDEAL_SIGNATURES,
    anticuboctahedron,
    cuboctahedron,
    hexagonal_prism,
    icosahedron,
)
from .geometry import (
    HullSettings,
    PlaneSettings,
    enumerate_center_planes,
    reconstruct_hull,
)
from .models import GeometryDiagnostics, HullDiagnostics, Vector3


REFERENCE_POINTS = {
    "cuboctahedron": cuboctahedron(),
    "anticuboctahedron": anticuboctahedron(),
    "icosahedron": icosahedron(),
    "hexagonal_prism": hexagonal_prism(0.5),
}


def _reference_graph(points: np.ndarray) -> list[set[int]]:
    """Independent ideal graph: pairs at the exact shortest reference distance."""
    delta = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    distances[distances < 1e-12] = np.inf
    edge_length = float(np.min(distances))
    adjacency = [set() for _ in range(len(points))]
    for i, j in combinations(range(len(points)), 2):
        if abs(float(distances[i, j]) - edge_length) <= 1e-8 * max(edge_length, 1.0):
            adjacency[i].add(j)
            adjacency[j].add(i)
    return adjacency


REFERENCE_GRAPHS = {name: _reference_graph(points) for name, points in REFERENCE_POINTS.items()}


def _actual_graph(hull: HullDiagnostics, point_ids: list[str]) -> list[set[int]]:
    index = {key: i for i, key in enumerate(point_ids)}
    graph = [set() for _ in point_ids]
    for a, b in hull.edges:
        if a in index and b in index:
            graph[index[a]].add(index[b])
            graph[index[b]].add(index[a])
    return graph


def _local_graph_signature(graph: list[set[int]], i: int) -> tuple:
    degrees = tuple(sorted(len(graph[j]) for j in graph[i]))
    triangles = sum(1 for a, b in combinations(sorted(graph[i]), 2) if b in graph[a])
    return len(graph[i]), degrees, triangles


def graph_isomorphisms(
    graph_a: list[set[int]], graph_b: list[set[int]], limit: int = 20000
) -> list[dict[int, int]]:
    """Return graph-consistent maps a-node -> b-node with degree/triangle pruning."""
    n = len(graph_a)
    if n != len(graph_b):
        return []
    if sorted(map(len, graph_a)) != sorted(map(len, graph_b)):
        return []
    sig_a = [_local_graph_signature(graph_a, i) for i in range(n)]
    sig_b = [_local_graph_signature(graph_b, i) for i in range(n)]
    candidates = {i: [j for j in range(n) if sig_a[i] == sig_b[j]] for i in range(n)}
    if any(not choices for choices in candidates.values()):
        return []
    def priority(i: int) -> tuple:
        return len(candidates[i]), -len(graph_a[i]), -sig_a[i][2], i

    # Visit the most constrained node first, then always the node with the most
    # already-placed neighbors, so a wrong partial assignment is rejected early.
    order: list[int] = []
    remaining = set(range(n))
    while remaining:
        placed = set(order)
        node = min(remaining, key=lambda i: (-len(graph_a[i] & placed), priority(i)))
        order.append(node)
        remaining.remove(node)
    out: list[dict[int, int]] = []
    mapping: dict[int, int] = {}
    used: set[int] = set()

    def visit(depth: int) -> None:
        if len(out) >= limit:
            return
        if depth == n:
            out.append(mapping.copy())
            return
        node = order[depth]
        for target in candidates[node]:
            if target in used:
                continue
            if any((other in graph_a[node]) != (mapped in graph_b[target])
                   for other, mapped in mapping.items()):
                continue
            mapping[node] = target
            used.add(target)
            visit(depth + 1)
            used.remove(target)
            del mapping[node]

    visit(0)
    return out


def _proper_uniform_fit_rmsd(actual_by_reference: np.ndarray, reference: np.ndarray) -> float:
    """Best center-preserving proper rotation and one uniform scale; no radial scaling."""
    a = np.asarray(actual_by_reference, dtype=float)
    b = np.asarray(reference, dtype=float)
    covariance = a.T @ b
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular * np.diag(correction)) / np.sum(a * a))
    if scale <= 0:
        return float("inf")
    residual = scale * a @ rotation - b
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))) /
                 np.sqrt(np.mean(np.sum(b * b, axis=1))))


def _prism_family_rmsd(actual_by_reference: np.ndarray) -> float:
    """Fit a right regular hexagonal prism while allowing its declared height parameter."""
    actual = np.asarray(actual_by_reference, dtype=float)

    def score(half_height: float) -> float:
        template = hexagonal_prism(float(half_height))
        # The graph-isomorphism map preserves the base and vertical edge classes.
        return _proper_uniform_fit_rmsd(actual, template)

    result = minimize_scalar(score, bounds=(0.02, 5.0), method="bounded",
                             options={"xatol": 1e-5, "maxiter": 60})
    return min(score(float(result.x)), score(0.5))


def _best_shape_rmsd(
    family: str, points: np.ndarray, mappings: list[dict[int, int]]
) -> float | None:
    """Smallest fit residual over the graph-consistent vertex assignments."""
    reference = REFERENCE_POINTS[family]
    if not mappings:
        return None
    best = float("inf")
    for actual_to_reference in mappings:
        actual_by_ref = np.empty_like(points)
        for actual_i, reference_i in actual_to_reference.items():
            actual_by_ref[reference_i] = points[actual_i]
        value = (_prism_family_rmsd(actual_by_ref) if family == "hexagonal_prism"
                 else _proper_uniform_fit_rmsd(actual_by_ref, reference))
        best = min(best, value)
    return best if np.isfinite(best) else None


def _size_signature(hull: HullDiagnostics) -> dict[int, int]:
    return {int(size): int(count) for size, count in hull.face_size_counts.items()}


def _candidate_families(
    bisecting_plane_count: int, hull: HullDiagnostics, point_ids: list[str]
) -> tuple[list[str], dict[str, list[str]], dict[str, list[dict[int, int]]]]:
    """Families whose full signature matches, why the others fail, and the
    vertex assignments found for each match (reused by the shape fit)."""
    if not hull.success or not hull.all_points_are_hull_vertices or not hull.center_inside:
        return [], {"all": ["invalid or incomplete convex hull"]}, {}
    actual_graph = _actual_graph(hull, point_ids)
    sizes = _size_signature(hull)
    edges = hull.edge_count
    degrees = hull.vertex_degrees
    found: list[str] = []
    reasons: dict[str, list[str]] = {}
    mappings: dict[str, list[dict[int, int]]] = {}
    for family, sig in EXPECTED_IDEAL_SIGNATURES.items():
        fails: list[str] = []
        if sizes != sig["faces"]:
            fails.append(f"polygon face sizes {sizes} do not match {sig['faces']}")
        if edges != sig["edges"]:
            fails.append(f"edge count {edges} does not match {sig['edges']}")
        if any(d != sig["degree"] for d in degrees):
            fails.append(f"vertex degrees {degrees} do not all equal {sig['degree']}")
        if hull.face_adjacency_counts != sig["face_adjacency"]:
            fails.append(f"face adjacency {hull.face_adjacency_counts} does not match {sig['face_adjacency']}")
        if bisecting_plane_count != sig["bisecting_planes"]:
            fails.append(f"qualifying central hexagon planes {bisecting_plane_count} do not match {sig['bisecting_planes']}")
        if hull.euler_characteristic != 2:
            fails.append(f"Euler characteristic {hull.euler_characteristic} is not 2")
        if len(hull.vertex_degrees) != 12:
            fails.append("outer hull does not contain 12 vertices")
        if not fails:
            mappings[family] = graph_isomorphisms(actual_graph, REFERENCE_GRAPHS[family])
            if not mappings[family]:
                fails.append("full vertex-edge graph is not isomorphic to the independently built reference graph")
        if not fails:
            found.append(family)
        reasons[family] = fails
    return found, reasons, mappings


def analyze_geometry(
    relative_points: np.ndarray,
    neighbor_ids: list[str],
    center_cartesian: Vector3,
    plane_settings: PlaneSettings = PlaneSettings(),
    hull_settings: HullSettings = HullSettings(),
    classification_settings: dict[str, float] | None = None,
) -> GeometryDiagnostics:
    points = np.asarray(relative_points, dtype=float)
    if points.shape != (12, 3):
        raise ValueError(f"Geometry classification requires exactly 12 points; got {points.shape}")
    settings = classification_settings or {
        "ideal_relative_spread": 1e-6,
        "accepted_radius_spread": 0.08,
        "accepted_edge_spread": 0.12,
        "accepted_face_planarity_relative": 0.02,
        "accepted_shape_rmsd": 0.08,
    }
    planes, length_scale = enumerate_center_planes(
        points, neighbor_ids, center_cartesian=center_cartesian, settings=plane_settings
    )
    hull = reconstruct_hull(
        points, neighbor_ids, characteristic_len=length_scale,
        center_cartesian=center_cartesian, settings=hull_settings,
    )
    bisecting_count = sum(p.is_bisecting_hexagon for p in planes)
    candidate_families, topology_failures, family_mappings = _candidate_families(
        bisecting_count, hull, neighbor_ids
    )
    radii = np.linalg.norm(points, axis=1)
    center_spread = float((radii.max() - radii.min()) / radii.mean())
    edge_lengths = np.asarray(hull.edge_lengths_angstrom, dtype=float)
    edge_spread = (float((edge_lengths.max() - edge_lengths.min()) / edge_lengths.mean())
                   if edge_lengths.size else None)
    planarity_rel = (float(hull.max_face_planarity_residual_angstrom / length_scale)
                     if hull.success and hull.max_face_planarity_residual_angstrom is not None else None)
    reasons: list[str] = []
    candidate_family: str | None = None
    final_classification: str | None = None
    quality = "unresolved"
    shape_rmsd: float | None = None
    if len(candidate_families) > 1:
        final_classification = None
        quality = "ambiguous"
        reasons.append(f"Topology signatures match multiple families: {candidate_families}.")
    elif len(candidate_families) == 1:
        candidate_family = candidate_families[0]
        shape_rmsd = _best_shape_rmsd(candidate_family, points, family_mappings[candidate_family])
        ideal = (
            center_spread <= settings["ideal_relative_spread"]
            and edge_spread is not None and edge_spread <= settings["ideal_relative_spread"]
            and planarity_rel is not None and planarity_rel <= settings["ideal_relative_spread"]
            and shape_rmsd is not None and shape_rmsd <= settings["ideal_relative_spread"]
        )
        accepted = (
            center_spread <= settings["accepted_radius_spread"]
            and edge_spread is not None and edge_spread <= settings["accepted_edge_spread"]
            and planarity_rel is not None
            and planarity_rel <= settings["accepted_face_planarity_relative"]
            and shape_rmsd is not None and shape_rmsd <= settings["accepted_shape_rmsd"]
        )
        if ideal:
            quality = "ideal_within_numerical_tolerance"
            final_classification = candidate_family
            reasons.append("Plane and full connectivity signatures match; metric residuals fall within the numerical ideal thresholds.")
        elif accepted:
            quality = "accepted_distorted_geometry"
            final_classification = candidate_family
            reasons.append("Plane and full connectivity signatures match; metric distortion falls within the configured acceptance thresholds.")
        else:
            quality = "strongly_distorted"
            final_classification = "other"
            reasons.append("Topology suggests a target family, but one or more configured metric distortion limits are exceeded.")
        reasons.extend(
            f"{family}: {text}" for family, failures in topology_failures.items() if family != candidate_family
            for text in failures[:1]
        )
    elif hull.success:
        final_classification = "other"
        quality = "non_target_geometry"
        reasons.append("No target family passed the plane, polygon, face-adjacency, degree, and graph-isomorphism checks.")
    else:
        final_classification = None
        quality = "analysis_error"
        reasons.append(f"Hull status is {hull.status}; this is an analysis failure, not a scientific geometry label.")

    if candidate_family is None and hull.success:
        reasons.extend(topology_failures.get("all", []))
    return GeometryDiagnostics(
        characteristic_length_angstrom=length_scale,
        total_distinct_center_planes=len(planes),
        exactly_six_neighbor_planes=sum(p.exactly_six for p in planes),
        convex_six_vertex_rings=sum(p.is_convex_hexagon and p.exactly_six for p in planes),
        qualifying_bisecting_hexagonal_planes=bisecting_count,
        planes=planes,
        hull=hull,
        center_distance_spread=center_spread,
        edge_length_spread=edge_spread,
        max_face_planarity_relative=planarity_rel,
        normalized_shape_rmsd=shape_rmsd,
        candidate_family=candidate_family,
        classification=final_classification,
        geometry_quality=quality,
        classification_reasons=reasons,
        matching_thresholds=settings,
    )
