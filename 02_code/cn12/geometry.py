"""Transparent rank/SVD plane enumeration and polygonal convex-hull analysis."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .models import FaceRecord, HullDiagnostics, PlaneRecord, Vector3


@dataclass(frozen=True)
class PlaneSettings:
    degeneracy_relative: float = 1e-10
    membership_relative: float = 1e-5
    normal_merge_radians: float = 1e-4


@dataclass(frozen=True)
class HullSettings:
    face_merge_relative: float = 1e-5
    face_normal_merge_radians: float = 1e-5
    duplicate_relative: float = 1e-10


@dataclass(frozen=True)
class _PlaneCandidate:
    normal: np.ndarray
    seed_indices: tuple[int, ...]
    singular_values: tuple[float, float, float]
    max_residual: float
    rms_residual: float


# The 924 ways to choose six of the twelve neighbors, in itertools order.
_SIX_OF_TWELVE = np.asarray(list(combinations(range(12), 6)), dtype=int)


def characteristic_length(points: np.ndarray) -> float:
    """Use median center-neighbor radius; never normalize each point separately."""
    radii = np.linalg.norm(np.asarray(points, dtype=float), axis=1)
    positive = radii[radii > 0]
    if positive.size == 0:
        raise ValueError("All neighbor positions coincide with the center.")
    return float(np.median(positive))


def _canonical_normal(normal: np.ndarray) -> np.ndarray:
    n = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    pivot = int(np.argmax(np.abs(n)))
    return -n if n[pivot] < 0 else n


def _angle_between_planes(a: np.ndarray, b: np.ndarray) -> float:
    """Sign-invariant angle in radians."""
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(np.arccos(dot))


def signed_plane_distances(
    points: np.ndarray, normal: np.ndarray, offset: float
) -> np.ndarray:
    """Evaluate signed point-to-plane distances for ``n dot x + d = 0``."""
    n = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(n))
    if norm == 0:
        raise ValueError("Plane normal must be nonzero")
    return (np.asarray(points, dtype=float) @ (n / norm) + float(offset) / norm)


def _fit_center_plane(points: np.ndarray, degeneracy_abs: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit a plane constrained through the origin using a 6x3 SVD."""
    _, singular_values, vt = np.linalg.svd(points, full_matrices=True)
    padded = np.zeros(3, dtype=float)
    padded[: min(3, singular_values.size)] = singular_values[:3]
    if padded[1] <= degeneracy_abs:
        raise ValueError("candidate points are coincident or collinear")
    normal = _canonical_normal(vt[-1])
    return normal, padded


def _project_hexagon(
    points: np.ndarray,
    member_indices: list[int],
    length_scale: float,
) -> tuple[bool, bool, list[int]]:
    """Return convex-six-ring, origin-strictly-inside, and cyclic input indices."""
    if len(member_indices) != 6:
        return False, False, []
    rows = points[member_indices]
    _, _, vt = np.linalg.svd(rows, full_matrices=True)
    basis = vt[:2].T
    projected = rows @ basis
    # A 2D hull must retain all six samples: this rejects interior and duplicate points.
    try:
        ring_local = ConvexHull(projected).vertices.tolist()
    except QhullError:
        return False, False, []
    if len(ring_local) != 6:
        return False, False, []
    ring_xy = projected[ring_local]
    crosses = []
    for i in range(6):
        a = ring_xy[i]
        b = ring_xy[(i + 1) % 6]
        crosses.append(float(a[0] * b[1] - a[1] * b[0]))
    area2 = float(sum(crosses))
    tol = max(np.finfo(float).eps * length_scale**2 * 100, length_scale**2 * 1e-12)
    same_sign = all(x > tol for x in crosses) or all(x < -tol for x in crosses)
    inside = same_sign and abs(area2) > tol
    return True, inside, [member_indices[i] for i in ring_local]


def enumerate_center_planes(
    relative_points: np.ndarray,
    neighbor_ids: list[str],
    center_cartesian: Vector3 = (0.0, 0.0, 0.0),
    settings: PlaneSettings = PlaneSettings(),
) -> tuple[list[PlaneRecord], float]:
    """Enumerate the 924 six-position subsets for a twelve-position environment.

    The least-squares plane is constrained to pass through the actual center. A
    geometric membership tolerance is applied only after the strict degeneracy test.
    Candidate planes are grouped with sign-invariant complete-link angular checks,
    which prevents chained, input-order-dependent plane merges.
    """
    points = np.asarray(relative_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(neighbor_ids):
        raise ValueError("relative_points must be an N x 3 matrix matching neighbor_ids")
    if len(points) != 12:
        raise ValueError(f"Plane enumeration requires exactly 12 positions; got {len(points)}")
    length_scale = characteristic_length(points)
    membership_abs = settings.membership_relative * length_scale
    degeneracy_abs = settings.degeneracy_relative * length_scale

    # Screen all 924 subsets in one stacked SVD. The smallest singular value of a
    # six-vector subset equals the root-sum-square of its plane residuals, so a
    # subset that passes the exact test below must have it <= sqrt(6) * tolerance.
    # The screen is 1000 times looser than that bound; the loop then refits only
    # the few near-coplanar survivors, and acceptance is decided there as before.
    subsets = _SIX_OF_TWELVE
    smallest = np.linalg.svd(points[subsets], compute_uv=False)[:, -1]
    screen = 1e3 * np.sqrt(6.0) * membership_abs + 1e-9 * length_scale
    survivors = np.flatnonzero(smallest <= screen)

    candidates: list[_PlaneCandidate] = []
    for row in survivors:
        ids = tuple(int(i) for i in subsets[row])
        selected = points[list(ids)]
        try:
            normal, singular = _fit_center_plane(selected, degeneracy_abs)
        except ValueError:
            continue
        residuals = selected @ normal
        max_residual = float(np.max(np.abs(residuals)))
        if max_residual > membership_abs:
            continue
        candidates.append(
            _PlaneCandidate(
                normal=normal,
                seed_indices=ids,
                singular_values=tuple(float(x) for x in singular),
                max_residual=max_residual,
                rms_residual=float(np.sqrt(np.mean(residuals**2))),
            )
        )

    # Canonical geometric ordering plus complete-link compatibility makes grouping
    # independent of itertools' input order and blocks transitive chaining.
    candidates.sort(
        key=lambda c: tuple(float(x) for x in np.round(_canonical_normal(c.normal), 14))
    )
    clusters: list[list[_PlaneCandidate]] = []
    for candidate in candidates:
        compatible = [
            i
            for i, cluster in enumerate(clusters)
            if all(
                _angle_between_planes(candidate.normal, existing.normal)
                <= settings.normal_merge_radians
                for existing in cluster
            )
        ]
        if not compatible:
            clusters.append([candidate])
            continue
        # If two complete-link clusters both qualify, attach to the nearest one.
        best = min(
            compatible,
            key=lambda i: (
                max(_angle_between_planes(candidate.normal, x.normal) for x in clusters[i]),
                i,
            ),
        )
        clusters[best].append(candidate)

    records: list[PlaneRecord] = []
    center = np.asarray(center_cartesian, dtype=float)
    for cluster in clusters:
        seed_union = sorted({j for c in cluster for j in c.seed_indices})
        try:
            normal, singular = _fit_center_plane(points[seed_union], degeneracy_abs)
        except ValueError:
            continue
        signed = points @ normal
        member_indices = [i for i, d in enumerate(signed) if abs(float(d)) <= membership_abs]
        if len(member_indices) < 6:
            continue
        # Refit all members and recheck once, as required after candidate merging.
        normal, singular = _fit_center_plane(points[member_indices], degeneracy_abs)
        signed = points @ normal
        member_indices = [i for i, d in enumerate(signed) if abs(float(d)) <= membership_abs]
        if len(member_indices) < 6:
            continue
        selected_residuals = signed[member_indices]
        is_six_ring, center_inside, ring = _project_hexagon(points, member_indices, length_scale)
        on = len(member_indices)
        positive = sum(float(signed[i]) > membership_abs for i in range(12))
        negative = sum(float(signed[i]) < -membership_abs for i in range(12))
        side_counts = {"positive": positive, "negative": negative, "on_plane": on}
        reasons: list[str] = []
        if on != 6:
            reasons.append(f"plane contains {on} neighbors, not exactly six")
        if not is_six_ring:
            reasons.append("the six projected points do not form a convex six-vertex boundary")
        if is_six_ring and not center_inside:
            reasons.append("the projected center is not strictly inside the ring")
        if is_six_ring and on == 6 and (positive != 3 or negative != 3):
            reasons.append(f"remaining neighbors split {positive}:{negative}, not 3:3")
        is_bisecting = on == 6 and is_six_ring and center_inside and positive == 3 and negative == 3
        d_original = -float(np.dot(normal, center))
        records.append(
            PlaneRecord(
                plane_id="",
                normal=tuple(float(x) for x in normal),
                offset_angstrom=d_original,
                relative_offset_angstrom=0.0,
                neighbor_ids=list(neighbor_ids),
                signed_distances_angstrom=[float(x) for x in signed],
                member_ids=[neighbor_ids[i] for i in member_indices],
                member_count=on,
                exactly_six=(on == 6),
                is_convex_hexagon=is_six_ring,
                ring_order_ids=[neighbor_ids[i] for i in ring],
                projected_center_inside=center_inside,
                side_counts=side_counts,
                is_bisecting_hexagon=is_bisecting,
                singular_values=tuple(float(x) for x in singular),
                max_fit_residual_angstrom=float(np.max(np.abs(selected_residuals))),
                rms_fit_residual_angstrom=float(np.sqrt(np.mean(selected_residuals**2))),
                rejection_reasons=reasons,
            )
        )

    records.sort(key=lambda p: (tuple(np.round(p.normal, 12)), tuple(sorted(p.member_ids))))
    for i, record in enumerate(records, 1):
        record.plane_id = f"P{i:02d}"
    return records, length_scale


def _face_category(size: int) -> str:
    return {3: "triangle", 4: "quadrilateral", 5: "pentagon", 6: "hexagon"}.get(
        size, f"{size}-gon"
    )


def reconstruct_hull(
    relative_points: np.ndarray,
    point_ids: list[str],
    characteristic_len: float,
    center_cartesian: Vector3 = (0.0, 0.0, 0.0),
    settings: HullSettings = HullSettings(),
) -> HullDiagnostics:
    """Build a convex hull and merge coplanar SciPy triangles into polygon faces."""
    points = np.asarray(relative_points, dtype=float)
    n_points = len(points)
    if points.shape != (n_points, 3) or len(point_ids) != n_points:
        raise ValueError("Hull points and IDs must have matching N x 3 shapes")
    merge_dist = settings.face_merge_relative * characteristic_len
    duplicate_dist = settings.duplicate_relative * characteristic_len
    warnings: list[str] = []
    if n_points < 4:
        return HullDiagnostics(False, "insufficient_points", 0, 0, 0, {}, [], {}, None, None,
                               None, False, None, [], [], [], ["fewer than four outer points"])
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    if float(np.min(distances)) <= duplicate_dist:
        return HullDiagnostics(False, "duplicate_vertices", 0, 0, 0, {}, [], {}, None, None,
                               None, False, None, [], [], [], ["duplicate or coincident neighbor positions"])
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=duplicate_dist) < 3:
        return HullDiagnostics(False, "zero_volume", 0, 0, 0, {}, [], {}, None, None,
                               None, False, None, [], [], [], ["outer positions are not three-dimensional"])
    try:
        hull = ConvexHull(points)
    except QhullError as exc:
        return HullDiagnostics(False, "hull_error", 0, 0, 0, {}, [], {}, None, None,
                               None, False, None, [], [], [], [str(exc).splitlines()[0]])

    # Normalize facet planes. Qhull outward normals for coplanar triangles on one
    # polygon point in the same direction. Opposite faces remain distinct because
    # their signed offsets differ after consistent normalization.
    equations: list[tuple[np.ndarray, float]] = []
    for eq in hull.equations:
        scale = float(np.linalg.norm(eq[:3]))
        equations.append((np.asarray(eq[:3] / scale), float(eq[3] / scale)))
    simplices = [tuple(sorted(int(v) for v in tri)) for tri in hull.simplices]
    order = sorted(range(len(simplices)), key=lambda i: (equations[i][0].round(12).tolist(),
                                                         round(equations[i][1], 12), simplices[i]))
    compatible = np.zeros((len(simplices), len(simplices)), dtype=bool)
    score = np.full((len(simplices), len(simplices)), np.inf, dtype=float)
    for a in range(len(simplices)):
        compatible[a, a] = True
        score[a, a] = 0.0
        na, da = equations[a]
        for b in range(a + 1, len(simplices)):
            nb, db = equations[b]
            angle = float(np.arccos(np.clip(np.dot(na, nb), -1.0, 1.0)))
            offset = abs(da - db)
            okay = angle <= settings.face_normal_merge_radians and offset <= merge_dist
            compatible[a, b] = compatible[b, a] = okay
            if okay:
                angle_term = angle / max(settings.face_normal_merge_radians, 1e-15)
                distance_term = offset / max(merge_dist, 1e-15)
                score[a, b] = score[b, a] = max(angle_term, distance_term)

    clusters = [[i] for i in order]
    # Plain lists make the many single-element lookups below much faster.
    compatible = compatible.tolist()
    score = score.tolist()
    # Deterministic complete-link agglomeration: every cross-pair must match.
    while True:
        options: list[tuple[float, tuple[int, ...], int, int]] = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if all(compatible[a][b] for a in clusters[i] for b in clusters[j]):
                    cross = [score[a][b] for a in clusters[i] for b in clusters[j]]
                    options.append((max(cross), tuple(clusters[i]), i, j))
        if not options:
            break
        _, _, i, j = min(options)
        clusters[i] = sorted(clusters[i] + clusters[j])
        del clusters[j]

    faces: list[FaceRecord] = []
    center = np.asarray(center_cartesian, dtype=float)
    for cluster in clusters:
        vertex_set = sorted({v for facet in cluster for v in simplices[facet]})
        if len(vertex_set) < 3:
            continue
        face_points = points[vertex_set]
        centroid = face_points.mean(axis=0)
        centered = face_points - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=True)
        normal = vt[-1]
        normal /= np.linalg.norm(normal)
        offset_rel = -float(np.dot(normal, centroid))
        residuals = face_points @ normal + offset_rel
        max_resid = float(np.max(np.abs(residuals)))
        if max_resid > merge_dist:
            warnings.append(
                f"Rejected a merged hull face whose full-boundary residual {max_resid:.6g} A "
                f"exceeded {merge_dist:.6g} A."
            )
            # Keep original triangles instead of accepting a curved approximate face.
            for facet in cluster:
                tri = list(simplices[facet])
                tri_points = points[tri]
                ctr = tri_points.mean(axis=0)
                _, _, vt_tri = np.linalg.svd(tri_points - ctr, full_matrices=True)
                n_tri = vt_tri[-1]
                d_tri = -float(np.dot(n_tri, ctr))
                tri_resid = np.abs(tri_points @ n_tri + d_tri)
                projected = tri_points @ vt_tri[:2].T
                tri_order = [tri[k] for k in ConvexHull(projected).vertices]
                tri_edges = [tuple(sorted((point_ids[tri_order[k]], point_ids[tri_order[(k+1)%3]])))
                             for k in range(3)]
                d_original = d_tri - float(np.dot(n_tri, center))
                faces.append(FaceRecord("", tri_order, [point_ids[k] for k in tri_order], 3,
                                        tri_edges, tuple(float(x) for x in n_tri), d_original,
                                        float(np.max(tri_resid)), float(np.sqrt(np.mean(tri_resid**2))),
                                        [float(np.linalg.norm(points[tri_order[k]]-points[tri_order[(k+1)%3]]))
                                         for k in range(3)]))
            continue
        basis = vt[:2].T
        projected = face_points @ basis
        try:
            boundary_local = ConvexHull(projected).vertices.tolist()
        except QhullError:
            warnings.append("A merged face boundary could not be ordered in its plane.")
            continue
        boundary = [vertex_set[i] for i in boundary_local]
        boundary_edges = [
            tuple(sorted((point_ids[boundary[i]], point_ids[boundary[(i + 1) % len(boundary)]])))
            for i in range(len(boundary))
        ]
        edge_lengths = [
            float(np.linalg.norm(points[boundary[i]] - points[boundary[(i + 1) % len(boundary)]]))
            for i in range(len(boundary))
        ]
        offset_original = offset_rel - float(np.dot(normal, center))
        faces.append(
            FaceRecord(
                face_id="",
                vertex_indices=boundary,
                vertex_ids=[point_ids[i] for i in boundary],
                polygon_size=len(boundary),
                boundary_edges=boundary_edges,
                normal=tuple(float(x) for x in normal),
                offset_angstrom=offset_original,
                max_planarity_residual_angstrom=max_resid,
                rms_planarity_residual_angstrom=float(np.sqrt(np.mean(residuals**2))),
                edge_lengths_angstrom=edge_lengths,
            )
        )

    faces.sort(key=lambda f: (f.polygon_size, tuple(sorted(f.vertex_ids))))
    for i, face in enumerate(faces, 1):
        face.face_id = f"F{i:02d}"

    edge_to_faces: dict[tuple[str, str], list[int]] = {}
    edge_lengths_by_key: dict[tuple[str, str], float] = {}
    for face_i, face in enumerate(faces):
        for edge, length in zip(face.boundary_edges, face.edge_lengths_angstrom):
            edge_to_faces.setdefault(edge, []).append(face_i)
            edge_lengths_by_key[edge] = length
    edges = sorted(edge_to_faces)
    degrees = {key: 0 for key in point_ids}
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    adjacency: dict[str, int] = {}
    for attached in edge_to_faces.values():
        if len(attached) == 2:
            cats = sorted((_face_category(faces[attached[0]].polygon_size),
                           _face_category(faces[attached[1]].polygon_size)))
            key = "-".join(cats)
            adjacency[key] = adjacency.get(key, 0) + 1
        else:
            warnings.append(f"Hull boundary edge belongs to {len(attached)} polygon faces, not two.")
    face_size_counts: dict[str, int] = {}
    for face in faces:
        key = str(face.polygon_size)
        face_size_counts[key] = face_size_counts.get(key, 0) + 1
    all_hull_vertices = set(int(i) for i in hull.vertices) == set(range(n_points))
    if not all_hull_vertices:
        warnings.append("One or more selected neighbors lie inside the outer convex hull.")
    euler = len(hull.vertices) - len(edges) + len(faces)
    center_inside = bool(np.all(hull.equations[:, :3] @ np.zeros(3) + hull.equations[:, 3] <= merge_dist))
    return HullDiagnostics(
        success=True,
        status="ok",
        vertex_count=len(hull.vertices),
        edge_count=len(edges),
        face_count=len(faces),
        face_size_counts=face_size_counts,
        vertex_degrees=[degrees[key] for key in point_ids],
        face_adjacency_counts=adjacency,
        euler_characteristic=int(euler),
        center_inside=center_inside,
        hull_volume_angstrom3=float(hull.volume),
        all_points_are_hull_vertices=all_hull_vertices,
        max_face_planarity_residual_angstrom=max(
            (f.max_planarity_residual_angstrom for f in faces), default=0.0
        ),
        edge_lengths_angstrom=[edge_lengths_by_key[e] for e in edges],
        edges=edges,
        faces=faces,
        warnings=warnings,
    )
