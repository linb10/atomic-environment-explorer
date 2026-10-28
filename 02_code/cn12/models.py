"""Named records shared by the CN12 parser, geometry, and export layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class AtomComponent:
    source_id: str
    label: str
    element: str
    occupancy: float
    disorder_group: str | None = None


@dataclass(frozen=True)
class AtomSite:
    source_id: str
    label: str
    element: str
    fractional: Vector3
    occupancy: float
    disorder_group: str | None = None


@dataclass(frozen=True)
class PhysicalSite:
    """One occupied physical position in the unit cell, including mixed species."""

    position_id: str
    fractional: Vector3
    components: tuple[AtomComponent, ...]
    symmetry_sources: tuple[str, ...]
    incompatible_overlap: bool = False


@dataclass(frozen=True)
class CenterSite:
    site_id: str
    display_label: str
    position_id: str
    fractional: Vector3
    components: tuple[AtomComponent, ...]
    source_ids: tuple[str, ...]
    symmetry_multiplicity: int


@dataclass
class CrystalBlock:
    source_path: str
    block_name: str
    formula: str
    space_group: str
    space_group_number: int | None
    cell_lengths_angstrom: Vector3
    cell_angles_degrees: Vector3
    lattice_columns_angstrom: list[list[float]]
    atom_sites: list[AtomSite]
    physical_sites: list[PhysicalSite]
    center_sites: list[CenterSite]
    parser: str
    parser_version: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Neighbor:
    neighbor_id: str
    position_id: str
    source_labels: tuple[str, ...]
    species_occupancies: tuple[tuple[str, float], ...]
    symmetry_sources: tuple[str, ...]
    image_translation: tuple[int, int, int]
    fractional: Vector3
    cartesian_angstrom: Vector3
    relative_angstrom: Vector3
    distance_angstrom: float
    mixed_occupancy: bool
    incompatible_overlap: bool


@dataclass
class SelectionDiagnostics:
    selected_count: int
    selection_mode: str
    status: str
    shell_distance_angstrom: float | None
    d12_angstrom: float | None
    d13_angstrom: float | None
    boundary_tie_count: int
    d12_shell_count: int
    boundary_shell_neighbor_ids: list[str]
    complete_search_radius_angstrom: float
    candidate_count: int
    cn_estimate: int | None
    cn_estimate_method: str
    rationale: str


@dataclass
class PlaneRecord:
    plane_id: str
    normal: Vector3
    offset_angstrom: float
    relative_offset_angstrom: float
    neighbor_ids: list[str]
    signed_distances_angstrom: list[float]
    member_ids: list[str]
    member_count: int
    exactly_six: bool
    is_convex_hexagon: bool
    ring_order_ids: list[str]
    projected_center_inside: bool
    side_counts: dict[str, int]
    is_bisecting_hexagon: bool
    singular_values: tuple[float, float, float]
    max_fit_residual_angstrom: float
    rms_fit_residual_angstrom: float
    rejection_reasons: list[str]


@dataclass
class FaceRecord:
    face_id: str
    vertex_indices: list[int]
    vertex_ids: list[str]
    polygon_size: int
    boundary_edges: list[tuple[str, str]]
    normal: Vector3
    offset_angstrom: float
    max_planarity_residual_angstrom: float
    rms_planarity_residual_angstrom: float
    edge_lengths_angstrom: list[float]


@dataclass
class HullDiagnostics:
    success: bool
    status: str
    vertex_count: int
    edge_count: int
    face_count: int
    face_size_counts: dict[str, int]
    vertex_degrees: list[int]
    face_adjacency_counts: dict[str, int]
    euler_characteristic: int | None
    center_inside: bool | None
    hull_volume_angstrom3: float | None
    all_points_are_hull_vertices: bool
    max_face_planarity_residual_angstrom: float | None
    edge_lengths_angstrom: list[float]
    edges: list[tuple[str, str]]
    faces: list[FaceRecord]
    warnings: list[str] = field(default_factory=list)


@dataclass
class GeometryDiagnostics:
    characteristic_length_angstrom: float
    total_distinct_center_planes: int
    exactly_six_neighbor_planes: int
    convex_six_vertex_rings: int
    qualifying_bisecting_hexagonal_planes: int
    planes: list[PlaneRecord]
    hull: HullDiagnostics
    center_distance_spread: float | None
    edge_length_spread: float | None
    max_face_planarity_relative: float | None
    normalized_shape_rmsd: float | None
    candidate_family: str | None
    classification: str | None
    geometry_quality: str
    classification_reasons: list[str]
    matching_thresholds: dict[str, float]


@dataclass
class AnalysisResult:
    status: str
    source_path: str
    source_sha256: str
    block_name: str
    center_site_id: str
    center_label: str
    center_fractional: Vector3
    center_cartesian_angstrom: Vector3
    center_species_occupancies: tuple[tuple[str, float], ...]
    neighbors: list[Neighbor]
    selection: SelectionDiagnostics
    geometry: GeometryDiagnostics | None
    parser: str
    parser_version: str
    settings: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def to_jsonable(value: Any) -> Any:
    """Convert nested dataclasses/tuples into standard JSON data."""
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    return value
