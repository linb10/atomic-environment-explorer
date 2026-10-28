"""Single-site and all-site analysis entry points."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import importlib.metadata as metadata
from typing import Iterable

import numpy as np

from .classification import analyze_geometry
from .crystal import (
    extract_nearest_neighbors,
    normalized_text_sha256,
    parse_cif_blocks,
    select_block,
    sha256_file,
)
from .geometry import HullSettings, PlaneSettings
from .models import AnalysisResult, CrystalBlock


@dataclass(frozen=True)
class AnalysisSettings:
    neighbor_count: int = 12
    candidate_target: int = 16
    plane_degeneracy_relative: float = 1e-10
    plane_membership_relative: float = 1e-5
    plane_normal_merge_radians: float = 1e-4
    face_merge_relative: float = 1e-5
    face_normal_merge_radians: float = 1e-5
    neighbor_tie_relative: float = 1e-5
    max_translation_shell: int = 80
    accepted_radius_spread: float = 0.08
    accepted_edge_spread: float = 0.12
    accepted_face_planarity_relative: float = 0.02
    accepted_shape_rmsd: float = 0.08
    ideal_relative_spread: float = 1e-6

    def as_dict(self, characteristic_length_angstrom: float | None = None) -> dict:
        values = asdict(self)
        values["tolerance_units"] = {
            "plane_degeneracy_relative": "dimensionless length / L",
            "plane_membership_relative": "dimensionless length / L",
            "plane_membership_effective_angstrom": (
                None if characteristic_length_angstrom is None else
                self.plane_membership_relative * characteristic_length_angstrom
            ),
            "face_merge_relative": "dimensionless length / L",
            "face_merge_effective_angstrom": (
                None if characteristic_length_angstrom is None else
                self.face_merge_relative * characteristic_length_angstrom
            ),
            "plane_normal_merge_radians": "radians; equivalent transverse distance is L*sin(angle)",
            "plane_normal_merge_effective_angstrom": (
                None if characteristic_length_angstrom is None else
                characteristic_length_angstrom * np.sin(self.plane_normal_merge_radians)
            ),
            "face_normal_merge_radians": "radians; equivalent transverse distance is L*sin(angle)",
            "face_normal_merge_effective_angstrom": (
                None if characteristic_length_angstrom is None else
                characteristic_length_angstrom * np.sin(self.face_normal_merge_radians)
            ),
            "neighbor_tie_relative": "dimensionless distance / median of first 16 neighbors",
        }
        return values


def _package_versions() -> dict[str, str | None]:
    packages = ["numpy", "scipy", "gemmi", "cifkit", "pyvista", "trame",
                "trame-vtk", "trame-vuetify", "ipywidgets", "ipykernel",
                "nbformat", "nbclient", "pandas", "matplotlib"]
    out = {}
    for name in packages:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def analyze_parsed_block(
    block: CrystalBlock,
    source_path: str | Path,
    center_site: str | None = None,
    settings: AnalysisSettings = AnalysisSettings(),
) -> AnalysisResult:
    center, neighbors, selection = extract_nearest_neighbors(
        block,
        center_site,
        count=settings.neighbor_count,
        candidate_target=max(settings.candidate_target, settings.neighbor_count + 1),
        tie_relative_tolerance=settings.neighbor_tie_relative,
        max_translation_shell=settings.max_translation_shell,
    )
    lattice = np.asarray(block.lattice_columns_angstrom, dtype=float)
    center_cart = lattice @ np.asarray(center.fractional, dtype=float)
    warnings = list(block.warnings)
    warnings.extend(
        f"Physical position {n.position_id} has overlapping occupancies summing above 1."
        for n in neighbors if n.incompatible_overlap
    )
    geometry = None
    if selection.status == "unambiguous" and len(neighbors) == settings.neighbor_count == 12:
        relative = np.asarray([n.relative_angstrom for n in neighbors], dtype=float)
        plane_cfg = PlaneSettings(
            degeneracy_relative=settings.plane_degeneracy_relative,
            membership_relative=settings.plane_membership_relative,
            normal_merge_radians=settings.plane_normal_merge_radians,
        )
        hull_cfg = HullSettings(
            face_merge_relative=settings.face_merge_relative,
            face_normal_merge_radians=settings.face_normal_merge_radians,
        )
        classification_cfg = {
            "ideal_relative_spread": settings.ideal_relative_spread,
            "accepted_radius_spread": settings.accepted_radius_spread,
            "accepted_edge_spread": settings.accepted_edge_spread,
            "accepted_face_planarity_relative": settings.accepted_face_planarity_relative,
            "accepted_shape_rmsd": settings.accepted_shape_rmsd,
        }
        geometry = analyze_geometry(
            relative,
            [n.neighbor_id for n in neighbors],
            tuple(float(x) for x in center_cart),
            plane_settings=plane_cfg,
            hull_settings=hull_cfg,
            classification_settings=classification_cfg,
        )
        warnings.extend(geometry.hull.warnings)
        if geometry.total_distinct_center_planes == 0:
            warnings.append("Zero qualifying center-containing planes is a valid geometric result.")
    elif selection.status == "boundary_tie":
        warnings.append("Geometry classification is unset because the d12/d13 shell is tied.")
    elif selection.status == "unsupported_input":
        warnings.append("Geometry classification is unset because occupancy metadata is incompatible.")
    else:
        warnings.append("Geometry classification is unset because the periodic neighbor search is incomplete.")
    status = "complete" if geometry is not None else selection.status
    return AnalysisResult(
        status=status,
        source_path=str(Path(source_path).resolve()),
        source_sha256=sha256_file(source_path),
        block_name=block.block_name,
        center_site_id=center.site_id,
        center_label=center.display_label,
        center_fractional=center.fractional,
        center_cartesian_angstrom=tuple(float(x) for x in center_cart),
        center_species_occupancies=tuple(sorted(
            (component.element, float(component.occupancy)) for component in center.components
        )),
        neighbors=neighbors,
        selection=selection,
        geometry=geometry,
        parser=block.parser,
        parser_version=block.parser_version,
        settings=settings.as_dict(
            geometry.characteristic_length_angstrom if geometry is not None else None
        ),
        warnings=warnings,
    )


def analyze_file(
    file_path: str | Path,
    center_site: str | None = None,
    block_name: str | None = None,
    settings: AnalysisSettings = AnalysisSettings(),
) -> AnalysisResult:
    """Callable single-file/site API used by the notebook and CLI."""
    blocks = parse_cif_blocks(file_path)
    block = select_block(blocks, block_name)
    return analyze_parsed_block(block, file_path, center_site, settings)


def analyze_all_sites(
    file_path: str | Path,
    block_name: str | None = None,
    settings: AnalysisSettings = AnalysisSettings(),
) -> list[AnalysisResult]:
    """Analyze every distinct center site in a CIF without naming any of them.

    The file is parsed once. With ``block_name`` left as None, every data block
    is included, so a multi-block file needs no extra input either.
    """
    blocks = parse_cif_blocks(file_path)
    if block_name is not None:
        blocks = [select_block(blocks, block_name)]
    return [
        analyze_parsed_block(block, file_path, center.site_id, settings)
        for block in blocks
        for center in block.center_sites
    ]


def describe_result(result: AnalysisResult) -> dict:
    """The handful of values that answer "what is this site?" at a glance."""
    geometry = result.geometry
    return {
        "file": Path(result.source_path).name,
        "block": result.block_name,
        "center": result.center_label,
        "status": result.status,
        "classification": geometry.classification if geometry else None,
        "quality": geometry.geometry_quality if geometry else None,
        "hexagonal_planes": geometry.qualifying_bisecting_hexagonal_planes if geometry else None,
        "faces": _face_text(geometry.hull.face_size_counts) if geometry else None,
        "d12_angstrom": result.selection.d12_angstrom,
        "d13_angstrom": result.selection.d13_angstrom,
    }


def plain_verdict(result: AnalysisResult) -> str:
    """One sentence a reader can act on, including why a label was withheld."""
    geometry = result.geometry
    selection = result.selection
    if geometry is None:
        if result.status == "boundary_tie":
            return (f"not classified: the 12th and 13th neighbors are tied at "
                    f"{selection.d12_angstrom:.4f} A ({selection.d12_shell_count} positions in that shell), "
                    "so there is no unique set of 12")
        if result.status == "unsupported_input":
            return "not classified: a neighbor position has overlapping occupancies above 1"
        return "not classified: fewer than 13 neighbor positions were found"
    if geometry.geometry_quality == "ideal_within_numerical_tolerance":
        return f"{geometry.classification} (ideal)"
    if geometry.geometry_quality == "accepted_distorted_geometry":
        return f"{geometry.classification} (slightly distorted)"
    if geometry.geometry_quality == "strongly_distorted":
        return f"other ({geometry.candidate_family} connectivity, but too distorted to accept)"
    if geometry.geometry_quality == "ambiguous":
        return "not classified: the connectivity matches more than one family"
    if geometry.geometry_quality == "analysis_error":
        return f"not classified: the convex hull could not be built ({geometry.hull.status})"
    return "other (none of the four CN12 families)"


def _face_text(face_size_counts: dict[str, int]) -> str:
    names = {"3": "triangles", "4": "quadrilaterals", "5": "pentagons", "6": "hexagons"}
    parts = [f"{count} {names.get(str(size), f'{size}-gons')}"
             for size, count in sorted(face_size_counts.items(), key=lambda item: int(item[0]))]
    return ", ".join(parts)


def analyze_batch(
    paths: Iterable[str | Path],
    settings: AnalysisSettings = AnalysisSettings(),
) -> dict:
    """Analyze every representative center in every CIF; retain all failures."""
    file_rows: list[dict] = []
    site_rows: list[dict] = []
    normalized_hashes: dict[str, list[str]] = {}
    for item in sorted({str(Path(p).resolve()) for p in paths}):
        path = Path(item)
        try:
            normalized_hash = normalized_text_sha256(path)
            normalized_hashes.setdefault(normalized_hash, []).append(str(path))
            blocks = parse_cif_blocks(path)
        except Exception as exc:
            file_rows.append({"path": str(path), "status": "parse_error", "error": str(exc)})
            continue
        file_rows.append({
            "path": str(path), "status": "parsed", "normalized_sha256": normalized_hash,
            "block_count": len(blocks), "center_representative_count": sum(len(b.center_sites) for b in blocks),
        })
        for block in blocks:
            for center in block.center_sites:
                try:
                    result = analyze_parsed_block(block, path, center.site_id, settings)
                    geometry = result.geometry
                    site_rows.append({
                        "path": str(path),
                        "normalized_sha256": normalized_hash,
                        "block_name": block.block_name,
                        "center_site_id": center.site_id,
                        "center_label": center.display_label,
                        "symmetry_multiplicity": center.symmetry_multiplicity,
                        "formula": block.formula,
                        "space_group": block.space_group,
                        "selection_status": result.selection.status,
                        "selected_count": result.selection.selected_count,
                        "d12_angstrom": result.selection.d12_angstrom,
                        "d13_angstrom": result.selection.d13_angstrom,
                        "cn_estimate": result.selection.cn_estimate,
                        "plane_count": (geometry.qualifying_bisecting_hexagonal_planes if geometry else None),
                        "all_center_planes": (geometry.total_distinct_center_planes if geometry else None),
                        "face_size_counts": (geometry.hull.face_size_counts if geometry else None),
                        "edge_count": (geometry.hull.edge_count if geometry else None),
                        "candidate_family": (geometry.candidate_family if geometry else None),
                        "classification": (geometry.classification if geometry else None),
                        "geometry_quality": (geometry.geometry_quality if geometry else None),
                        "status": result.status,
                        "warnings": result.warnings,
                    })
                except Exception as exc:
                    site_rows.append({
                        "path": str(path), "normalized_sha256": normalized_hash,
                        "block_name": block.block_name, "center_site_id": center.site_id,
                        "center_label": center.display_label, "selection_status": "analysis_error",
                        "status": "analysis_error", "error": str(exc),
                    })
    unique_site_keys = {
        (row.get("normalized_sha256"), row.get("block_name"), row.get("center_site_id"))
        for row in site_rows
    }
    parse_error_count = sum(row["status"] != "parsed" for row in file_rows)
    return {
        "files_seen": len(file_rows),
        "files_parsed": len(file_rows) - parse_error_count,
        "file_parse_errors": parse_error_count,
        "unique_structures_after_newline_normalization": len(normalized_hashes),
        "duplicate_provenance": [paths for paths in normalized_hashes.values() if len(paths) > 1],
        "site_results": len(site_rows),
        "unique_representative_sites": len(unique_site_keys),
        "file_rows": file_rows,
        "site_rows": site_rows,
        "package_versions": _package_versions(),
    }
