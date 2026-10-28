"""Reproducible, non-overwriting CSV/JSON and viewer exports."""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re
import uuid

from . import __version__
from .models import AnalysisResult, to_jsonable
from .pipeline import _package_versions


def _slug(text: str, limit: int = 32) -> str:
    """Folder-name-safe version of a file stem or site label."""
    cleaned = re.sub(r"[^A-Za-z0-9+]+", "-", str(text)).strip("-")
    return cleaned[:limit] or "x"


def _run_id(prefix: str = "run", *labels: str) -> str:
    """Unique folder name: prefix, UTC time, readable labels, random suffix."""
    parts = [prefix, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")]
    parts.extend(_slug(label) for label in labels if label)
    parts.append(uuid.uuid4().hex[:8])
    return "_".join(parts)


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _source_hashes() -> dict[str, str]:
    from hashlib import sha256

    package_dir = Path(__file__).parent
    values = {}
    for path in sorted(package_dir.glob("*.py")):
        values[path.name] = sha256(path.read_bytes()).hexdigest()
    return values


def _rows_for_result(result: AnalysisResult) -> dict[str, list[dict]]:
    summary = [{
        "status": result.status,
        "source_path": result.source_path,
        "source_sha256": result.source_sha256,
        "block_name": result.block_name,
        "center_site_id": result.center_site_id,
        "center_label": result.center_label,
        "center_fractional": result.center_fractional,
        "center_cartesian_angstrom": result.center_cartesian_angstrom,
        "center_species_occupancies": result.center_species_occupancies,
        "selected_count": result.selection.selected_count,
        "selection_mode": result.selection.selection_mode,
        "selection_status": result.selection.status,
        "d12_angstrom": result.selection.d12_angstrom,
        "d13_angstrom": result.selection.d13_angstrom,
        "d12_shell_count": result.selection.d12_shell_count,
        "boundary_tie_count": result.selection.boundary_tie_count,
        "cn_estimate": result.selection.cn_estimate,
        "cn_estimate_method": result.selection.cn_estimate_method,
        "qualifying_bisecting_hexagonal_planes": (
            result.geometry.qualifying_bisecting_hexagonal_planes if result.geometry else None
        ),
        "candidate_family": result.geometry.candidate_family if result.geometry else None,
        "classification": result.geometry.classification if result.geometry else None,
        "geometry_quality": result.geometry.geometry_quality if result.geometry else None,
    }]
    neighbors = []
    for i, n in enumerate(result.neighbors, 1):
        neighbors.append({
            "order_by_distance": i,
            "neighbor_id": n.neighbor_id,
            "source_labels": n.source_labels,
            "species_occupancies": n.species_occupancies,
            "symmetry_sources": n.symmetry_sources,
            "image_translation": n.image_translation,
            "fractional_coordinates": n.fractional,
            "original_cartesian_angstrom": n.cartesian_angstrom,
            "center_relative_cartesian_angstrom": n.relative_angstrom,
            "center_distance_angstrom": n.distance_angstrom,
            "mixed_occupancy": n.mixed_occupancy,
            "incompatible_overlap": n.incompatible_overlap,
        })
    planes = []
    faces = []
    edges = []
    if result.geometry:
        for plane in result.geometry.planes:
            planes.append({
                "plane_id": plane.plane_id,
                "normal_unit": plane.normal,
                "offset_angstrom_original_coordinates": plane.offset_angstrom,
                "relative_offset_angstrom": plane.relative_offset_angstrom,
                "equation": f"({plane.normal[0]:.12g})*x + ({plane.normal[1]:.12g})*y + ({plane.normal[2]:.12g})*z + ({plane.offset_angstrom:.12g}) = 0",
                "member_ids": plane.member_ids,
                "member_count": plane.member_count,
                "exactly_six": plane.exactly_six,
                "convex_hexagon": plane.is_convex_hexagon,
                "cyclic_ring_order": plane.ring_order_ids,
                "projected_center_inside": plane.projected_center_inside,
                "side_counts": plane.side_counts,
                "qualifying_bisecting_hexagon": plane.is_bisecting_hexagon,
                "singular_values": plane.singular_values,
                "max_fit_residual_angstrom": plane.max_fit_residual_angstrom,
                "rms_fit_residual_angstrom": plane.rms_fit_residual_angstrom,
                "signed_distances_angstrom": plane.signed_distances_angstrom,
                "rejection_reasons": plane.rejection_reasons,
            })
        for face in result.geometry.hull.faces:
            faces.append({
                "face_id": face.face_id,
                "polygon_size": face.polygon_size,
                "vertex_ids_cyclic": face.vertex_ids,
                "boundary_edges": face.boundary_edges,
                "normal_unit": face.normal,
                "offset_angstrom_original_coordinates": face.offset_angstrom,
                "max_planarity_residual_angstrom": face.max_planarity_residual_angstrom,
                "rms_planarity_residual_angstrom": face.rms_planarity_residual_angstrom,
                "edge_lengths_angstrom": face.edge_lengths_angstrom,
            })
        edge_map: dict[tuple[str, str], dict] = {}
        for face in result.geometry.hull.faces:
            for edge, length in zip(face.boundary_edges, face.edge_lengths_angstrom):
                key = tuple(sorted(edge))
                item = edge_map.setdefault(key, {"length": length, "faces": []})
                item["faces"].append(face.face_id)
        edges = [{
            "vertex_a": key[0],
            "vertex_b": key[1],
            "adjacent_face_ids": sorted(value["faces"]),
            "edge_length_angstrom": value["length"],
            "edge_role": "unique polyhedron face boundary; not an independently established chemical bond",
        } for key, value in sorted(edge_map.items())]
    return {"summary": summary, "neighbors": neighbors, "planes": planes,
            "faces": faces, "edges": edges}


def export_analysis(
    result: AnalysisResult,
    output_root: str | Path = "04_results",
    include_interactive_html: bool = True,
    include_static_png: bool = True,
    sphere_size_factor: float = 0.45,
) -> Path:
    """Export one result into a unique directory; never overwrites prior evidence."""
    started = datetime.now(timezone.utc).isoformat()
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # The folder name says which file and site it holds, e.g.
    # run_20261011T031500Z_3-Mg-1_Mg_1a2b3c4d
    run_dir = root / _run_id("run", Path(result.source_path).stem, result.center_label)
    run_dir.mkdir(exist_ok=False)
    rows = _rows_for_result(result)
    summary_record = rows["summary"][0]
    first_fields = [
        "classification", "candidate_family", "geometry_quality",
        "selection_status", "selected_count", "d12_angstrom", "d13_angstrom",
        "qualifying_bisecting_hexagonal_planes", "status",
    ]
    ordered_fields = first_fields + [
        key for key in summary_record if key not in first_fields
    ]
    summary_rows = [
        {"field": key, "value": _csv_value(summary_record[key])}
        for key in ordered_fields
    ]
    # The primary summary is vertical so it is easy to scan in VS Code.
    _write_csv(run_dir / "summary.csv", ["field", "value"], summary_rows)
    _write_csv(run_dir / "neighbors.csv", [
        "order_by_distance", "neighbor_id", "source_labels", "species_occupancies",
        "symmetry_sources", "image_translation", "fractional_coordinates",
        "original_cartesian_angstrom", "center_relative_cartesian_angstrom",
        "center_distance_angstrom", "mixed_occupancy", "incompatible_overlap",
    ], rows["neighbors"])
    _write_csv(run_dir / "planes.csv", [
        "plane_id", "normal_unit", "offset_angstrom_original_coordinates",
        "relative_offset_angstrom", "equation", "member_ids", "member_count",
        "exactly_six", "convex_hexagon", "cyclic_ring_order", "projected_center_inside",
        "side_counts", "qualifying_bisecting_hexagon", "singular_values",
        "max_fit_residual_angstrom", "rms_fit_residual_angstrom",
        "signed_distances_angstrom", "rejection_reasons",
    ], rows["planes"])
    _write_csv(run_dir / "faces.csv", [
        "face_id", "polygon_size", "vertex_ids_cyclic", "boundary_edges", "normal_unit",
        "offset_angstrom_original_coordinates", "max_planarity_residual_angstrom",
        "rms_planarity_residual_angstrom", "edge_lengths_angstrom",
    ], rows["faces"])
    _write_csv(run_dir / "edges.csv", [
        "vertex_a", "vertex_b", "adjacent_face_ids", "edge_length_angstrom", "edge_role",
    ], rows["edges"])

    viewer_errors = []
    outputs = ["summary.csv", "neighbors.csv", "planes.csv", "faces.csv", "edges.csv"]
    if result.geometry is not None and result.status == "complete":
        if include_static_png:
            try:
                from .plotting import save_static_png
                save_static_png(result, run_dir / "coordination_model.png", sphere_size_factor)
                outputs.append("coordination_model.png")
            except Exception as exc:
                viewer_errors.append(f"static PNG export failed: {type(exc).__name__}: {exc}")
        if include_interactive_html:
            try:
                from .plotting import export_interactive_html
                export_interactive_html(result, run_dir / "coordination_model.html", sphere_size_factor)
                outputs.append("coordination_model.html")
            except Exception as exc:
                viewer_errors.append(f"interactive HTML export failed: {type(exc).__name__}: {exc}")

    geometry = result.geometry
    manifest = {
        "run_id": run_dir.name,
        "created_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "application": "CN12 analysis",
        "code_version": __version__,
        "source_code_sha256": _source_hashes(),
        "original_source": {
            "path": result.source_path,
            "sha256": result.source_sha256,
            "block_name": result.block_name,
            "center_site_id": result.center_site_id,
            "center_label": result.center_label,
        },
        "parser": {"name": result.parser, "version": result.parser_version},
        "package_versions": _package_versions(),
        "coordinate_metadata": {
            "fractional_source_precision": "preserved from parsed CIF numeric values; no coordinate rounding in geometry",
            "cartesian_units": "angstrom",
            "cell_angle_units_in_manifest": "degrees",
            "cifkit_unit_cell_angle_api": "radians in installed CIFKit API; not used for core geometry",
            "positions_are_center_relative_for_geometry": True,
        },
        "selection_policy": {
            "mode": result.selection.selection_mode,
            "selected_count": result.selection.selected_count,
            "status": result.selection.status,
            "d12_angstrom": result.selection.d12_angstrom,
            "d13_angstrom": result.selection.d13_angstrom,
            "d12_shell_count": result.selection.d12_shell_count,
            "boundary_tie_count": result.selection.boundary_tie_count,
            "rationale": result.selection.rationale,
        },
        "settings": result.settings,
        # The full geometry (planes, faces, edges) is in result.json; the
        # manifest keeps only the outcome so the two files do not duplicate it.
        "geometry_summary": None if geometry is None else {
            "classification": geometry.classification,
            "candidate_family": geometry.candidate_family,
            "geometry_quality": geometry.geometry_quality,
            "qualifying_bisecting_hexagonal_planes": geometry.qualifying_bisecting_hexagonal_planes,
            "face_size_counts": geometry.hull.face_size_counts,
            "edge_count": geometry.hull.edge_count,
            "full_record": "result.json",
        },
        "warnings": result.warnings + viewer_errors,
        "outputs": outputs,
        "viewer_export_errors": viewer_errors,
    }
    payload = to_jsonable(result)
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return run_dir


def export_batch_summary(batch: dict, output_root: str | Path = "04_results") -> Path:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / _run_id("batch")
    run_dir.mkdir(exist_ok=False)
    file_rows = batch.get("file_rows", [])
    site_rows = batch.get("site_rows", [])
    file_columns = sorted({key for row in file_rows for key in row}) or ["path", "status"]
    site_columns = sorted({key for row in site_rows for key in row}) or ["path", "status"]
    _write_csv(run_dir / "files.csv", file_columns, file_rows)
    _write_csv(run_dir / "sites.csv", site_columns, site_rows)
    manifest = {key: value for key, value in batch.items() if key not in {"file_rows", "site_rows"}}
    manifest["run_id"] = run_dir.name
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["code_version"] = __version__
    manifest["source_code_sha256"] = _source_hashes()
    (run_dir / "batch_manifest.json").write_text(json.dumps(to_jsonable(manifest), indent=2), encoding="utf-8")
    return run_dir
