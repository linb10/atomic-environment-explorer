"""Per-site feature records: schema, JSON record and flat CSV row.

Conventions (also in docs/ATOMIC_ENVIRONMENT_EXPLORER.md):

* Lengths are in angstrom (A). Angles are in degrees unless the key ends in ``_radians``.
* Cartesian frame: ``r = L @ f`` with ``L`` = ``structure.lattice_columns_angstrom``
  (columns are the a, b, c cell vectors from Gemmi's orthogonalization: a along x,
  b in the xy plane). This frame depends on that convention and is not rotation invariant.
* ``center.fractional`` is the representative wrapped into [0, 1). A neighbor's
  ``fractional`` is unwrapped: it equals the stored site position plus the integer
  ``image_translation``, so ``relative_angstrom = L @ (f_neighbor - f_center)``.
* Missing, withheld or unsupported values are ``null`` in JSON and empty in CSV,
  with a ``status`` or ``reason`` field saying why. Nothing is imputed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import csv
import io
import json

import numpy as np

from cn12.models import to_jsonable

from .api import VERSIONS, SiteAnalysis

SCHEMA_VERSION = "atomenv.site_features/1.0"

FIELD_DOCS: dict[str, dict] = {
    "schema_version": {"description": "Feature schema identifier."},
    "created_utc": {"description": "Time the record was produced (UTC, ISO 8601)."},
    "software": {"description": "atomenv and cn12 package versions."},
    "provenance": {"description": "Source (demo/upload/zip/cod/local), file name, archive, public record ID, retrieval time, SHA-256 of raw bytes, parse status, license, citation."},
    "structure": {"description": "CIF block name, formula as written in the CIF, space group, cell lengths (A), angles (degrees), lattice column matrix (A)."},
    "center": {"description": "Center site ID, label, species/occupancy pairs, symmetry multiplicity, fractional (wrapped) and Cartesian (A) position."},
    "selection": {"description": "Neighbor-selection rule and diagnostics: requested CN, status (unambiguous / boundary_tie / incomplete_search / unsupported_input), d_N and d_N+1 (A), relative gap, tie counts, candidate count, heuristic CN estimate (unvalidated)."},
    "neighbors": {"description": "Selected neighbors ordered by distance: stable ID, labels, species/occupancy, image translation, unwrapped fractional, Cartesian and center-relative coordinates (A), distance (A), mixed-occupancy flag."},
    "hull": {"description": "Convex hull of the neighbor positions with coplanar triangles merged into polygons: face-size counts, edge count, vertex degrees, Euler characteristic, center inside, volume (A^3), max face planarity residual (A). Geometric edges, not chemical bonds."},
    "cn12": {"description": "CN12 plugin output when CN=12 and the shell is unambiguous: plane counts, plane records (normal, residuals, members), distortion metrics, candidate family, classification, quality, reasons. Labels are program outputs, not verified structural facts."},
    "capability": {"description": "What is implemented for the requested CN; family analysis 'available' or 'planned'."},
    "settings": {"description": "All analysis settings, with effective A tolerances where defined."},
    "warnings": {"description": "Parser and analysis warnings."},
}

CSV_COLUMNS: list[tuple[str, str]] = [
    ("schema_version", ""), ("source", ""), ("filename", ""), ("record_id", ""), ("sha256", ""),
    ("block_name", ""), ("formula", ""), ("space_group", ""), ("space_group_number", ""),
    ("a_angstrom", "A"), ("b_angstrom", "A"), ("c_angstrom", "A"),
    ("alpha_degrees", "deg"), ("beta_degrees", "deg"), ("gamma_degrees", "deg"),
    ("center_site_id", ""), ("center_label", ""), ("center_species", "element:occupancy"),
    ("symmetry_multiplicity", ""),
    ("center_x_frac", ""), ("center_y_frac", ""), ("center_z_frac", ""),
    ("center_x_angstrom", "A"), ("center_y_angstrom", "A"), ("center_z_angstrom", "A"),
    ("requested_cn", ""), ("selection_status", ""), ("d_n_angstrom", "A"), ("d_n_plus_1_angstrom", "A"),
    ("shell_gap_relative", "(d_N+1 - d_N)/d_N"), ("boundary_tie_count", ""), ("cn_estimate_heuristic", ""),
    ("neighbor_distances_angstrom", "A; semicolon list"), ("neighbor_elements", "semicolon list"),
    ("mean_distance_angstrom", "A"), ("distance_std_angstrom", "A"),
    ("hull_status", ""), ("hull_face_sizes", "size:count"), ("hull_edge_count", ""), ("hull_face_count", ""),
    ("hull_volume_angstrom3", "A^3"), ("hull_center_inside", ""), ("hull_max_face_planarity_angstrom", "A"),
    ("cn12_status", ""), ("cn12_verdict", ""), ("cn12_classification", ""), ("cn12_quality", ""),
    ("cn12_candidate_family", ""), ("cn12_bisecting_hexagonal_planes", ""), ("cn12_total_center_planes", ""),
    ("cn12_exactly_six_planes", ""), ("cn12_max_plane_residual_angstrom", "A"),
    ("cn12_center_distance_spread", "relative"), ("cn12_edge_length_spread", "relative"),
    ("cn12_normalized_shape_rmsd", "relative"), ("family_analysis", ""), ("warning_count", ""),
]


def _species(pairs) -> list[dict]:
    return [{"element": e, "occupancy": float(o)} for e, o in pairs]


def _hull_dict(analysis: SiteAnalysis) -> dict:
    hull = analysis.hull
    if hull is None:
        return {"status": "not_computed", "reason": analysis.hull_reason or "not computed"}
    data = to_jsonable(hull)
    data["note"] = "Edges are convex-hull polygon edges, not assigned chemical bonds."
    return data


def _cn12_dict(analysis: SiteAnalysis) -> dict:
    if analysis.cn != 12:
        return {"status": "not_applicable",
                "reason": f"CN12 plugin runs only for CN=12; requested CN={analysis.cn}"}
    result = analysis.cn12
    geometry = result.geometry
    if geometry is None:
        return {"status": result.status, "verdict": analysis.verdict, "classification": None,
                "reason": "classification withheld because the nearest-12 shell is not unique or complete"}
    data = to_jsonable(geometry)
    data.pop("hull", None)  # stored once under "hull"
    data["status"] = "complete"
    data["verdict"] = analysis.verdict
    data["label_is_program_output"] = True
    return data


def feature_record(analysis: SiteAnalysis) -> dict:
    block, center, sel = analysis.block, analysis.center, analysis.selection
    lattice = np.asarray(block.lattice_columns_angstrom, dtype=float)
    distances = [n.distance_angstrom for n in analysis.neighbors]
    gap = None
    if sel.d12_angstrom and sel.d13_angstrom is not None:
        gap = (sel.d13_angstrom - sel.d12_angstrom) / sel.d12_angstrom
    provenance = (analysis.input_file.provenance() if analysis.input_file else
                  {"source": "unknown", "filename": None})
    record = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "software": dict(VERSIONS),
        "units": {"length": "angstrom", "angle": "degree", "volume": "angstrom^3"},
        "coordinate_frame": ("Cartesian r = L @ f, L = structure.lattice_columns_angstrom (Gemmi "
                             "orthogonalization: a along x, b in xy plane). Neighbor fractional "
                             "coordinates are unwrapped (site + image_translation)."),
        "provenance": provenance,
        "structure": {
            "block_name": block.block_name, "formula": block.formula,
            "space_group": block.space_group, "space_group_number": block.space_group_number,
            "cell_lengths_angstrom": list(block.cell_lengths_angstrom),
            "cell_angles_degrees": list(block.cell_angles_degrees),
            "lattice_columns_angstrom": lattice.tolist(),
            "parser": block.parser, "parser_version": block.parser_version,
        },
        "center": {
            "site_id": center.site_id, "label": center.display_label,
            "species": [{"label": c.label, "element": c.element, "occupancy": c.occupancy}
                        for c in center.components],
            "symmetry_multiplicity": center.symmetry_multiplicity,
            "fractional": list(center.fractional),
            "cartesian_angstrom": list(analysis.center_cartesian),
        },
        "selection": {
            "requested_cn": analysis.cn,
            "rule": f"nearest {analysis.cn} distinct periodic physical positions (center excluded)",
            "status": sel.status,
            "d_n_angstrom": sel.d12_angstrom,
            "d_n_plus_1_angstrom": sel.d13_angstrom,
            "shell_gap_relative": gap,
            "boundary_tie_count": sel.boundary_tie_count,
            "boundary_shell_count": sel.d12_shell_count,
            "boundary_shell_neighbor_ids": sel.boundary_shell_neighbor_ids,
            "candidate_count": sel.candidate_count,
            "complete_search_radius_angstrom": sel.complete_search_radius_angstrom,
            "cn_estimate_heuristic": sel.cn_estimate,
            "cn_estimate_method": sel.cn_estimate_method,
            "rationale": sel.rationale,
            "neighbors_authoritative": sel.status == "unambiguous",
        },
        "neighbors": [
            {
                "rank": i + 1, "neighbor_id": n.neighbor_id, "labels": list(n.source_labels),
                "species": _species(n.species_occupancies),
                "image_translation": list(n.image_translation),
                "fractional_unwrapped": list(n.fractional),
                "cartesian_angstrom": list(n.cartesian_angstrom),
                "relative_angstrom": list(n.relative_angstrom),
                "distance_angstrom": n.distance_angstrom,
                "mixed_occupancy": n.mixed_occupancy,
                "in_boundary_shell": n.neighbor_id in set(sel.boundary_shell_neighbor_ids),
            }
            for i, n in enumerate(analysis.neighbors)
        ],
        "distance_summary": {
            "mean_angstrom": float(np.mean(distances)) if distances else None,
            "std_angstrom": float(np.std(distances)) if distances else None,
            "min_angstrom": float(np.min(distances)) if distances else None,
            "max_angstrom": float(np.max(distances)) if distances else None,
        },
        "hull": _hull_dict(analysis),
        "cn12": _cn12_dict(analysis),
        "capability": analysis.capability,
        "settings": analysis.settings.as_dict(
            analysis.cn12.geometry.characteristic_length_angstrom
            if analysis.cn12 is not None and analysis.cn12.geometry is not None else None),
        "warnings": list(dict.fromkeys(analysis.warnings)),
    }
    return to_jsonable(record)


def flat_row(record: dict) -> dict:
    """One CSV row; empty string means absent / not computed (see status columns)."""
    p, s, c, sel = record["provenance"], record["structure"], record["center"], record["selection"]
    hull, cn12 = record["hull"], record["cn12"]
    has_hull = "face_size_counts" in hull
    planes = cn12.get("planes") or []
    residuals = [pl["max_fit_residual_angstrom"] for pl in planes]
    row = {
        "schema_version": record["schema_version"], "source": p.get("source"),
        "filename": p.get("filename"), "record_id": p.get("record_id"), "sha256": p.get("sha256"),
        "block_name": s["block_name"], "formula": s["formula"], "space_group": s["space_group"],
        "space_group_number": s["space_group_number"],
        "a_angstrom": s["cell_lengths_angstrom"][0], "b_angstrom": s["cell_lengths_angstrom"][1],
        "c_angstrom": s["cell_lengths_angstrom"][2],
        "alpha_degrees": s["cell_angles_degrees"][0], "beta_degrees": s["cell_angles_degrees"][1],
        "gamma_degrees": s["cell_angles_degrees"][2],
        "center_site_id": c["site_id"], "center_label": c["label"],
        "center_species": ";".join(f"{x['element']}:{x['occupancy']:g}" for x in c["species"]),
        "symmetry_multiplicity": c["symmetry_multiplicity"],
        "center_x_frac": c["fractional"][0], "center_y_frac": c["fractional"][1], "center_z_frac": c["fractional"][2],
        "center_x_angstrom": c["cartesian_angstrom"][0], "center_y_angstrom": c["cartesian_angstrom"][1],
        "center_z_angstrom": c["cartesian_angstrom"][2],
        "requested_cn": sel["requested_cn"], "selection_status": sel["status"],
        "d_n_angstrom": sel["d_n_angstrom"], "d_n_plus_1_angstrom": sel["d_n_plus_1_angstrom"],
        "shell_gap_relative": sel["shell_gap_relative"], "boundary_tie_count": sel["boundary_tie_count"],
        "cn_estimate_heuristic": sel["cn_estimate_heuristic"],
        "neighbor_distances_angstrom": ";".join(f"{n['distance_angstrom']:.6f}" for n in record["neighbors"]),
        "neighbor_elements": ";".join("+".join(x["element"] for x in n["species"]) for n in record["neighbors"]),
        "mean_distance_angstrom": record["distance_summary"]["mean_angstrom"],
        "distance_std_angstrom": record["distance_summary"]["std_angstrom"],
        "hull_status": hull.get("status"),
        "hull_face_sizes": ";".join(f"{k}:{v}" for k, v in sorted(hull.get("face_size_counts", {}).items(), key=lambda kv: int(kv[0]))) if has_hull else None,
        "hull_edge_count": hull.get("edge_count") if has_hull else None,
        "hull_face_count": hull.get("face_count") if has_hull else None,
        "hull_volume_angstrom3": hull.get("hull_volume_angstrom3"),
        "hull_center_inside": hull.get("center_inside"),
        "hull_max_face_planarity_angstrom": hull.get("max_face_planarity_residual_angstrom"),
        "cn12_status": cn12.get("status"), "cn12_verdict": cn12.get("verdict"),
        "cn12_classification": cn12.get("classification"), "cn12_quality": cn12.get("geometry_quality"),
        "cn12_candidate_family": cn12.get("candidate_family"),
        "cn12_bisecting_hexagonal_planes": cn12.get("qualifying_bisecting_hexagonal_planes"),
        "cn12_total_center_planes": cn12.get("total_distinct_center_planes"),
        "cn12_exactly_six_planes": cn12.get("exactly_six_neighbor_planes"),
        "cn12_max_plane_residual_angstrom": max(residuals) if residuals else None,
        "cn12_center_distance_spread": cn12.get("center_distance_spread"),
        "cn12_edge_length_spread": cn12.get("edge_length_spread"),
        "cn12_normalized_shape_rmsd": cn12.get("normalized_shape_rmsd"),
        "family_analysis": record["capability"].get("family_analysis"),
        "warning_count": len(record["warnings"]),
    }
    return {k: ("" if row.get(k) is None else row.get(k)) for k, _ in CSV_COLUMNS}


def records_to_csv(records: list[dict]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=[k for k, _ in CSV_COLUMNS], lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(flat_row(record))
    return buffer.getvalue()


def records_to_json(records: list[dict]) -> str:
    return json.dumps({"schema_version": SCHEMA_VERSION, "records": records}, indent=2, ensure_ascii=False)


def schema_document() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_fields": FIELD_DOCS,
        "csv_columns": [{"name": k, "unit": u} for k, u in CSV_COLUMNS],
        "missing_values": "JSON null / empty CSV cell; see status and reason fields.",
    }
