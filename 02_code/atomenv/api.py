"""Documented local Python API. The Streamlit app calls exactly these functions.

Example::

    import sys; sys.path.insert(0, "02_code")
    from atomenv import api
    records = api.analyze_cif("03_data/demo_cod/9008468.cif", cn=12)
    print(records[0]["cn12"]["classification"])

Reuse of the existing research code (unchanged):

* ``cn12.crystal.parse_cif_blocks`` - Gemmi parsing, symmetry expansion, site orbits.
* ``cn12.crystal.extract_nearest_neighbors`` - nearest-N periodic search and tie diagnostics.
* ``cn12.crystal._generate_neighbors_within_radius`` - complete periodic search to a radius
  (used for the extended region view).
* ``cn12.pipeline.analyze_parsed_block`` / ``AnalysisSettings`` / ``plain_verdict`` - the full
  CN12 plane, hull and family analysis.
* ``cn12.geometry.reconstruct_hull`` / ``characteristic_length`` - convex-hull polygon faces and
  edges, used on its own for coordination numbers other than 12.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from cn12 import __version__ as CN12_VERSION
from cn12.crystal import _find_center, _generate_neighbors_within_radius, extract_nearest_neighbors
from cn12.geometry import HullSettings, characteristic_length, reconstruct_hull
from cn12.models import AnalysisResult, CenterSite, CrystalBlock, HullDiagnostics, Neighbor, SelectionDiagnostics
from cn12.pipeline import AnalysisSettings, analyze_parsed_block, plain_verdict

from . import __version__ as ATOMENV_VERSION
from .registry import MAX_CN, MIN_CN, capability
from .sources import InputFile, Workspace

EXTENDED_ATOM_CAP = 100
UNIT_CELL_ATOM_CAP = 600


@dataclass
class SiteAnalysis:
    """Everything computed for one center site at one requested CN."""

    input_file: InputFile | None
    block: CrystalBlock
    center: CenterSite
    cn: int
    settings: AnalysisSettings
    neighbors: list[Neighbor]
    selection: SelectionDiagnostics
    center_cartesian: tuple[float, float, float]
    capability: dict
    hull: HullDiagnostics | None = None
    hull_reason: str = ""
    cn12: AnalysisResult | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.cn12 is not None:
            return plain_verdict(self.cn12)
        if self.selection.status != "unambiguous":
            return f"no unique nearest-{self.cn} shell ({self.selection.status.replace('_', ' ')})"
        return f"CN{self.cn} shell computed; no family analysis for CN{self.cn} yet"


# ---- loading ---------------------------------------------------------------
def load_paths(paths, workspace: Workspace | None = None, source: str = "local") -> Workspace:
    """Load local CIF paths (or ``.zip`` archives) into a workspace with provenance."""
    workspace = workspace or Workspace()
    for item in paths:
        path = Path(item)
        if path.suffix.lower() == ".zip":
            workspace.add_zip(path.name, path.read_bytes())
        else:
            workspace.add_path(path, source=source)
    return workspace


def site_table(block: CrystalBlock) -> list[dict]:
    """One row per symmetry-distinct center site (representative of each orbit)."""
    rows = []
    for site in block.center_sites:
        rows.append({
            "site_id": site.site_id,
            "label": site.display_label,
            "species": ", ".join(f"{c.element} {c.occupancy:g}" for c in site.components),
            "multiplicity": site.symmetry_multiplicity,
            "x_frac": site.fractional[0], "y_frac": site.fractional[1], "z_frac": site.fractional[2],
        })
    return rows


# ---- analysis ----------------------------------------------------------------
def analyze_site(
    block: CrystalBlock,
    site_id: str,
    cn: int = 12,
    settings: AnalysisSettings = AnalysisSettings(),
    input_file: InputFile | None = None,
    source_path: str | Path | None = None,
) -> SiteAnalysis:
    """Analyze one center site.

    CN12 runs the unchanged ``cn12.pipeline.analyze_parsed_block``. Other CNs in
    4-16 run the same neighbor search with ``count=cn`` and, when the shell is
    unambiguous, the generic convex-hull reconstruction. No family label is made
    for CNs without a registered plugin.
    """
    if not MIN_CN <= cn <= MAX_CN:
        raise ValueError(f"CN must be between {MIN_CN} and {MAX_CN}; got {cn}")
    cap = capability(cn)
    lattice = np.asarray(block.lattice_columns_angstrom, dtype=float)
    path = source_path or (input_file.path if input_file else None) or block.source_path
    if cn == 12:
        run_settings = replace(settings, neighbor_count=12)
        result = analyze_parsed_block(block, path, site_id, run_settings)
        center, _ = _find_center(block, site_id)
        return SiteAnalysis(
            input_file=input_file, block=block, center=center, cn=12, settings=run_settings,
            neighbors=result.neighbors, selection=result.selection,
            center_cartesian=result.center_cartesian_angstrom, capability=cap,
            hull=result.geometry.hull if result.geometry else None,
            hull_reason="" if result.geometry else "classification withheld; see selection status",
            cn12=result, warnings=list(result.warnings),
        )
    run_settings = replace(settings, neighbor_count=cn)
    center, neighbors, selection = extract_nearest_neighbors(
        block, site_id, count=cn,
        candidate_target=max(settings.candidate_target, cn + 1),
        tie_relative_tolerance=settings.neighbor_tie_relative,
        max_translation_shell=settings.max_translation_shell,
    )
    center_cart = tuple(float(x) for x in lattice @ np.asarray(center.fractional, dtype=float))
    warnings = list(block.warnings)
    hull, reason = None, ""
    if selection.status == "unambiguous" and len(neighbors) == cn:
        relative = np.asarray([n.relative_angstrom for n in neighbors], dtype=float)
        hull = reconstruct_hull(
            relative, [n.neighbor_id for n in neighbors], characteristic_length(relative),
            center_cartesian=center_cart,
            settings=HullSettings(face_merge_relative=settings.face_merge_relative,
                                  face_normal_merge_radians=settings.face_normal_merge_radians),
        )
        warnings.extend(hull.warnings)
    else:
        reason = f"hull not computed: selection status is {selection.status}"
        warnings.append(f"No unique nearest-{cn} shell; hull diagnostics withheld.")
    return SiteAnalysis(
        input_file=input_file, block=block, center=center, cn=cn, settings=run_settings,
        neighbors=neighbors, selection=selection, center_cartesian=center_cart,
        capability=cap, hull=hull, hull_reason=reason, warnings=warnings,
    )


def extended_region(analysis: SiteAnalysis, max_atoms: int = 80) -> tuple[list[Neighbor], str]:
    """Nearest periodic positions around the center for display (capped).

    Returns the positions and a sentence explaining the cap that was applied.
    """
    max_atoms = int(min(max(max_atoms, analysis.cn + 1), EXTENDED_ATOM_CAP))
    block, center = analysis.block, analysis.center
    physical = next(p for p in block.physical_sites if p.position_id == center.position_id)
    radius = max(analysis.selection.complete_search_radius_angstrom, 1.0)
    found: list[Neighbor] = []
    for _ in range(8):
        found = _generate_neighbors_within_radius(block, center, physical, radius,
                                                  max_translation_shell=analysis.settings.max_translation_shell)
        if len(found) >= max_atoms:
            break
        radius *= 1.35
    shown = found[:max_atoms]
    note = (f"Showing the nearest {len(shown)} periodic positions around the center "
            f"(cap {max_atoms}; hard limit {EXTENDED_ATOM_CAP} to keep the view responsive).")
    return shown, note


def unit_cell_positions(block: CrystalBlock) -> tuple[list[dict], str]:
    """Physical sites in the cell, plus images on faces/edges/corners for display only."""
    lattice = np.asarray(block.lattice_columns_angstrom, dtype=float)
    rows = []
    for site in block.physical_sites:
        f0 = np.asarray(site.fractional, dtype=float)
        species = {}
        for c in site.components:
            species[c.element] = species.get(c.element, 0.0) + c.occupancy
        for shift in np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij")).reshape(3, -1).T:
            if np.any((shift == 1) & (np.abs(f0) > 1e-8)):
                continue
            f = f0 + shift
            rows.append({
                "position_id": site.position_id,
                "labels": "+".join(sorted({c.label for c in site.components})),
                "species": tuple(sorted(species.items())),
                "fractional": tuple(float(x) for x in f),
                "cartesian": tuple(float(x) for x in lattice @ f),
                "boundary_image": bool(np.any(shift)),
            })
    note = ""
    if len(rows) > UNIT_CELL_ATOM_CAP:
        rows = [r for r in rows if not r["boundary_image"]][:UNIT_CELL_ATOM_CAP]
        note = (f"Cell has many positions; boundary images hidden and display capped at "
                f"{UNIT_CELL_ATOM_CAP} positions.")
    return rows, note


# ---- convenience for scripts -----------------------------------------------
def analyze_cif(path: str | Path, site: str | None = None, block: str | None = None,
                cn: int = 12, settings: AnalysisSettings = AnalysisSettings()) -> list[dict]:
    """Parse a local CIF and return feature records (one per analyzed site).

    The file is copied into a private workspace exactly as the app does, so the
    provenance block (hash, parse status) matches an app export of the same file.
    """
    from .features import feature_record as _record

    workspace = Workspace()
    try:
        record = workspace.add_path(path, source="local")
        if record.status != "parsed":
            raise ValueError(f"{record.name}: {record.status}: {record.reason}")
        blocks = [b for b in record.blocks if block is None or b.block_name == block]
        if not blocks:
            raise KeyError(f"No usable CIF block named {block!r}")
        out = []
        for b in blocks:
            sites = [s for s in b.center_sites if site is None or site in
                     {s.site_id, s.display_label, *s.source_ids}]
            for s in sites:
                out.append(_record(analyze_site(b, s.site_id, cn, settings, input_file=record)))
        if site is not None and not out:
            raise KeyError(f"No center site matching {site!r}")
        return out
    finally:
        workspace.close()


def feature_record(analysis: SiteAnalysis) -> dict:
    from .features import feature_record as _record

    return _record(analysis)


VERSIONS = {"atomenv": ATOMENV_VERSION, "cn12": CN12_VERSION}


# ---- plane extension -------------------------------------------------------
PLANE_EXTENSION_CAP = 400


@dataclass
class PlaneExtension:
    """Periodic positions lying on one center plane, beyond the CN shell."""

    plane_id: str
    normal: tuple[float, float, float]
    miller: tuple[int, int, int] | None
    d_spacing_angstrom: float | None
    tolerance_angstrom: float
    atoms: list[Neighbor]
    in_plane_xy: np.ndarray            # (n, 2) in-plane coordinates relative to the center
    off_plane_angstrom: np.ndarray     # signed distance of each atom from the plane
    shell_member_ids: set[str]
    search_radius_angstrom: float


def _in_plane_basis(normal: np.ndarray, anchor: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal (u, v) spanning the plane; u points at ``anchor`` when given."""
    if anchor is not None:
        u = anchor - (anchor @ normal) * normal
    if anchor is None or np.linalg.norm(u) < 1e-8:
        u = np.cross(normal, [1.0, 0.0, 0.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(normal, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    return u, np.cross(normal, u)


def miller_indices(block: CrystalBlock, normal, max_index: int = 12) -> tuple[tuple[int, int, int] | None, float | None]:
    """Smallest integer (hkl) whose lattice plane has this Cartesian normal, and d_hkl.

    Uses n·(A f) = (Aᵀn)·f, so (hkl) ∝ Aᵀn. Returns (None, None) when no index
    with |h|,|k|,|l| <= max_index matches within 0.1°.
    """
    lattice = np.asarray(block.lattice_columns_angstrom, dtype=float)
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    g = lattice.T @ n
    g = g / np.max(np.abs(g))
    recip = np.linalg.inv(lattice).T  # columns: reciprocal vectors (no 2π)
    for scale in range(1, max_index + 1):
        hkl = np.round(g * scale)
        if np.max(np.abs(hkl)) > max_index or not np.any(hkl):
            continue
        cart = recip @ hkl
        cos = abs(float(cart @ n) / np.linalg.norm(cart))
        if cos > np.cos(np.radians(0.1)):
            h = hkl.astype(int)
            from math import gcd
            k = gcd(gcd(abs(h[0]), abs(h[1])), abs(h[2])) or 1
            h = h // k
            if (h < 0).sum() > (h > 0).sum():
                h = -h
            return (int(h[0]), int(h[1]), int(h[2])), float(1.0 / np.linalg.norm(recip @ h))
    return None, None


def plane_extension(analysis: SiteAnalysis, plane_id: str, n_atoms: int = 100,
                    tolerance_angstrom: float = 0.05) -> PlaneExtension:
    """Nearest ``n_atoms`` periodic positions lying on a center plane (|signed distance| <= tol)."""
    if analysis.cn12 is None or analysis.cn12.geometry is None:
        raise ValueError("plane analysis is not available for this site")
    plane = next(p for p in analysis.cn12.geometry.planes if p.plane_id == plane_id)
    n_atoms = int(min(max(n_atoms, 1), PLANE_EXTENSION_CAP))
    normal = np.asarray(plane.normal, dtype=float)
    normal /= np.linalg.norm(normal)
    block, center = analysis.block, analysis.center
    physical = next(p for p in block.physical_sites if p.position_id == center.position_id)
    radius = max(analysis.selection.d12_angstrom or 2.0, 1.0) * 2.5
    on_plane: list[Neighbor] = []
    for _ in range(10):
        try:
            found = _generate_neighbors_within_radius(block, center, physical, radius,
                                                      max_translation_shell=analysis.settings.max_translation_shell)
        except ValueError:
            break
        on_plane = [n for n in found if abs(float(np.asarray(n.relative_angstrom) @ normal)) <= tolerance_angstrom]
        if len(on_plane) > n_atoms or len(found) > 60000:
            break
        radius *= 1.3
    atoms = on_plane[:n_atoms]
    rel = np.asarray([a.relative_angstrom for a in atoms], dtype=float).reshape(-1, 3)
    members = [n for n in analysis.neighbors if n.neighbor_id in set(plane.member_ids)]
    anchor = np.asarray(members[0].relative_angstrom, dtype=float) if members else None
    u, v = _in_plane_basis(normal, anchor)
    miller, d = miller_indices(block, normal)
    return PlaneExtension(
        plane_id=plane_id, normal=tuple(float(x) for x in normal), miller=miller, d_spacing_angstrom=d,
        tolerance_angstrom=tolerance_angstrom, atoms=atoms,
        in_plane_xy=np.column_stack([rel @ u, rel @ v]) if len(rel) else np.zeros((0, 2)),
        off_plane_angstrom=rel @ normal if len(rel) else np.zeros(0),
        shell_member_ids=set(plane.member_ids), search_radius_angstrom=radius,
    )


def in_plane_shells(ext: PlaneExtension, rel_tol: float = 1e-3) -> list[dict]:
    """Group in-plane atoms into distance shells (count and species per shell)."""
    shells: list[dict] = []
    for a in ext.atoms:
        d = a.distance_angstrom
        if shells and abs(d - shells[-1]["distance (Å)"]) <= rel_tol * d:
            shells[-1]["count"] += 1
            shells[-1]["_sp"].update(e for e, _ in a.species_occupancies)
        else:
            shells.append({"shell": len(shells) + 1, "distance (Å)": round(d, 5), "count": 1,
                           "_sp": {e for e, _ in a.species_occupancies}})
    for s in shells:
        s["species"] = ", ".join(sorted(s.pop("_sp")))
    first = shells[0]["distance (Å)"] if shells else None
    for s in shells:
        s["d / d₁"] = round(s["distance (Å)"] / first, 4) if first else None
    return shells
