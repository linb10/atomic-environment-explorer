"""Gemmi-backed CIF parsing and mathematically bounded periodic neighbor search."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from math import ceil
from pathlib import Path
import re
from typing import Iterable

import gemmi
import numpy as np

from .models import (
    AtomComponent,
    AtomSite,
    CenterSite,
    CrystalBlock,
    Neighbor,
    PhysicalSite,
    SelectionDiagnostics,
)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text_sha256(path: str | Path) -> str:
    """Hash CIF content after newline normalization to find line-ending duplicates."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return sha256(data).hexdigest()


def _clean_cif_value(value: str | None) -> str:
    if value is None:
        return ""
    value = str(value).strip().strip("'\"")
    return "" if value in {".", "?"} else value


def _wrap_fractional(value: Iterable[float]) -> np.ndarray:
    f = np.asarray(tuple(value), dtype=float)
    f = f - np.floor(f)
    # Snap values within 1e-10 of a cell edge onto 0 so 0 and 1 are one position.
    f[np.abs(f - 1.0) <= 1e-10] = 0.0
    f[np.abs(f) <= 1e-10] = 0.0
    return f


def _element_from_site(site: gemmi.SmallStructure.Site) -> str:
    symbol = _clean_cif_value(site.type_symbol)
    if symbol:
        match = re.match(r"^([A-Z][a-z]?)", symbol)
        if match:
            candidate = match.group(1)
            if gemmi.Element(candidate).atomic_number > 0:
                return candidate
    # CIF metadata sometimes omits type_symbol. Extract an element token from the
    # original label while preserving the label separately.
    match = re.match(r"^([A-Z][a-z]?)", _clean_cif_value(site.label))
    if match and gemmi.Element(match.group(1)).atomic_number > 0:
        return match.group(1)
    raise ValueError(f"Could not determine the chemical species for site {site.label!r}")


def _spacegroup_operations(small: gemmi.SmallStructure) -> list[gemmi.Op]:
    group = getattr(small, "spacegroup", None)
    if group is not None:
        try:
            operations = list(group.operations())
            if operations:
                return operations
        except (AttributeError, TypeError):
            pass
    parsed = []
    for op_text in getattr(small, "symops", []) or []:
        try:
            parsed.append(gemmi.Op(str(op_text)))
        except Exception:
            continue
    if parsed:
        return parsed
    return [gemmi.Op("x,y,z")]


def _cartesian_from_fractional(cell: gemmi.UnitCell, fractional: Iterable[float]) -> np.ndarray:
    v = cell.orthogonalize(gemmi.Fractional(*[float(x) for x in fractional]))
    return np.asarray([v.x, v.y, v.z], dtype=float)


def _fractional_close(a: np.ndarray, b: np.ndarray, tolerance: float) -> bool:
    delta = np.abs((a - b + 0.5) % 1.0 - 0.5)
    return bool(np.max(delta) <= tolerance)


def _make_components(rows: list[AtomSite]) -> tuple[AtomComponent, ...]:
    return tuple(
        AtomComponent(
            source_id=row.source_id,
            label=row.label,
            element=row.element,
            occupancy=float(row.occupancy),
            disorder_group=row.disorder_group,
        )
        for row in sorted(rows, key=lambda r: (r.source_id, r.label, r.element))
    )


def _components_incompatible(components: tuple[AtomComponent, ...]) -> bool:
    total = sum(max(0.0, component.occupancy) for component in components)
    return total > 1.0001


def _build_physical_sites(
    small: gemmi.SmallStructure,
    atom_sites: list[AtomSite],
    operations: list[gemmi.Op],
) -> list[PhysicalSite]:
    # Group crystallographic symmetry images by their wrapped physical position.
    cell = small.cell
    min_singular = float(np.linalg.svd(np.column_stack([
        _cartesian_from_fractional(cell, (1, 0, 0)),
        _cartesian_from_fractional(cell, (0, 1, 0)),
        _cartesian_from_fractional(cell, (0, 0, 1)),
    ]), compute_uv=False)[-1])
    # PCD-like files often store symmetry-special fractions such as 1/3 to six
    # decimal places. Merge positions within 1e-4 A in the most sensitive lattice
    # direction so those rounded copies do not become fictitious near-zero neighbors.
    frac_tol = min(1e-4, 1e-4 / max(min_singular, 1e-12))
    entries: list[tuple[np.ndarray, str, str]] = []
    for atom in atom_sites:
        for op_i, op in enumerate(operations):
            transformed = np.asarray(op.apply_to_xyz(list(atom.fractional)), dtype=float)
            wrapped = _wrap_fractional(transformed)
            entries.append((wrapped, atom.source_id, f"{atom.source_id}|symop:{op_i:03d}"))

    groups: list[dict] = []
    for fractional, source_id, symmetry_source in entries:
        match_i = next((i for i, group in enumerate(groups)
                        if _fractional_close(fractional, group["fractional"], frac_tol)), None)
        if match_i is None:
            groups.append({"fractional": fractional, "source_ids": {source_id},
                           "symmetry_sources": {symmetry_source}})
        else:
            groups[match_i]["source_ids"].add(source_id)
            groups[match_i]["symmetry_sources"].add(symmetry_source)

    row_by_id = {row.source_id: row for row in atom_sites}
    groups.sort(key=lambda g: (tuple(sorted(g["source_ids"])), tuple(np.round(g["fractional"], 10))))
    physical_sites: list[PhysicalSite] = []
    for i, group in enumerate(groups, 1):
        rows = [row_by_id[source_id] for source_id in sorted(group["source_ids"])]
        components = _make_components(rows)
        labels = "+".join(sorted({component.label for component in components}))
        frac_key = ",".join(float(x).hex() for x in group["fractional"])
        position_id = f"POS{i:04d}|{labels}|f={frac_key}"
        physical_sites.append(
            PhysicalSite(
                position_id=position_id,
                fractional=tuple(float(x) for x in group["fractional"]),
                components=components,
                symmetry_sources=tuple(sorted(group["symmetry_sources"])),
                incompatible_overlap=_components_incompatible(components),
            )
        )
    return physical_sites


def _build_center_sites(
    small: gemmi.SmallStructure,
    atom_sites: list[AtomSite],
    operations: list[gemmi.Op],
    physical_sites: list[PhysicalSite],
) -> list[CenterSite]:
    cell = small.cell
    lattice = np.column_stack([
        _cartesian_from_fractional(cell, (1, 0, 0)),
        _cartesian_from_fractional(cell, (0, 1, 0)),
        _cartesian_from_fractional(cell, (0, 0, 1)),
    ])
    orbit_tolerance = min(1e-4, 1e-4 / max(float(np.linalg.svd(lattice, compute_uv=False)[-1]), 1e-12))
    frac_orbits: dict[str, tuple[np.ndarray, ...]] = {}
    for atom in atom_sites:
        orbit = []
        for op in operations:
            orbit.append(_wrap_fractional(op.apply_to_xyz(list(atom.fractional))))
        unique: list[np.ndarray] = []
        for f in orbit:
            if not any(_fractional_close(f, old, orbit_tolerance) for old in unique):
                unique.append(f)
        frac_orbits[atom.source_id] = tuple(unique)

    orbit_groups: list[list[AtomSite]] = []
    for atom in atom_sites:
        orbit = frac_orbits[atom.source_id]
        match = None
        for i, existing in enumerate(orbit_groups):
            other = frac_orbits[existing[0].source_id]
            if len(orbit) == len(other) and all(
                any(_fractional_close(a, b, orbit_tolerance) for b in other) for a in orbit
            ):
                match = i
                break
        if match is None:
            orbit_groups.append([atom])
        else:
            orbit_groups[match].append(atom)

    center_sites: list[CenterSite] = []
    for group in orbit_groups:
        representative = group[0]
        wrapped = _wrap_fractional(representative.fractional)
        physical = next(
            p for p in physical_sites if _fractional_close(np.asarray(p.fractional), wrapped, orbit_tolerance)
        )
        labels = sorted({row.label for row in group})
        source_ids = tuple(sorted(row.source_id for row in group))
        label_text = "+".join(labels)
        site_id = f"{label_text}|{source_ids[0]}"
        center_sites.append(
            CenterSite(
                site_id=site_id,
                display_label=label_text,
                position_id=physical.position_id,
                fractional=tuple(float(x) for x in wrapped),
                components=_make_components(group),
                source_ids=source_ids,
                symmetry_multiplicity=len(frac_orbits[representative.source_id]),
            )
        )
    center_sites.sort(key=lambda site: (site.display_label, site.site_id))
    return center_sites


def _block_to_record(path: Path, block: gemmi.cif.Block) -> CrystalBlock:
    small = gemmi.make_small_structure_from_block(block)
    if len(small.sites) == 0:
        raise ValueError(f"CIF block {block.name!r} has no atom-site coordinates")
    if any(not np.all(np.isfinite([site.fract.x, site.fract.y, site.fract.z])) for site in small.sites):
        raise ValueError(f"CIF block {block.name!r} has missing or non-finite fractional coordinates")
    atom_sites: list[AtomSite] = []
    for i, site in enumerate(small.sites, 1):
        label = _clean_cif_value(site.label) or f"site{i}"
        element = _element_from_site(site)
        frac = tuple(float(x) for x in (site.fract.x, site.fract.y, site.fract.z))
        occupancy = float(site.occ)
        if not np.isfinite(occupancy) or occupancy < 0.0:
            raise ValueError(f"Invalid occupancy {occupancy!r} for site {label}")
        disorder = _clean_cif_value(getattr(site, "disorder_group", "")) or None
        atom_sites.append(AtomSite(f"row:{i:04d}:{label}", label, element, frac,
                                   occupancy, disorder))
    operations = _spacegroup_operations(small)
    physical_sites = _build_physical_sites(small, atom_sites, operations)
    center_sites = _build_center_sites(small, atom_sites, operations, physical_sites)
    cell = small.cell
    lengths = (float(cell.a), float(cell.b), float(cell.c))
    angles = (float(cell.alpha), float(cell.beta), float(cell.gamma))
    lattice = np.column_stack([
        _cartesian_from_fractional(cell, (1.0, 0.0, 0.0)),
        _cartesian_from_fractional(cell, (0.0, 1.0, 0.0)),
        _cartesian_from_fractional(cell, (0.0, 0.0, 1.0)),
    ])
    sg = getattr(small, "spacegroup", None)
    sg_name = _clean_cif_value(getattr(sg, "hm", "")) or _clean_cif_value(
        block.find_value("_space_group_name_H-M_alt")
    ) or _clean_cif_value(block.find_value("_symmetry_space_group_name_H-M")) or "unknown"
    number_text = _clean_cif_value(block.find_value("_space_group_IT_number")) or _clean_cif_value(
        block.find_value("_symmetry_Int_Tables_number")
    )
    try:
        number = int(number_text)
    except (ValueError, TypeError):
        number = int(sg.number) if sg is not None else None
    formula = _clean_cif_value(block.find_value("_chemical_formula_sum")) or _clean_cif_value(
        block.find_value("_chemical_formula_structural")
    ) or "unknown"
    return CrystalBlock(
        source_path=str(path.resolve()),
        block_name=block.name,
        formula=formula,
        space_group=sg_name,
        space_group_number=number,
        cell_lengths_angstrom=lengths,
        cell_angles_degrees=angles,
        lattice_columns_angstrom=lattice.tolist(),
        atom_sites=atom_sites,
        physical_sites=physical_sites,
        center_sites=center_sites,
        parser="Gemmi SmallStructure",
        parser_version=gemmi.__version__,
        warnings=[],
    )


def parse_cif_blocks(path: str | Path) -> list[CrystalBlock]:
    """Parse every data block without modifying the supplied raw CIF."""
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        document = gemmi.cif.read_file(str(source))
    except Exception as exc:
        raise ValueError(f"Gemmi could not parse {source}: {exc}") from exc
    blocks: list[CrystalBlock] = []
    failures = []
    for block in document:
        try:
            blocks.append(_block_to_record(source, block))
        except Exception as exc:
            failures.append(f"block {block.name!r}: {exc}")
    if not blocks:
        raise ValueError("No usable CIF data blocks. " + "; ".join(failures))
    if failures:
        for result in blocks:
            result.warnings.extend(failures)
    return blocks


def select_block(blocks: list[CrystalBlock], block_name: str | None = None) -> CrystalBlock:
    if block_name is None:
        if len(blocks) == 1:
            return blocks[0]
        raise ValueError(
            "CIF contains multiple usable data blocks: "
            + ", ".join(block.block_name for block in blocks)
            + ". Pass block_name explicitly."
        )
    for block in blocks:
        if block.block_name == block_name:
            return block
    raise KeyError(f"No usable CIF block named {block_name!r}")


def center_sites_table(block: CrystalBlock) -> list[dict]:
    """Readable center-site selection table, one representative per symmetry orbit."""
    rows = []
    for site in block.center_sites:
        species = ", ".join(f"{c.element} {c.occupancy:g}" for c in site.components)
        rows.append({
            "site_id": site.site_id,
            "label": site.display_label,
            "species_occupancies": species,
            "fractional": site.fractional,
            "symmetry_multiplicity": site.symmetry_multiplicity,
            "source_ids": site.source_ids,
        })
    return rows


def _find_center(block: CrystalBlock, selector: str | None) -> tuple[CenterSite, PhysicalSite]:
    if selector is None:
        if len(block.center_sites) != 1:
            options = ", ".join(f"{s.site_id} ({s.display_label})" for s in block.center_sites)
            raise ValueError(f"Choose a center site. Available representatives: {options}")
        selected = block.center_sites[0]
    else:
        matches = [site for site in block.center_sites
                   if selector in {site.site_id, site.display_label, *site.source_ids}
                   or selector in {c.label for c in site.components}]
        if len(matches) != 1:
            options = ", ".join(f"{s.site_id} ({s.display_label})" for s in block.center_sites)
            raise ValueError(f"Center selector {selector!r} matched {len(matches)} sites. Available: {options}")
        selected = matches[0]
    physical = next(p for p in block.physical_sites if p.position_id == selected.position_id)
    return selected, physical


def _species_occupancies(components: tuple[AtomComponent, ...]) -> tuple[tuple[str, float], ...]:
    by_element: dict[str, float] = defaultdict(float)
    for component in components:
        by_element[component.element] += component.occupancy
    return tuple(sorted((element, float(occupancy)) for element, occupancy in by_element.items()))


def _translation_bound(radius: float, lattice: np.ndarray) -> int:
    minimum_singular = float(np.linalg.svd(lattice, compute_uv=False)[-1])
    if minimum_singular <= 0:
        raise ValueError("Unit-cell lattice is singular")
    # If |B x| <= R, then |x| <= R/sigma_min(B). Since each fractional site
    # offset lies in (-1,1), this integer cube contains every translation that
    # could contribute within the requested Cartesian search radius.
    return int(ceil(radius / minimum_singular + 1.0))


def _generate_neighbors_within_radius(
    block: CrystalBlock,
    center_site: CenterSite,
    center_physical: PhysicalSite,
    radius: float,
    max_translation_shell: int = 80,
) -> list[Neighbor]:
    lattice = np.asarray(block.lattice_columns_angstrom, dtype=float)
    center_frac = np.asarray(center_site.fractional, dtype=float)
    center_cart = lattice @ center_frac
    bound = _translation_bound(radius, lattice)
    if bound > max_translation_shell:
        raise ValueError(
            f"A complete search to {radius:.4g} A in this skewed cell requires image bound {bound}; "
            f"configured computational cap is {max_translation_shell}. Increase it after reviewing the cell."
        )
    # Every lattice translation in the bounded cube, in the same order that
    # itertools.product would visit them.
    steps = np.arange(-bound, bound + 1)
    translations = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
    # One vectorized pass discards the images that are clearly out of range; the
    # few survivors are then measured one at a time with the reference formula,
    # so the reported distances do not depend on this shortcut.
    prefilter_radius = radius * (1.0 + 1e-9) + 1e-9
    neighbors: list[Neighbor] = []
    for site in block.physical_sites:
        frac0 = np.asarray(site.fractional, dtype=float)
        rough = np.linalg.norm((frac0 + translations - center_frac) @ lattice.T, axis=1)
        labels = tuple(sorted({c.label for c in site.components}))
        for index in np.flatnonzero(rough <= prefilter_radius):
            translation = translations[index]
            t = tuple(int(x) for x in translation)
            fractional = frac0 + translation
            relative = lattice @ (fractional - center_frac)
            distance = float(np.linalg.norm(relative))
            if distance <= 1e-9:
                # Exclude only the central physical position itself. A nonzero
                # periodic image of the same crystallographic site is retained.
                continue
            if distance > radius:
                continue
            cartesian = center_cart + relative
            neighbor_id = f"{site.position_id}|T={t[0]},{t[1]},{t[2]}"
            neighbors.append(
                Neighbor(
                    neighbor_id=neighbor_id,
                    position_id=site.position_id,
                    source_labels=labels,
                    species_occupancies=_species_occupancies(site.components),
                    symmetry_sources=site.symmetry_sources,
                    image_translation=t,
                    fractional=tuple(float(x) for x in fractional),
                    cartesian_angstrom=tuple(float(x) for x in cartesian),
                    relative_angstrom=tuple(float(x) for x in relative),
                    distance_angstrom=distance,
                    mixed_occupancy=(len(site.components) > 1 or
                                     any(c.occupancy < 0.999 for c in site.components)),
                    incompatible_overlap=site.incompatible_overlap,
                )
            )
    # Same physical position should occur once even when the underlying CIF lists
    # multiple alternative species rows at that position; _build_physical_sites
    # already groups those rows before this search.
    neighbors.sort(key=lambda n: (n.distance_angstrom, n.neighbor_id))
    return neighbors


def extract_nearest_neighbors(
    block: CrystalBlock,
    center_selector: str | None,
    count: int = 12,
    candidate_target: int = 16,
    tie_relative_tolerance: float = 1e-5,
    max_expansions: int = 12,
    max_translation_shell: int = 80,
) -> tuple[CenterSite, list[Neighbor], SelectionDiagnostics]:
    """Return a complete periodic search through at least the 13th-neighbor shell."""
    center, center_physical = _find_center(block, center_selector)
    lattice = np.asarray(block.lattice_columns_angstrom, dtype=float)
    smallest_cell_vector = min(np.linalg.norm(lattice[:, i]) for i in range(3))
    radius = max(1.0, smallest_cell_vector * 0.5)
    candidates: list[Neighbor] = []
    for _ in range(max_expansions):
        candidates = _generate_neighbors_within_radius(
            block, center, center_physical, radius,
            max_translation_shell=max_translation_shell,
        )
        if len(candidates) >= max(candidate_target, count + 1):
            break
        radius *= 1.7
    if len(candidates) < count + 1:
        diag = SelectionDiagnostics(
            selected_count=min(count, len(candidates)),
            selection_mode=f"nearest_{count}_preview",
            status="incomplete_search",
            shell_distance_angstrom=None,
            d12_angstrom=candidates[count-1].distance_angstrom if len(candidates) >= count else None,
            d13_angstrom=None,
            boundary_tie_count=0,
            d12_shell_count=0,
            boundary_shell_neighbor_ids=[],
            complete_search_radius_angstrom=radius,
            candidate_count=len(candidates),
            cn_estimate=None,
            cn_estimate_method="not available: fewer than 13 distinct periodic positions found",
            rationale="The complete image search did not find enough distinct positions to inspect d13.",
        )
        return center, candidates[:count], diag

    selected = candidates[:count]
    d12 = candidates[count - 1].distance_angstrom
    d13 = candidates[count].distance_angstrom
    median_distance = float(np.median([n.distance_angstrom for n in candidates[:candidate_target]]))
    tie_abs = max(1e-8, tie_relative_tolerance * median_distance)
    tied = [n for n in candidates if abs(n.distance_angstrom - d12) <= tie_abs]
    tie = abs(d13 - d12) <= tie_abs
    status = "boundary_tie" if tie else "unambiguous"

    # A reproducible, explicitly heuristic CN estimate: the largest relative gap
    # before the first 24 candidates. It is descriptive and is not a universal CN rule.
    gap_limit = min(24, len(candidates))
    gaps = []
    for i in range(1, gap_limit):
        before = candidates[i-1].distance_angstrom
        after = candidates[i].distance_angstrom
        gaps.append((after - before) / max(before, 1e-12))
    if gaps:
        estimate = int(np.argmax(gaps)) + 1
        cn_method = "radial_gap_first_max_24 (heuristic; not a validated universal CN method)"
    else:
        estimate = None
        cn_method = "not available"
    rationale = (
        f"Selected the nearest {count} distinct physical positions from a search complete through "
        f"{radius:.6g} A; d{count}={d12:.8g} A and d{count+1}={d13:.8g} A."
    )
    if tie:
        rationale += (
            f" The boundary falls inside a shell of {len(tied)} tied positions; the displayed first-{count} "
            "subset is deterministic preview only and must not be treated as an authoritative geometry."
        )
    if any(n.incompatible_overlap for n in candidates[:max(count, len(tied))]):
        status = "unsupported_input"
        rationale += " A selected position has incompatible overlapping occupancies."
    diag = SelectionDiagnostics(
        selected_count=len(selected),
        selection_mode=f"nearest_{count}_physical_positions",
        status=status,
        shell_distance_angstrom=d12,
        d12_angstrom=d12,
        d13_angstrom=d13,
        boundary_tie_count=len(tied) if tie else 0,
        d12_shell_count=len(tied),
        boundary_shell_neighbor_ids=[n.neighbor_id for n in tied],
        complete_search_radius_angstrom=radius,
        candidate_count=len(candidates),
        cn_estimate=estimate,
        cn_estimate_method=cn_method,
        rationale=rationale,
    )
    return center, selected, diag
