"""Which analyses exist for which coordination number.

Every CN from 4 to 16 gets the coordination-number-agnostic layer: the nearest-N
periodic neighbor search with its tie diagnostics (``cn12.crystal``) and the
convex-hull face/edge reconstruction (``cn12.geometry.reconstruct_hull``), which
are not specific to 12 points. Polyhedral *family* analysis is a plugin per CN;
only CN12 exists today. Planned entries are listed so the interface can show them
as unavailable instead of inventing results.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_CN = 4
MAX_CN = 16


@dataclass(frozen=True)
class FamilyPlugin:
    cn: int
    key: str
    name: str
    status: str  # "available" or "planned"
    families: tuple[str, ...]
    description: str


PLUGINS: dict[int, FamilyPlugin] = {
    12: FamilyPlugin(
        cn=12, key="cn12", name="CN12 plane and polyhedron analysis (cn12 v0.2)",
        status="available",
        families=("cuboctahedron", "anticuboctahedron", "icosahedron", "hexagonal prism"),
        description=("Enumerates the 924 six-neighbor planes through the center, rebuilds "
                     "polygon faces, and screens four reference families by graph matching "
                     "and shape fit. Defined in 00_project_notes/CN12_METHOD_AND_VALIDATION.md."),
    ),
}

# Research targets named in the project plan; nothing is computed for these yet.
PLANNED_FAMILIES: dict[int, tuple[str, ...]] = {
    4: ("tetrahedron", "square plane"),
    6: ("octahedron", "trigonal prism"),
    8: ("cube", "square antiprism"),
    14: ("rhombic dodecahedron", "Frank-Kasper CN14"),
    16: ("Frank-Kasper CN16",),
}


def capability(cn: int) -> dict:
    """Describe what is computed for a requested coordination number."""
    if not MIN_CN <= cn <= MAX_CN:
        return {"cn": cn, "shell": False, "hull": False, "family_analysis": "unsupported",
                "plugin": None, "note": f"CN must be between {MIN_CN} and {MAX_CN}."}
    plugin = PLUGINS.get(cn)
    if plugin is not None and plugin.status == "available":
        return {"cn": cn, "shell": True, "hull": True, "family_analysis": "available",
                "plugin": plugin.key, "plugin_name": plugin.name,
                "families": list(plugin.families), "note": plugin.description}
    planned = PLANNED_FAMILIES.get(cn, ())
    return {"cn": cn, "shell": True, "hull": True, "family_analysis": "planned",
            "plugin": None, "families": list(planned),
            "note": ("Neighbor shell and convex-hull diagnostics only. Family analysis for "
                     f"CN{cn} is planned and not implemented; no family label is produced.")}


def capability_table() -> list[dict]:
    rows = []
    for cn in range(MIN_CN, MAX_CN + 1):
        cap = capability(cn)
        rows.append({
            "CN": cn,
            "neighbor shell": "yes",
            "hull faces/edges": "yes",
            "family analysis": cap["family_analysis"],
            "families": ", ".join(cap["families"]) or "-",
        })
    return rows
