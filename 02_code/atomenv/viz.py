"""Plotly figures for the app. Display options never change analysis values.

PyVista (used by ``cn12.plotting`` for desktop windows and saved HTML/PNG) is not
embedded: an earlier attempt to run its Trame server inside a notebook kernel
ended the kernel on the project computer (see the CN12 method note). Plotly
renders in any browser, including phones, so the in-app 3D views use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import plotly.graph_objects as go

from .api import PlaneExtension, SiteAnalysis

# Colour-blind-aware defaults for common elements; others get a deterministic palette slot.
ELEMENT_COLORS = {
    "Cu": "#c27a3a", "Mg": "#8c6bb1", "W": "#4c566a", "Al": "#8f9aa6", "Ti": "#5b8db8",
    "V": "#72a0c1", "Fe": "#d55e00", "Ni": "#009e73", "Zr": "#cc79a7", "Na": "#e69f00",
    "Cl": "#2a9d4b", "O": "#d62728", "Zn": "#56b4e9", "Si": "#b0a46a", "Au": "#d4a017",
}
PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#edc948",
           "#b07aa1", "#ff9da7", "#9c755f", "#7f7f7f"]
CENTER_COLOR = "#e4572e"
FALLBACK_RADIUS = 1.2

# Visual styles. "Classic" reproduces the colours of the CN12 PyVista program (cn12.plotting).
STYLES = {
    "Standard": dict(edge="#374151", edge_width=3, face="#9ca3af", face_opacity=0.15, plane="#2b6cb0",
                     plane_opacity=0.35, member="#2b6cb0", contact="#9ca3af", bg="#ffffff", center=CENTER_COLOR,
                     outline="#1f2937"),
    "Classic (CN12 program)": dict(edge="#152536", edge_width=4, face="#49a5a1", face_opacity=0.22,
                                   plane="#e2b544", plane_opacity=0.30, member="#e04b35", contact="#8a8a8a",
                                   bg="#ffffff", center=None, outline="#152536"),
    "Dark": dict(edge="#e5e7eb", edge_width=3, face="#60a5fa", face_opacity=0.18, plane="#fbbf24",
                 plane_opacity=0.30, member="#f87171", contact="#6b7280", bg="#0f172a", center="#f97316",
                 outline="#e5e7eb"),
}


def element_color(element: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and element in overrides:
        return overrides[element]
    if element in ELEMENT_COLORS:
        return ELEMENT_COLORS[element]
    try:
        import gemmi

        return PALETTE[gemmi.Element(element).atomic_number % len(PALETTE)]
    except Exception:
        return "#999999"


def _radius_table() -> dict[str, float]:
    try:
        from cifkit.sources import radius

        return {e: float(d["CIF"]) for e, d in radius.data().items()}
    except Exception:
        return {}


_RADII = _radius_table()


def species_radius(species) -> float:
    total = sum(max(0.0, o) for _, o in species) or 1.0
    return float(sum(_RADII.get(e, FALLBACK_RADIUS) * max(0.0, o) / total for e, o in species)) or FALLBACK_RADIUS


def majority_element(species) -> str:
    return max(species, key=lambda item: item[1])[0] if species else "?"


def species_text(species) -> str:
    return " + ".join(f"{e} {o:g}" for e, o in species)


def _hex_rgb(color: str) -> np.ndarray:
    c = color.lstrip("#")
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def species_color(species, options: "DisplayOptions") -> str:
    """Majority-element colour, or occupancy-weighted blend (as in cn12.plotting) when enabled."""
    if not options.blend_mixed or len(species) <= 1:
        return element_color(majority_element(species), options.color_overrides)
    total = sum(max(0.0, o) for _, o in species) or 1.0
    rgb = sum(_hex_rgb(element_color(e, options.color_overrides)) * max(0.0, o) / total for e, o in species)
    return "#%02x%02x%02x" % tuple(int(round(x)) for x in rgb)


@dataclass
class DisplayOptions:
    atom_scale: float = 1.0
    atom_opacity: float = 0.95
    color_overrides: dict[str, str] = field(default_factory=dict)
    show_labels: bool = False
    show_contacts: bool = True
    show_edges: bool = True
    show_faces: bool = False
    face_opacity: float | None = None
    plane_color: str | None = None
    plane_opacity: float | None = None
    show_cell: bool = False
    show_extended: bool = False
    height: int = 600
    style: str = "Standard"
    blend_mixed: bool = True
    show_axes: bool = True
    show_legend: bool = True
    orthographic: bool = False

    @property
    def s(self) -> dict:
        base = dict(STYLES.get(self.style, STYLES["Standard"]))
        if self.face_opacity is not None:
            base["face_opacity"] = self.face_opacity
        if self.plane_color:
            base["plane"] = self.plane_color
        if self.plane_opacity is not None:
            base["plane_opacity"] = self.plane_opacity
        return base


def plot_config(scroll_zoom: bool = False, filename: str = "atomic_environment") -> dict:
    """Plotly config. Scroll-wheel zoom is off by default so the page scrolls past 3D models."""
    return {
        "displaylogo": False,
        "responsive": True,
        "scrollZoom": scroll_zoom,
        "toImageButtonOptions": {"format": "png", "scale": 3, "filename": filename},
        "modeBarButtonsToRemove": ["resetCameraLastSave3d"],
    }


PLOT_CONFIG = plot_config()


def _layout(fig: go.Figure, title: str, options: DisplayOptions, axis_note: str) -> go.Figure:
    s = options.s
    dark = s["bg"] != "#ffffff"
    fg = "#e5e7eb" if dark else "#111827"
    axis = dict(showbackground=False, gridcolor="#334155" if dark else "#e5e7eb",
                zerolinecolor="#475569" if dark else "#d1d5db", title_font_size=11, color=fg,
                visible=options.show_axes)
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=15, color=fg)),
        height=options.height, margin=dict(l=0, r=0, t=48, b=0),
        paper_bgcolor=s["bg"], plot_bgcolor=s["bg"], font=dict(color=fg),
        showlegend=options.show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=0.0, x=0.0,
                    bgcolor="rgba(15,23,42,0.6)" if dark else "rgba(255,255,255,0.8)", font=dict(size=11)),
        scene=dict(aspectmode="data", bgcolor=s["bg"],
                   xaxis=dict(axis, title=f"x {axis_note}"), yaxis=dict(axis, title=f"y {axis_note}"),
                   zaxis=dict(axis, title=f"z {axis_note}"),
                   camera=dict(eye=dict(x=1.3, y=1.15, z=0.75),
                               projection=dict(type="orthographic" if options.orthographic else "perspective"))),
        uirevision="keep",  # keep the camera when options change
    )
    return fig


def _marker_size(species, options: DisplayOptions, base: float = 16.0) -> float:
    return base * options.atom_scale * species_radius(species) / 1.3


def _group_atoms(points, species_list, labels, hover, name_prefix, options, symbol, size_factor=1.0,
                 opacity=None, legendgroup=None, outline=None, showlegend=True):
    traces = []
    by_color: dict[tuple[str, str], list[int]] = {}
    for i, sp in enumerate(species_list):
        by_color.setdefault((majority_element(sp), species_color(sp, options)), []).append(i)
    for (element, color), idx in sorted(by_color.items()):
        pts = np.asarray([points[i] for i in idx], dtype=float)
        mixed = any(len(species_list[i]) > 1 for i in idx)
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers+text" if options.show_labels else "markers",
            text=[labels[i] for i in idx] if options.show_labels else None,
            textposition="top center", textfont=dict(size=10),
            hovertext=[hover[i] for i in idx], hoverinfo="text",
            name=f"{name_prefix}: {element}{' (mixed)' if mixed and options.blend_mixed else ''}",
            legendgroup=legendgroup or name_prefix, showlegend=showlegend,
            marker=dict(size=[_marker_size(species_list[i], options) * size_factor for i in idx],
                        color=color, symbol=symbol,
                        opacity=options.atom_opacity if opacity is None else opacity,
                        line=dict(color=outline or options.s["outline"], width=1)),
        ))
    return traces


def _ordered_polygon(points: np.ndarray, normal: np.ndarray | None = None) -> np.ndarray:
    """Order coplanar points by angle around their centroid (display only)."""
    c = points.mean(axis=0)
    if normal is None:
        _, _, vt = np.linalg.svd(points - c)
        normal = vt[-1]
    u = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(normal, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    angles = np.arctan2((points - c) @ v, (points - c) @ u)
    return points[np.argsort(angles)]


def _fan_mesh(polygon: np.ndarray, color: str, opacity: float, name: str, hover: str,
              showlegend: bool, legendgroup: str) -> go.Mesh3d:
    n = len(polygon)
    c = polygon.mean(axis=0)
    verts = np.vstack([polygon, c])
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=[n] * n, j=list(range(n)),
                     k=[(t + 1) % n for t in range(n)], color=color, opacity=opacity, name=name,
                     hovertext=hover, hoverinfo="text", flatshading=True, showlegend=showlegend,
                     legendgroup=legendgroup)


def _center_trace(analysis: SiteAnalysis, options: DisplayOptions) -> go.Scatter3d:
    center_species = tuple(sorted((c.element, c.occupancy) for c in analysis.center.components))
    color = options.s["center"] or species_color(center_species, options)
    return go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers+text" if options.show_labels else "markers",
        text=[analysis.center.display_label] if options.show_labels else None,
        hovertext=[f"Center {analysis.center.display_label} ({species_text(center_species)})"],
        hoverinfo="text", name=f"Center: {analysis.center.display_label}",
        marker=dict(size=_marker_size(center_species, options) * 1.15, color=color,
                    symbol="diamond" if options.s["center"] else "circle",
                    line=dict(color=options.s["outline"], width=3), opacity=1.0),
    )


def environment_figure(analysis: SiteAnalysis, options: DisplayOptions,
                       plane_ids: list[str] | None = None, extended=None,
                       isolate_planes: bool = False) -> go.Figure:
    """Local environment, centered on the selected site (coordinates relative to it).

    ``isolate_planes`` hides neighbors that are not members of the drawn planes.
    """
    s = options.s
    fig = go.Figure()
    neighbors = analysis.neighbors
    rel = np.asarray([n.relative_angstrom for n in neighbors], dtype=float) if neighbors else np.zeros((0, 3))
    index_of = {n.neighbor_id: i for i, n in enumerate(neighbors)}
    boundary = set(analysis.selection.boundary_shell_neighbor_ids)
    authoritative = analysis.selection.status == "unambiguous"
    planes = ({p.plane_id: p for p in analysis.cn12.geometry.planes}
              if analysis.cn12 is not None and analysis.cn12.geometry is not None else {})
    drawn = [planes[p] for p in (plane_ids or []) if p in planes]
    plane_members = set().union(*[set(p.member_ids) for p in drawn]) if drawn else set()

    if extended:
        selected = set(index_of)
        others = [n for n in extended if n.neighbor_id not in selected]
        if others:
            fig.add_traces(_group_atoms(
                [n.relative_angstrom for n in others], [n.species_occupancies for n in others],
                ["+".join(n.source_labels) for n in others],
                [f"{'+'.join(n.source_labels)} ({species_text(n.species_occupancies)})<br>"
                 f"d = {n.distance_angstrom:.4f} Å<br>image {n.image_translation}" for n in others],
                "Region", options, "circle", size_factor=0.7, opacity=min(0.35, options.atom_opacity),
                outline="#9ca3af"))

    if options.show_faces and analysis.hull is not None and not isolate_planes:
        for f_i, face in enumerate(analysis.hull.faces):
            pts = np.asarray([rel[index_of[v]] for v in face.vertex_ids if v in index_of])
            if len(pts) >= 3:
                fig.add_trace(_fan_mesh(_ordered_polygon(pts, np.asarray(face.normal)), s["face"],
                                        s["face_opacity"], "Polyhedron faces",
                                        f"{face.face_id}: {face.polygon_size}-gon",
                                        showlegend=(f_i == 0), legendgroup="faces"))

    if options.show_edges and analysis.hull is not None and not isolate_planes:
        xs, ys, zs = [], [], []
        for a, b in analysis.hull.edges:
            if a in index_of and b in index_of:
                for p in (rel[index_of[a]], rel[index_of[b]]):
                    xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
                xs.append(None); ys.append(None); zs.append(None)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                   line=dict(color=s["edge"], width=s["edge_width"]),
                                   name="Polyhedron edges", hoverinfo="skip"))

    if options.show_contacts and len(rel):
        xs, ys, zs = [], [], []
        for i, p in enumerate(rel):
            if isolate_planes and neighbors[i].neighbor_id not in plane_members:
                continue
            xs += [0, p[0], None]; ys += [0, p[1], None]; zs += [0, p[2], None]
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                   line=dict(color=s["contact"], width=2, dash="dot"),
                                   name="Center–neighbor lines", hoverinfo="skip"))

    for plane in drawn:
        members = np.asarray([rel[index_of[m]] for m in plane.member_ids if m in index_of])
        if len(members) >= 3:
            poly = _ordered_polygon(members, np.asarray(plane.normal))
            fig.add_trace(_fan_mesh(poly, s["plane"], s["plane_opacity"], f"Plane {plane.plane_id}",
                                    f"Plane {plane.plane_id}: {plane.member_count} members; "
                                    f"max residual {plane.max_fit_residual_angstrom:.2e} Å",
                                    showlegend=True, legendgroup=f"plane{plane.plane_id}"))
            ring = np.vstack([poly, poly[:1]])
            fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="lines",
                                       line=dict(color=s["plane"], width=5), hoverinfo="skip",
                                       showlegend=False, legendgroup=f"plane{plane.plane_id}"))

    if len(rel):
        hover, keep = [], []
        for i, n in enumerate(neighbors):
            if isolate_planes and n.neighbor_id not in plane_members:
                continue
            keep.append(i)
            flags = []
            if n.neighbor_id in boundary:
                flags.append("in boundary shell")
            if n.mixed_occupancy:
                flags.append("mixed/partial occupancy")
            hover.append(f"#{i+1} {'+'.join(n.source_labels)} ({species_text(n.species_occupancies)})<br>"
                         f"d = {n.distance_angstrom:.5f} Å<br>image {n.image_translation}"
                         + (f"<br>{', '.join(flags)}" if flags else ""))
        if keep:
            name = "Neighbors" if authoritative else "Neighbors (shell not unique)"
            fig.add_traces(_group_atoms(rel[keep], [neighbors[i].species_occupancies for i in keep],
                                        [f"{i+1}" for i in keep], hover, name, options,
                                        "circle" if authoritative else "circle-open"))

    fig.add_trace(_center_trace(analysis, options))
    title = f"{analysis.center.display_label} — nearest {analysis.cn}"
    return _layout(fig, title, options, "− x_center (Å)")


def plane_extension_figure(analysis: SiteAnalysis, ext: PlaneExtension, options: DisplayOptions,
                           context: list | None = None) -> go.Figure:
    """The isolated plane: center, its in-plane CN shell members and the next in-plane atoms (3D)."""
    s = options.s
    fig = go.Figure()
    if context:
        on = {a.neighbor_id for a in ext.atoms}
        rest = [n for n in context if n.neighbor_id not in on]
        if rest:
            fig.add_traces(_group_atoms(
                [n.relative_angstrom for n in rest], [n.species_occupancies for n in rest],
                ["" for _ in rest], [f"off-plane {'+'.join(n.source_labels)}" for n in rest],
                "Off-plane", options, "circle", size_factor=0.55, opacity=0.12, outline="#9ca3af"))
    rel = np.asarray([a.relative_angstrom for a in ext.atoms], dtype=float).reshape(-1, 3)
    if len(rel):
        span = float(np.max(np.linalg.norm(rel, axis=1))) * 1.05
        n = np.asarray(ext.normal)
        u = rel[0] - (rel[0] @ n) * n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        ang = np.linspace(0, 2 * np.pi, 49)
        disk = np.array([span * (np.cos(t) * u + np.sin(t) * v) for t in ang])
        fig.add_trace(_fan_mesh(disk[:-1], s["plane"], s["plane_opacity"] * 0.5, f"Plane {ext.plane_id}",
                                f"Plane {ext.plane_id}", True, "plane"))
    shell = [i for i, a in enumerate(ext.atoms) if a.neighbor_id in ext.shell_member_ids]
    outer = [i for i, a in enumerate(ext.atoms) if a.neighbor_id not in ext.shell_member_ids]
    if shell:
        ring = _ordered_polygon(rel[shell], np.asarray(ext.normal))
        ring = np.vstack([ring, ring[:1]])
        fig.add_trace(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="lines",
                                   line=dict(color=s["member"], width=6), name="CN12 hexagon",
                                   hoverinfo="skip"))
    for idx, name, factor in ((shell, "Hexagon members", 1.0), (outer, "In-plane atoms", 0.8)):
        if idx:
            atoms = [ext.atoms[i] for i in idx]
            fig.add_traces(_group_atoms(
                rel[idx], [a.species_occupancies for a in atoms], [f"{i+1}" for i in idx],
                [f"#{i+1} {'+'.join(a.source_labels)} ({species_text(a.species_occupancies)})<br>"
                 f"d = {a.distance_angstrom:.4f} Å<br>off-plane {ext.off_plane_angstrom[i]:+.1e} Å"
                 for i, a in zip(idx, atoms)], name, options, "circle", size_factor=factor))
    fig.add_trace(_center_trace(analysis, options))
    _layout(fig, f"Plane {ext.plane_id} — {len(ext.atoms)} in-plane positions", options, "(Å)")
    n = np.asarray(ext.normal) * 1.6 + np.array([0.12, 0.08, 0.05])  # near face-on, slight tilt for depth
    fig.update_layout(scene_camera_eye=dict(x=n[0], y=n[1], z=n[2]), uirevision=f"plane-{ext.plane_id}")
    return fig


def plane_map_figure(analysis: SiteAnalysis, ext: PlaneExtension, options: DisplayOptions,
                     shell_circles: list[float] | None = None) -> go.Figure:
    """Face-on 2D map of the plane (in-plane coordinates relative to the center)."""
    s = options.s
    dark = s["bg"] != "#ffffff"
    fig = go.Figure()
    xy = ext.in_plane_xy
    for r in shell_circles or []:
        t = np.linspace(0, 2 * np.pi, 121)
        fig.add_trace(go.Scatter(x=r * np.cos(t), y=r * np.sin(t), mode="lines", hoverinfo="skip",
                                 line=dict(color="#94a3b8", width=1, dash="dot"), showlegend=False))
    shell = [i for i, a in enumerate(ext.atoms) if a.neighbor_id in ext.shell_member_ids]
    if shell:
        ring = xy[shell]
        ring = ring[np.argsort(np.arctan2(ring[:, 1], ring[:, 0]))]
        ring = np.vstack([ring, ring[:1]])
        fig.add_trace(go.Scatter(x=ring[:, 0], y=ring[:, 1], mode="lines", name="CN12 hexagon",
                                 line=dict(color=s["member"], width=3), hoverinfo="skip"))
    groups: dict[tuple[str, str], list[int]] = {}
    for i, a in enumerate(ext.atoms):
        groups.setdefault((majority_element(a.species_occupancies), species_color(a.species_occupancies, options)),
                          []).append(i)
    for (el, color), idx in sorted(groups.items()):
        fig.add_trace(go.Scatter(
            x=xy[idx, 0], y=xy[idx, 1], mode="markers+text" if options.show_labels else "markers",
            text=[str(i + 1) for i in idx] if options.show_labels else None, textposition="top center",
            name=el, marker=dict(size=[_marker_size(ext.atoms[i].species_occupancies, options, 13) for i in idx],
                                 color=color, opacity=options.atom_opacity,
                                 line=dict(color=[s["member"] if ext.atoms[i].neighbor_id in ext.shell_member_ids
                                                  else s["outline"] for i in idx],
                                           width=[3 if ext.atoms[i].neighbor_id in ext.shell_member_ids else 1
                                                  for i in idx])),
            hovertext=[f"#{i+1} {'+'.join(ext.atoms[i].source_labels)}<br>d = {ext.atoms[i].distance_angstrom:.4f} Å"
                       for i in idx], hoverinfo="text"))
    center_species = tuple(sorted((c.element, c.occupancy) for c in analysis.center.components))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name=f"Center {analysis.center.display_label}",
                             marker=dict(size=_marker_size(center_species, options, 15),
                                         color=s["center"] or species_color(center_species, options),
                                         symbol="diamond", line=dict(color=s["outline"], width=2))))
    fg = "#e5e7eb" if dark else "#111827"
    grid = "#334155" if dark else "#eef2f7"
    fig.update_layout(height=options.height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor=s["bg"], plot_bgcolor=s["bg"], font=dict(color=fg),
                      showlegend=options.show_legend,
                      title=dict(text=f"Plane {ext.plane_id}, face-on (Å)", x=0.01, font=dict(size=14)),
                      xaxis=dict(gridcolor=grid, zeroline=False, visible=options.show_axes),
                      yaxis=dict(gridcolor=grid, zeroline=False, scaleanchor="x", scaleratio=1,
                                 visible=options.show_axes),
                      legend=dict(orientation="h", y=-0.05), uirevision=f"map-{ext.plane_id}")
    return fig


def unit_cell_figure(block, positions: list[dict], options: DisplayOptions,
                     highlight_position: str | None = None) -> go.Figure:
    s = options.s
    fig = go.Figure()
    lattice = np.asarray(block.lattice_columns_angstrom, dtype=float)
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
    xs, ys, zs = [], [], []
    for a in range(8):
        for b in range(a + 1, 8):
            if np.sum(np.abs(corners[a] - corners[b])) == 1:
                for c in (corners[a], corners[b]):
                    p = lattice @ c
                    xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
                xs.append(None); ys.append(None); zs.append(None)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=s["edge"], width=3),
                               name="Unit cell", hoverinfo="skip"))
    for axis_i, name in enumerate("abc"):
        tip = lattice[:, axis_i]
        fig.add_trace(go.Scatter3d(x=[tip[0]], y=[tip[1]], z=[tip[2]], mode="text", text=[name],
                                   textfont=dict(size=14, color=s["edge"]), showlegend=False,
                                   hoverinfo="skip"))
    inside = [p for p in positions if not p["boundary_image"]]
    images = [p for p in positions if p["boundary_image"]]
    for group, label, factor in ((inside, "Sites", 1.0), (images, "Cell-edge images", 0.8)):
        if not group:
            continue
        fig.add_traces(_group_atoms(
            [p["cartesian"] for p in group], [p["species"] for p in group],
            [p["labels"] for p in group],
            [f"{p['labels']} ({species_text(p['species'])})<br>frac ({p['fractional'][0]:.4f}, "
             f"{p['fractional'][1]:.4f}, {p['fractional'][2]:.4f})" for p in group],
            label, options, "circle", size_factor=factor,
            opacity=options.atom_opacity if group is inside else options.atom_opacity * 0.6))
    if highlight_position:
        hi = [p for p in inside if p["position_id"] == highlight_position]
        if hi:
            p = np.asarray([h["cartesian"] for h in hi])
            fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                                       marker=dict(size=18 * options.atom_scale, symbol="diamond-open",
                                                   color=CENTER_COLOR, line=dict(width=4, color=CENTER_COLOR)),
                                       name="Selected center", hoverinfo="skip"))
    return _layout(fig, f"Unit cell — {block.formula}", options, "(Å)")


def distance_figure(analysis: SiteAnalysis, extra: int = 6, options: DisplayOptions | None = None) -> go.Figure:
    """Neighbor distance by rank, with the selection boundary marked."""
    from cn12.crystal import _generate_neighbors_within_radius  # reuse the same complete search

    options = options or DisplayOptions()
    s = options.s
    block, center = analysis.block, analysis.center
    physical = next(p for p in block.physical_sites if p.position_id == center.position_id)
    candidates = _generate_neighbors_within_radius(
        block, center, physical, analysis.selection.complete_search_radius_angstrom,
        max_translation_shell=analysis.settings.max_translation_shell)[: analysis.cn + extra]
    ranks = list(range(1, len(candidates) + 1))
    d = [n.distance_angstrom for n in candidates]
    colors = ["#2b6cb0" if r <= analysis.cn else "#cbd5e1" for r in ranks]
    fig = go.Figure(go.Bar(x=ranks, y=d, marker_color=colors,
                           hovertext=[f"#{r} {'+'.join(n.source_labels)}: {n.distance_angstrom:.5f} Å"
                                      for r, n in zip(ranks, candidates)], hoverinfo="text"))
    fig.add_vline(x=analysis.cn + 0.5, line_dash="dash", line_color=CENTER_COLOR,
                  annotation_text=f"CN{analysis.cn} cutoff", annotation_position="top")
    lo = min(d) * 0.9 if d else 0
    dark = s["bg"] != "#ffffff"
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor=s["bg"],
                      plot_bgcolor=s["bg"], showlegend=False, font=dict(color="#e5e7eb" if dark else "#111827"),
                      xaxis=dict(title="neighbor rank", dtick=1, gridcolor="#f1f5f9"),
                      yaxis=dict(title="distance (Å)", range=[lo, max(d) * 1.02 if d else 1],
                                 gridcolor="#e5e7eb"))
    return fig
