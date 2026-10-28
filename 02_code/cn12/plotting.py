"""PyVista visualization helpers; plotting options never modify analysis geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import pyvista as pv

from .models import AnalysisResult


ELEMENT_COLORS = {
    "Al": "#7f7f7f", "Cu": "#b87333", "Mg": "#8c6bb1", "W": "#4c566a",
    "Ti": "#5b8db8", "V": "#72a0c1", "Fe": "#d55e00", "Ni": "#009e73",
    "Zr": "#cc79a7", "Sc": "#e69f00", "Zn": "#56b4e9",
}
FALLBACK_COLOR = "#999999"
FALLBACK_RADIUS_ANGSTROM = 1.2


def _radius_table() -> dict[str, float]:
    """Return CIFKit's packaged CIF-radius values in angstroms."""
    try:
        from cifkit.sources import radius

        return {element: float(data["CIF"]) for element, data in radius.data().items()}
    except Exception:
        try:
            from cifkit.data.radius import get_radius_data

            return {element: float(data["CIF_radius"])
                    for element, data in get_radius_data().items()}
        except Exception:
            return {}


def _element_color(element: str) -> str:
    if element in ELEMENT_COLORS:
        return ELEMENT_COLORS[element]
    try:
        # The palette index depends on the element's atomic number, not its CIF label.
        from matplotlib import colormaps
        import gemmi

        z = gemmi.Element(element).atomic_number
        palette = colormaps["tab20"]
        return tuple(float(x) for x in palette(z % 20))
    except Exception:
        return FALLBACK_COLOR


def _mixed_color(species_occupancies: tuple[tuple[str, float], ...]) -> str | tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    usable = [(element, max(0.0, occupancy)) for element, occupancy in species_occupancies]
    total = sum(occupancy for _, occupancy in usable)
    if not usable or total <= 0:
        return FALLBACK_COLOR
    rgb = sum(np.asarray(to_rgb(_element_color(element))) * (occupancy / total)
              for element, occupancy in usable)
    return tuple(float(x) for x in rgb)


def _mixed_radius(
    species_occupancies: tuple[tuple[str, float], ...], radius_table: dict[str, float]
) -> tuple[float, bool]:
    usable = [(element, max(0.0, occupancy)) for element, occupancy in species_occupancies]
    total = sum(occupancy for _, occupancy in usable)
    if not usable or total <= 0:
        return FALLBACK_RADIUS_ANGSTROM, True
    missing = any(element not in radius_table for element, _ in usable)
    radius = sum(
        (radius_table.get(element, FALLBACK_RADIUS_ANGSTROM) * occupancy / total)
        for element, occupancy in usable
    )
    return float(radius), missing


def _actor_visibility(actors: dict[str, list]) -> dict[str, bool]:
    return {name: True for name in actors}


def create_plotter(
    result: AnalysisResult,
    sphere_size_factor: float = 0.45,
    show_labels: bool = False,
    off_screen: bool = True,
) -> tuple[pv.Plotter, dict[str, object]]:
    """Create a PyVista scene with chemically colored atoms and separate layers."""
    if len(result.neighbors) != 12:
        raise ValueError("The viewer requires exactly 12 selected neighbor positions")
    if sphere_size_factor <= 0:
        raise ValueError("sphere_size_factor must be positive")
    points = {n.neighbor_id: np.asarray(n.relative_angstrom, dtype=float)
              for n in result.neighbors}
    radii = _radius_table()
    # notebook=False: every view is either written to a file or shown in its own window.
    plotter = pv.Plotter(off_screen=off_screen, notebook=False, window_size=(900, 760))
    plotter.set_background("white")
    actors: dict[str, object] = {
        "atoms": [],
        "coordination_links": [],
        "polyhedron_edges": [],
        "faces": [],
        "labels": [],
        "planes": {},
        "plane_members": {},
        "warnings": [],
    }

    center_species = result.center_species_occupancies
    center_radius, center_missing = _mixed_radius(center_species, radii)
    center_color = _mixed_color(center_species)
    center_mesh = pv.Sphere(radius=1.0, center=(0.0, 0.0, 0.0),
                            theta_resolution=32, phi_resolution=32)
    center_actor = plotter.add_mesh(center_mesh, color=center_color,
                                    smooth_shading=True, specular=0.25)
    center_actor.SetScale(center_radius * sphere_size_factor)
    actors["atoms"].append((center_actor, center_radius))
    if center_missing:
        actors["warnings"].append("A center species is missing CIFKit CIF-radius data; 1.2 A fallback used.")

    all_points = [np.zeros(3)]
    all_labels = [f"CENTER {result.center_label}\n" + _format_species(center_species)]
    for i, neighbor in enumerate(result.neighbors, 1):
        q = points[neighbor.neighbor_id]
        all_points.append(q)
        label_text = ",".join(neighbor.source_labels)
        all_labels.append(f"N{i:02d} {label_text}\n" + _format_species(neighbor.species_occupancies))
        radius, missing = _mixed_radius(neighbor.species_occupancies, radii)
        mesh = pv.Sphere(radius=1.0, center=(0.0, 0.0, 0.0),
                         theta_resolution=24, phi_resolution=24)
        atom_actor = plotter.add_mesh(mesh, color=_mixed_color(neighbor.species_occupancies),
                                      smooth_shading=True, specular=0.25)
        atom_actor.SetPosition(*q)
        atom_actor.SetScale(radius * sphere_size_factor)
        actors["atoms"].append((atom_actor, radius))
        if missing:
            actors["warnings"].append(
                f"{neighbor.neighbor_id}: missing CIFKit CIF-radius data; 1.2 A fallback used."
            )
        line = pv.Line((0, 0, 0), q)
        actors["coordination_links"].append(
            plotter.add_mesh(line, color="#8a8a8a", line_width=1.5, opacity=0.7)
        )

    if result.geometry is not None and result.geometry.hull.success:
        for a, b in result.geometry.hull.edges:
            line = pv.Line(points[a], points[b])
            actors["polyhedron_edges"].append(
                plotter.add_mesh(line, color="#152536", line_width=3.0)
            )
        for face in result.geometry.hull.faces:
            vertices = np.asarray([points[key] for key in face.vertex_ids], dtype=float)
            faces = np.asarray([len(vertices), *range(len(vertices))], dtype=np.int64)
            mesh = pv.PolyData(vertices, faces)
            actors["faces"].append(
                plotter.add_mesh(mesh, color="#49a5a1", opacity=0.22,
                                 show_edges=False, smooth_shading=False)
            )
        for plane in result.geometry.planes:
            normal = np.asarray(plane.normal, dtype=float)
            helper = np.asarray([0.0, 0.0, 1.0])
            if abs(float(np.dot(normal, helper))) > 0.9:
                helper = np.asarray([0.0, 1.0, 0.0])
            u = np.cross(normal, helper)
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            extent = max(np.linalg.norm(p) for p in points.values()) * 1.15
            corners = np.asarray([-extent*u-extent*v, extent*u-extent*v,
                                  extent*u+extent*v, -extent*u+extent*v])
            mesh = pv.PolyData(corners, np.asarray([4, 0, 1, 2, 3]))
            actors["planes"][plane.plane_id] = plotter.add_mesh(
                mesh, color="#e2b544", opacity=0.18, show_edges=False
            )
            ring_points = [points[key] for key in plane.ring_order_ids]
            if ring_points:
                member_mesh = pv.PolyData(np.asarray(ring_points))
                member_spheres = member_mesh.glyph(
                    geom=pv.Sphere(radius=max(extent*0.035, 0.025)),
                    scale=False, orient=False,
                )
                actors["plane_members"][plane.plane_id] = plotter.add_mesh(
                    member_spheres, color="#e04b35", opacity=0.95
                )

    label_actor = plotter.add_point_labels(
        np.asarray(all_points), all_labels, font_size=11, text_color="#111111",
        shape_opacity=0.0, always_visible=True, show_points=False,
    )
    actors["labels"].append(label_actor)
    visibility = _actor_visibility({
        "coordination_links": actors["coordination_links"],
        "polyhedron_edges": actors["polyhedron_edges"],
        "faces": actors["faces"],
        "labels": actors["labels"],
    })
    visibility["labels"] = bool(show_labels)
    for name, visible in visibility.items():
        for actor in actors[name]:
            actor.SetVisibility(visible)
    for actor in actors["planes"].values():
        actor.SetVisibility(False)
    for actor in actors["plane_members"].values():
        actor.SetVisibility(False)
    plotter.add_axes()
    plotter.set_scale(1.0, 1.0, 1.0)
    plotter.view_isometric()
    plotter.reset_camera()
    return plotter, actors


def _format_species(species: tuple[tuple[str, float], ...]) -> str:
    return " ".join(f"{element}:{occupancy:g}" for element, occupancy in species)


def _set_layer(actors: dict[str, object], name: str, visible: bool) -> None:
    for actor in actors[name]:
        actor.SetVisibility(bool(visible))


def show_desktop_window(result: AnalysisResult, sphere_size_factor: float = 0.45,
                        title: str = "CN12 model") -> None:
    """Open the model in its own window, with on-screen layer switches.

    This call returns only when the window is closed, so the notebook starts it
    in a separate Python process (``run_cn12.py view``) rather than calling it.
    Drag to rotate, scroll to zoom; press P to step through the detected planes.
    """
    plotter, actors = create_plotter(result, sphere_size_factor, show_labels=False, off_screen=False)
    _, height = plotter.window_size
    layers = [
        ("coordination_links", "Center-to-neighbor links", True),
        ("polyhedron_edges", "Polyhedron edges", True),
        ("faces", "Faces", True),
        ("labels", "Labels", False),
    ]
    for row, (name, text, shown) in enumerate(layers):
        y = height - 40 - 32 * row

        def toggle(flag, name=name):
            _set_layer(actors, name, flag)
            plotter.render()

        plotter.add_checkbox_button_widget(toggle, value=shown, position=(12, y), size=22,
                                           border_size=2, color_on="#49a5a1", color_off="#d0d0d0")
        plotter.add_text(text, position=(44, y + 2), font_size=9, color="#111111", name=f"layer_{name}")

    plane_ids = list(actors["planes"])
    state = {"index": -1}

    def describe() -> str:
        if not plane_ids:
            return "No plane through the center holds six neighbors"
        current = "none" if state["index"] < 0 else plane_ids[state["index"]]
        return f"Press P to step through the {len(plane_ids)} detected plane(s). Showing: {current}"

    def next_plane() -> None:
        if not plane_ids:
            return
        state["index"] = state["index"] + 1 if state["index"] + 1 < len(plane_ids) else -1
        selected = None if state["index"] < 0 else plane_ids[state["index"]]
        for plane_id in plane_ids:
            actors["planes"][plane_id].SetVisibility(plane_id == selected)
            if plane_id in actors["plane_members"]:
                actors["plane_members"][plane_id].SetVisibility(plane_id == selected)
        plotter.add_text(describe(), position=(12, 12), font_size=9, color="#111111", name="plane_text")
        plotter.render()

    plotter.add_key_event("p", next_plane)
    plotter.add_text(describe(), position=(12, 12), font_size=9, color="#111111", name="plane_text")
    plotter.show(title=title)


def export_interactive_html(result: AnalysisResult, path: str | Path,
                            sphere_size_factor: float = 0.45) -> Path:
    """Write PyVista's standalone VTK.js HTML viewer for rotation and zoom."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    # HTML export only needs the scene graph, but PyVista starts a Trame server
    # to produce it. Starting that server inside a notebook kernel ended the kernel
    # in VS Code on this project's Windows setup (checked 2026-10-11), so the
    # notebook runs every export through run_cn12.py in a separate process.
    plotter, _ = create_plotter(result, sphere_size_factor, off_screen=True)
    plotter.export_html(str(output))
    plotter.close()
    return output


def save_static_png(result: AnalysisResult, path: str | Path,
                    sphere_size_factor: float = 0.45) -> Path:
    """Save a static PyVista screenshot; raise instead of silently substituting a plot."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter, _ = create_plotter(result, sphere_size_factor, show_labels=False, off_screen=True)
    plotter.show(interactive=False, auto_close=False)
    plotter.screenshot(str(output), transparent_background=False)
    plotter.close()
    return output
