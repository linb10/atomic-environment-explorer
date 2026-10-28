"""The notebook control panel: choose a CIF, choose a center, read the result.

Nothing here does any geometry. It calls the same functions as the command
line and only arranges their output, so a result shown in the notebook is the
result the rest of the program would give.

    from cn12.notebook_ui import launch
    app = launch()        # shows the panel
    app.result            # the AnalysisResult currently on screen
"""

from __future__ import annotations

from dataclasses import asdict
from html import escape
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import HTML, clear_output, display

from .crystal import parse_cif_blocks
from .export import export_batch_summary
from .files import find_project_root, list_cif_files, open_in_default_app, pick_cif_files
from .models import AnalysisResult, CrystalBlock
from .pipeline import (
    AnalysisSettings,
    analyze_batch,
    analyze_parsed_block,
    describe_result,
    plain_verdict,
)


def launch(
    project_root: str | Path | None = None,
    settings: AnalysisSettings = AnalysisSettings(),
    start_file: str | Path | None = None,
    sphere_size_factor: float = 0.45,
) -> "Cn12App":
    """Build the panel, show it, and analyze the first file straight away."""
    app = Cn12App(project_root, settings, start_file, sphere_size_factor)
    app.show()
    return app


class Cn12App:
    TAB_TITLES = ["Overview", "Neighbors", "3D model", "Planes", "Faces and edges",
                  "Settings", "Whole project"]

    def __init__(
        self,
        project_root: str | Path | None = None,
        settings: AnalysisSettings = AnalysisSettings(),
        start_file: str | Path | None = None,
        sphere_size_factor: float = 0.45,
    ) -> None:
        self.project_root = Path(project_root).resolve() if project_root else find_project_root()
        self.output_root = self.project_root / "04_results"
        self.settings = settings
        self.sphere_size_factor = sphere_size_factor

        # What is currently loaded. These are safe to read from other cells.
        self.file: Path | None = None
        self.blocks: list[CrystalBlock] = []
        self.results: dict[str, AnalysisResult] = {}   # every center site of the file
        self.result: AnalysisResult | None = None      # the center site on screen
        self.batch: dict | None = None                 # filled by the Whole project tab
        self.last_saved_folder: Path | None = None
        self._current_key = ""
        self._busy = False
        self._preview_folder: Path | None = None

        files = list_cif_files(self.project_root)
        if start_file is not None:
            start = Path(start_file).resolve()
            if start not in files:
                files.insert(0, start)
        else:
            start = files[0] if files else None
        self._file_options = [(self._file_label(path), str(path)) for path in files]

        self.file_dropdown = widgets.Dropdown(
            options=self._file_options, value=str(start) if start else None,
            description="CIF file", layout=widgets.Layout(width="620px"),
        )
        self.browse_button = widgets.Button(description="Browse...", icon="folder-open",
                                            tooltip="Choose any CIF file on this computer")
        self.site_dropdown = widgets.Dropdown(options=[], description="Center",
                                              layout=widgets.Layout(width="620px"))
        self.site_note = widgets.HTML()
        self.headline = widgets.HTML()
        self.save_button = widgets.Button(description="Save this result", icon="save",
                                          button_style="primary", layout=widgets.Layout(width="auto"),
                                          tooltip="Write tables, 3D model and picture to 04_results")
        self.folder_button = widgets.Button(description="Open results folder", icon="folder-open",
                                            layout=widgets.Layout(width="auto"),
                                            tooltip="Show 04_results in the file explorer")
        self.save_note = widgets.HTML()
        self.outputs = [widgets.Output() for _ in self.TAB_TITLES]
        self.tabs = widgets.Tab(children=self.outputs)
        for index, title in enumerate(self.TAB_TITLES):
            self.tabs.set_title(index, title)

        self.file_dropdown.observe(self._file_changed, names="value")
        self.site_dropdown.observe(self._site_changed, names="value")
        self.browse_button.on_click(self._browse)
        self.save_button.on_click(self._save)
        self.folder_button.on_click(
            lambda _button: open_in_default_app(self.last_saved_folder or self.output_root))

        self.panel = widgets.VBox([
            widgets.HBox([self.file_dropdown, self.browse_button]),
            widgets.HBox([self.site_dropdown, self.site_note]),
            self.headline,
            widgets.HBox([self.save_button, self.folder_button, self.save_note]),
            self.tabs,
        ])

    # ------------------------------------------------------------------ setup
    def show(self) -> None:
        display(self.panel)
        self._render_settings()
        self._render_project_tab()
        if self.file_dropdown.value:
            self.load_file(self.file_dropdown.value)
        else:
            self.headline.value = _message("No CIF files were found in the project. Use Browse.", "warn")

    def _file_label(self, path: Path) -> str:
        try:
            folder = path.parent.relative_to(self.project_root).as_posix()
        except ValueError:
            folder = str(path.parent)
        return f"{path.name}   ({folder})"

    # --------------------------------------------------------------- handlers
    def _browse(self, _button=None) -> None:
        try:
            start = self.file.parent if self.file else self.project_root / "03_data" / "cif_samples"
            chosen = pick_cif_files(start, multiple=False)
        except RuntimeError as exc:
            self.headline.value = _message(str(exc), "warn")
            return
        if not chosen:
            return
        path = chosen[0].resolve()
        if str(path) not in [value for _, value in self._file_options]:
            self._file_options = [(self._file_label(path), str(path)), *self._file_options]
            self._busy = True
            try:
                self.file_dropdown.options = self._file_options
            finally:
                self._busy = False
        if self.file_dropdown.value == str(path):
            self.load_file(path)          # same file chosen again: reload it
        else:
            self.file_dropdown.value = str(path)

    def _file_changed(self, change) -> None:
        if not self._busy and change["new"]:
            self.load_file(change["new"])

    def _site_changed(self, change) -> None:
        if not self._busy and change["new"]:
            self.select_site(change["new"])

    # ------------------------------------------------------------------ logic
    def load_file(self, path: str | Path) -> None:
        """Read a CIF, analyze every center site in it, and show the first."""
        self.file = Path(path)
        self.results, self.result, self.blocks = {}, None, []
        self.save_note.value = ""
        try:
            self.blocks = parse_cif_blocks(self.file)
            for block in self.blocks:
                for center in block.center_sites:
                    key = self._site_key(block, center.site_id)
                    self.results[key] = analyze_parsed_block(block, self.file, center.site_id, self.settings)
        except Exception as exc:
            self._set_site_options([])
            self.site_note.value = ""
            self.save_button.disabled = True
            self.headline.value = _message(f"{self.file.name} could not be analyzed: {exc}", "error")
            for output in self.outputs[:5]:
                with output:
                    clear_output()
            return

        options = []
        for block in self.blocks:
            for center in block.center_sites:
                prefix = f"{block.block_name} / " if len(self.blocks) > 1 else ""
                options.append((f"{prefix}{center.display_label}   (x{center.symmetry_multiplicity} in the cell)",
                                self._site_key(block, center.site_id)))
        self._set_site_options(options)
        count = len(options)
        self.site_note.value = (
            "<span style='color:#666'>the only distinct site in this file</span>" if count == 1 else
            f"<span style='color:#666'>{count} distinct sites; all analyzed, see Overview</span>"
        )
        if options:
            self.select_site(options[0][1])

    def select_site(self, key: str) -> None:
        self._current_key = key
        self.result = self.results[key]
        self.save_note.value = ""
        self.save_button.disabled = False
        verdict = plain_verdict(self.result)
        tone = "ok" if self.result.status == "complete" else "warn"
        self.headline.value = _message(
            f"{self.file.name}, center {self.result.center_label}: {verdict}", tone, big=True)
        self._render_overview()
        self._render_neighbors()
        self._render_model_tab()
        self._render_planes()
        self._render_faces()

    @staticmethod
    def _site_key(block: CrystalBlock, site_id: str) -> str:
        return f"{block.block_name}\x1f{site_id}"

    def _set_site_options(self, options: list[tuple[str, str]]) -> None:
        self._busy = True
        try:
            self.site_dropdown.options = options
            self.site_dropdown.value = options[0][1] if options else None
            self.site_dropdown.disabled = len(options) <= 1
        finally:
            self._busy = False

    # -------------------------------------------------------------- rendering
    def _render_overview(self) -> None:
        result = self.result
        with self.outputs[0]:
            clear_output(wait=True)
            block = next(b for b in self.blocks if b.block_name == result.block_name)
            a, b, c = block.cell_lengths_angstrom
            alpha, beta, gamma = block.cell_angles_degrees
            display(HTML(
                f"<b>{escape(block.formula)}</b> &nbsp; space group {escape(block.space_group)}"
                f" (no. {block.space_group_number}) &nbsp; cell {a:g} x {b:g} x {c:g} &Aring;,"
                f" angles {alpha:g}, {beta:g}, {gamma:g}&deg; &nbsp; data block {escape(block.block_name)}"
            ))
            rows = []
            for item in self.results.values():
                summary = describe_result(item)
                rows.append({
                    "center": summary["center"],
                    "result": plain_verdict(item),
                    "central hexagonal planes": summary["hexagonal_planes"],
                    "faces": summary["faces"],
                    "12th neighbor (A)": summary["d12_angstrom"],
                    "13th neighbor (A)": summary["d13_angstrom"],
                    "CN estimate (heuristic)": item.selection.cn_estimate,
                })
            display(_table(rows))
            geometry = result.geometry
            lines = [f"Selected center: {result.center_label}",
                     "  species: " + ", ".join(f"{el} {occ:g}" for el, occ in result.center_species_occupancies),
                     f"  neighbor selection: {result.selection.rationale}"]
            if geometry is not None:
                lines.append(
                    f"  distortion: radial spread {_number(geometry.center_distance_spread)}, "
                    f"edge spread {_number(geometry.edge_length_spread)}, "
                    f"shape RMSD {_number(geometry.normalized_shape_rmsd)}"
                )
                lines.append("  why this label:")
                lines.extend(f"    - {reason}" for reason in geometry.classification_reasons)
            if result.warnings:
                lines.append("  notes:")
                lines.extend(f"    - {warning}" for warning in result.warnings)
            print("\n".join(lines))

    def _render_neighbors(self) -> None:
        result = self.result
        with self.outputs[1]:
            clear_output(wait=True)
            if result.status != "complete":
                print("These are the 12 nearest positions for inspection only; "
                      "the result was not classified (see the headline).")
            rows = []
            for rank, neighbor in enumerate(result.neighbors, 1):
                x, y, z = neighbor.relative_angstrom
                rows.append({
                    "rank": rank,
                    "label": ", ".join(neighbor.source_labels),
                    "species : occupancy": ", ".join(f"{el}:{occ:g}" for el, occ in neighbor.species_occupancies),
                    "distance (A)": neighbor.distance_angstrom,
                    "x from center (A)": x, "y from center (A)": y, "z from center (A)": z,
                    "cell offset": neighbor.image_translation,
                    "mixed site": neighbor.mixed_occupancy,
                })
            display(_table(rows))
            print("Full precision, fractional coordinates and symmetry IDs are written to "
                  "neighbors.csv when you save the result.")

    def _render_model_tab(self) -> None:
        result = self.result
        with self.outputs[2]:
            clear_output(wait=True)
            if result.geometry is None or len(result.neighbors) != 12:
                print("No 3D model: this center has no unique set of 12 neighbors.")
                return
            browser_button = widgets.Button(description="Open 3D model in browser", icon="external-link",
                                            button_style="primary", layout=widgets.Layout(width="auto"))
            window_button = widgets.Button(description="Open 3D model in a window", icon="cube",
                                           layout=widgets.Layout(width="auto"))
            area = widgets.Output()

            def show_in_browser(_button=None):
                with area:
                    clear_output(wait=True)
                    print("Building the 3D model (a few seconds)...")
                    if self._preview_folder is None:
                        self._preview_folder = Path(tempfile.mkdtemp(prefix="cn12_preview_"))
                    folder, problem = self._export_in_separate_process(self._preview_folder, picture=False)
                    clear_output(wait=True)
                    viewer = folder / "coordination_model.html" if folder else None
                    if viewer is None or not viewer.is_file():
                        print(f"The 3D model could not be written. {problem}")
                        return
                    opened = open_in_default_app(viewer)
                    print("Opened in your browser." if opened else f"Open this file yourself: {viewer}")
                    print("This preview is temporary. Use 'Save this result' to keep a copy in 04_results.")

            def show_in_window(_button=None):
                with area:
                    clear_output(wait=True)
                    subprocess.Popen(self._command("view") + ["--sphere-size", str(self.sphere_size_factor)],
                                     creationflags=_NO_CONSOLE)
                    print("A window titled 'CN12: ...' opens in a few seconds. In it: tick boxes switch "
                          "links, edges, faces and labels;\npress P to step through the detected planes. "
                          "The notebook stays usable while it is open.")

            browser_button.on_click(show_in_browser)
            window_button.on_click(show_in_window)
            display(widgets.HBox([browser_button, window_button]))
            display(area)
            print("Drag to rotate, scroll to zoom. Gray lines join the center to its 12 neighbors; "
                  "dark lines are the edges of the polyhedron.\nSphere sizes use CIFKit's CIF radii "
                  f"at {self.sphere_size_factor:g} scale; they are a drawing choice, not a measurement.")

    # The 3D model and picture are made by PyVista. Doing that inside the notebook's
    # own Python process ended the kernel in VS Code, so those jobs are handed to
    # run_cn12.py in a separate process, exactly as the command line would run them.
    def _command(self, action: str) -> list[str]:
        block_name, site_id = self._current_key.split("\x1f", 1)
        return [sys.executable, str(self.project_root / "02_code" / "run_cn12.py"), action,
                "--file", str(self.file), "--block", block_name, "--site", site_id,
                "--settings", json.dumps(asdict(self.settings))]

    def _export_in_separate_process(self, output_root: Path, picture: bool = True) -> tuple[Path | None, str]:
        command = self._command("analyze") + ["--output", str(output_root), "--json"]
        if not picture:
            command.append("--no-image")
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=300,
                                  creationflags=_NO_CONSOLE)
            # The JSON report is the last thing printed; skip any library messages before it.
            start = done.stdout.rfind("\n[\n")
            records = json.loads(done.stdout[start + 1:] if start >= 0 else done.stdout)
            return Path(records[0]["result_directory"]), ""
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _render_planes(self) -> None:
        result = self.result
        with self.outputs[3]:
            clear_output(wait=True)
            geometry = result.geometry
            if geometry is None:
                print("No plane result: this center has no unique set of 12 neighbors.")
                return
            tolerance = self.settings.plane_membership_relative * geometry.characteristic_length_angstrom
            print(f"Planes through the center that hold at least six neighbors: {geometry.total_distinct_center_planes}\n"
                  f"  with exactly six neighbors: {geometry.exactly_six_neighbor_planes}\n"
                  f"  forming a convex hexagon around the center: {geometry.convex_six_vertex_rings}\n"
                  f"  and splitting the other six 3 : 3 (the classifying feature): "
                  f"{geometry.qualifying_bisecting_hexagonal_planes}\n"
                  f"A neighbor counts as on a plane within {tolerance:.3g} A "
                  f"({self.settings.plane_membership_relative:g} x the median neighbor distance).")
            if not geometry.planes:
                print("Zero planes is a valid result, not a failure.")
                return
            labels = {n.neighbor_id: f"N{i:02d}" for i, n in enumerate(result.neighbors, 1)}
            rows = [{
                "plane": plane.plane_id,
                "normal (unit)": tuple(round(v, 6) for v in plane.normal),
                "d in n.x + d = 0 (A)": plane.offset_angstrom,
                "neighbors on plane": plane.member_count,
                "which (rank in Neighbors tab)": " ".join(labels[i] for i in plane.ring_order_ids or plane.member_ids),
                "other neighbors, side + / side -": f"{plane.side_counts['positive']} / {plane.side_counts['negative']}",
                "max distance off plane (A)": plane.max_fit_residual_angstrom,
                "counts for classification": plane.is_bisecting_hexagon,
                "if not, why": "; ".join(plane.rejection_reasons),
            } for plane in geometry.planes]
            display(_table(rows))
            # The rank test behind the plane search, shown on the first plane.
            first = geometry.planes[0]
            by_id = {n.neighbor_id: n.relative_angstrom for n in result.neighbors}
            six = np.asarray([by_id[i] for i in first.member_ids[:6]], dtype=float)
            homogeneous = np.column_stack([np.vstack([six, np.zeros((1, 3))]), np.ones(7)])
            print(f"Rank check on {first.plane_id}: the 6 x 3 matrix of center-to-neighbor vectors has rank "
                  f"{np.linalg.matrix_rank(six)} (2 means one plane through the center);\n"
                  f"the 7 x 4 matrix [x y z 1] of the six neighbors plus the center, as in the TERRA paper, "
                  f"has rank {np.linalg.matrix_rank(homogeneous)} (3 means coplanar).")

    def _render_faces(self) -> None:
        result = self.result
        with self.outputs[4]:
            clear_output(wait=True)
            geometry = result.geometry
            if geometry is None:
                print("No polyhedron: this center has no unique set of 12 neighbors.")
                return
            hull = geometry.hull
            if not hull.success:
                print(f"The convex hull could not be built: {hull.status}.")
                return
            print(f"Vertices {hull.vertex_count}, edges {hull.edge_count}, faces {hull.face_count} "
                  f"(Euler characteristic {hull.euler_characteristic}; a closed polyhedron gives 2)\n"
                  f"Faces by number of sides: {hull.face_size_counts}\n"
                  f"Edges meeting at each vertex: {sorted(set(hull.vertex_degrees))}\n"
                  f"Which face types share an edge: {hull.face_adjacency_counts}\n"
                  f"Center inside the polyhedron: {hull.center_inside}; "
                  f"all 12 neighbors are corners: {hull.all_points_are_hull_vertices}")
            labels = {n.neighbor_id: f"N{i:02d}" for i, n in enumerate(result.neighbors, 1)}
            display(_table([{
                "face": face.face_id,
                "sides": face.polygon_size,
                "corners in order": " ".join(labels[i] for i in face.vertex_ids),
                "edge lengths (A)": tuple(round(v, 4) for v in face.edge_lengths_angstrom),
                "max distance off flat (A)": face.max_planarity_residual_angstrom,
            } for face in hull.faces]))
            display(_table([{
                "edge": f"{labels[a]} - {labels[b]}",
                "length (A)": length,
            } for (a, b), length in zip(hull.edges, hull.edge_lengths_angstrom)]))
            print("Edges are polyhedron boundaries (square diagonals removed); "
                  "they are not claimed to be chemical bonds.")

    def _render_settings(self) -> None:
        with self.outputs[5]:
            clear_output(wait=True)
            meaning = {
                "neighbor_count": "neighbors taken as the coordination shell",
                "candidate_target": "nearby positions examined when looking for shell gaps",
                "plane_degeneracy_relative": "below this, six points count as a line, not a plane (x L)",
                "plane_membership_relative": "how far off a plane a neighbor may sit and still be on it (x L)",
                "plane_normal_merge_radians": "two fitted planes closer than this angle are the same plane",
                "face_merge_relative": "hull triangles closer than this are one flat face (x L)",
                "face_normal_merge_radians": "hull triangles tilted less than this are one flat face",
                "neighbor_tie_relative": "12th and 13th neighbors closer than this are a tie (relative)",
                "max_translation_shell": "largest number of cell repeats searched in each direction",
                "accepted_radius_spread": "largest spread of center-neighbor distances still accepted",
                "accepted_edge_spread": "largest spread of edge lengths still accepted",
                "accepted_face_planarity_relative": "largest face bend still accepted (x L)",
                "accepted_shape_rmsd": "largest misfit to the ideal shape still accepted",
                "ideal_relative_spread": "below this, a shape is called ideal",
            }
            values = self.settings.as_dict()
            display(_table([{"setting": name, "value": f"{values[name]:g}", "what it controls": text}
                            for name, text in meaning.items()]))
            print("L is the median distance from the center to its 12 neighbors.\n"
                  "These are this program's working settings, not calibrated physical constants.\n"
                  "To change one, edit the SETTINGS line in the cell above and run that cell again, e.g.\n"
                  "    SETTINGS = AnalysisSettings(plane_membership_relative=1e-4)")

    def _render_project_tab(self) -> None:
        with self.outputs[6]:
            clear_output(wait=True)
            files = list_cif_files(self.project_root)
            button = widgets.Button(description=f"Analyze all {len(files)} CIF files",
                                    icon="play", button_style="warning",
                                    layout=widgets.Layout(width="260px"))
            area = widgets.Output()

            def run(_button=None):
                with area:
                    clear_output(wait=True)
                    print("Working...")
                    batch = analyze_batch(files, self.settings)
                    folder = export_batch_summary(batch, self.output_root)
                    clear_output(wait=True)
                    sites = pd.DataFrame(batch["site_rows"])
                    print(f"{batch['files_parsed']} of {batch['files_seen']} files read, "
                          f"{batch['site_results']} center sites analyzed "
                          f"({batch['unique_representative_sites']} after removing duplicate files).")
                    label = sites["classification"].fillna("not classified")
                    display(label.value_counts().rename_axis("result").to_frame("sites"))
                    columns = [c for c in ["path", "center_label", "classification", "geometry_quality",
                                           "plane_count", "face_size_counts", "d12_angstrom", "d13_angstrom",
                                           "selection_status"] if c in sites.columns]
                    view = sites[columns].copy()
                    view["path"] = view["path"].map(lambda value: Path(value).name)
                    with pd.option_context("display.max_rows", 200):
                        display(view)
                    print(f"Saved files.csv and sites.csv in {folder}")
                    self.batch = batch

            button.on_click(run)
            display(button, area)
            print("Runs every center site of every CIF in the project and writes two tables to 04_results.\n"
                  "These are the program's labels on a small development set, not a measured accuracy.")

    # ----------------------------------------------------------------- saving
    def _save(self, _button=None) -> None:
        if self.result is None:
            return
        self.save_button.disabled = True
        self.save_note.value = "<span style='color:#666'>saving (a few seconds)...</span>"
        folder, problem = self._export_in_separate_process(self.output_root)
        if folder is None:
            self.save_note.value = _message(f"could not save: {problem}", "error")
        else:
            names = ", ".join(sorted(path.name for path in folder.iterdir()))
            self.save_note.value = (
                f"saved in 04_results/{escape(folder.name)}"
                f"<br><span style='color:#666'>{escape(names)}</span>"
            )
            self.last_saved_folder = folder
        self.save_button.disabled = False


# Keeps a black console window from flashing up on Windows when a helper process starts.
_NO_CONSOLE = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _message(text: str, tone: str = "ok", big: bool = False) -> str:
    color = {"ok": "#1a7f37", "warn": "#9a6700", "error": "#cf222e"}[tone]
    size = "1.25em" if big else "1em"
    return f"<div style='font-size:{size};font-weight:600;color:{color};margin:6px 0'>{escape(text)}</div>"
