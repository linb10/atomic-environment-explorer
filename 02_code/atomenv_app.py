"""Atomic Environment Explorer - Streamlit user interface (CN12).

Launch with ``Run Atomic Environment Explorer.bat`` or
``.venv\\Scripts\\python.exe -m streamlit run 02_code\\atomenv_app.py``.
All science lives in ``atomenv`` (service layer) and ``cn12`` (analysis); this
file only arranges inputs, state and display.
"""

from __future__ import annotations

from dataclasses import replace
from html import escape
from pathlib import Path
import json
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from atomenv import api, sources
from atomenv.features import feature_record, flat_row, records_to_csv, records_to_json, schema_document
from atomenv.viz import (STYLES, DisplayOptions, distance_figure, element_color, environment_figure,
                         plane_extension_figure, plane_map_figure, plot_config, unit_cell_figure)
from cn12.pipeline import AnalysisSettings

CN = 12
st.set_page_config(page_title="Atomic Environment Explorer", page_icon="⚛", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
h1 {font-size: 1.55rem !important; font-weight: 650; margin-bottom: 0.2rem;}
h3 {font-size: 1.05rem !important; margin-top: 0.4rem;}
[data-testid="stSidebar"] h2 {font-size: 1.05rem !important;}
.ae-card {border:1px solid #e5e7eb; border-radius:10px; padding:14px 18px; margin:6px 0 10px 0;
          display:flex; flex-wrap:wrap; gap:18px 36px; align-items:center; background:#fafbfc;}
.ae-card.ok {border-left:6px solid #16a34a;} .ae-card.warn {border-left:6px solid #d97706;}
.ae-card.bad {border-left:6px solid #9ca3af;}
.ae-family {font-size:1.7rem; font-weight:700; color:#111827; line-height:1.1;}
.ae-quality {display:inline-block; margin-left:10px; padding:2px 10px; border-radius:12px; font-size:0.8rem;
             font-weight:600; vertical-align:middle;}
.q-ideal {background:#dcfce7; color:#166534;} .q-dist {background:#fef3c7; color:#92400e;}
.q-none {background:#f3f4f6; color:#374151;}
.ae-sub {color:#4b5563; font-size:0.9rem; margin-top:4px;}
.ae-stats {display:grid; grid-template-columns:repeat(auto-fit, minmax(92px, 1fr)); gap:4px 22px; flex:1;
           min-width:300px;}
.ae-stat .k {font-size:0.72rem; color:#6b7280; letter-spacing:0.02em;}
.ae-stat .v {font-size:1.02rem; font-weight:600; color:#111827; font-variant-numeric:tabular-nums;}
.ae-reason {color:#4b5563; font-size:0.85rem; flex-basis:100%; margin-top:-6px;}
@media (max-width: 640px) { .block-container {padding-left: 0.8rem; padding-right: 0.8rem;}
  .ae-family {font-size:1.35rem;} }
</style>
""", unsafe_allow_html=True)

BASELINE = AnalysisSettings()
SOURCES = ["Demo set (COD)", "Upload .cif", "Upload .zip", "COD ID"]
QUALITY = {"ideal_within_numerical_tolerance": ("Ideal", "q-ideal", "ok"),
           "accepted_distorted_geometry": ("Distorted", "q-dist", "ok"),
           "strongly_distorted": ("Strongly distorted", "q-dist", "warn"),
           "ambiguous": ("Ambiguous", "q-none", "warn"),
           "analysis_error": ("Hull failed", "q-none", "bad")}
STATUS = {"unambiguous": "unique", "boundary_tie": "tied", "incomplete_search": "incomplete",
          "unsupported_input": "unsupported"}


# ---------------------------------------------------------------- state helpers
def state(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def reset_dataset():
    ws = st.session_state.get("workspace")
    if ws is not None:
        ws.close()
    for key in ("workspace", "file_key", "block_name", "site_id", "analysis_cache", "planes_on",
                "dataset_export", "ext_plane"):
        st.session_state.pop(key, None)


def load(source, uploads=None, cod_id=""):
    reset_dataset()
    ws = sources.Workspace()
    try:
        if source == SOURCES[0]:
            sources.load_demo(ws)
        elif source == SOURCES[1]:
            for item in uploads or []:
                ws.add_bytes(item.name, item.getvalue(), "upload")
        elif source == SOURCES[2]:
            if uploads is not None:
                ws.add_zip(uploads.name, uploads.getvalue())
        else:
            sources.fetch_cod(ws, cod_id)
        st.session_state["workspace"] = ws
    except Exception as exc:  # keep the app alive on unexpected input failures
        ws.close()
        show_exception("Loading failed", exc)


def current_settings() -> AnalysisSettings:
    return replace(BASELINE,
                   neighbor_tie_relative=state("tie_rel", BASELINE.neighbor_tie_relative),
                   plane_membership_relative=state("plane_rel", BASELINE.plane_membership_relative))


def get_analysis(input_file, block, site_id, settings):
    cache = state("analysis_cache", {})
    key = (input_file.sha256, block.block_name, site_id, repr(sorted(settings.as_dict().items(), key=str)))
    if key not in cache:
        cache[key] = api.analyze_site(block, site_id, CN, settings, input_file=input_file)
    return cache[key]


def show_exception(message: str, exc: BaseException):
    st.error(f"{message}: {exc}")
    with st.expander("Traceback"):
        st.code("".join(traceback.format_exception(exc)), language="text")


def chart(fig, key, filename="atomic_environment"):
    st.plotly_chart(fig, config=plot_config(st.session_state.get("scroll_zoom", False), filename), key=key)


def fmt(v, spec=".4f", dash="—"):
    return format(v, spec) if v is not None else dash


# ---------------------------------------------------------------- sidebar: input
with st.sidebar:
    st.header("Load structures")
    source = st.radio("Source", SOURCES, key="source_choice", label_visibility="collapsed")
    uploads, cod_id = None, ""
    if source == SOURCES[1]:
        uploads = st.file_uploader("CIF files", type=["cif"], accept_multiple_files=True)
    elif source == SOURCES[2]:
        uploads = st.file_uploader("ZIP of CIFs", type=["zip"])
    elif source == SOURCES[3]:
        cod_id = st.text_input("COD ID", placeholder="9008468")
    if st.button("Load", type="primary", width="stretch"):
        with st.spinner("Parsing…"):
            load(source, uploads, cod_id)

    ws = st.session_state.get("workspace")
    if ws is not None and ws.files:
        c = ws.counts()
        st.caption(f"{c['parsed']}/{c['inputs']} files parsed · {c['blocks']} blocks · {c['sites']} distinct sites")
        problems = [f for f in ws.files if f.status != "parsed"]
        warned = [f for f in ws.parsed if f.reason]
        if problems or warned:
            with st.expander(f"{len(problems)} failed · {len(warned)} warnings"):
                for f in problems:
                    st.markdown(f"**{f.name}**{f' ({f.archive})' if f.archive else ''}: "
                                f"{f.status.replace('_', ' ')} — {f.reason}")
                for f in warned:
                    st.markdown(f"**{f.name}**: {f.reason}")


# ---------------------------------------------------------------- main: empty state
st.title("Atomic Environment Explorer")
ws = st.session_state.get("workspace")
if ws is None or not ws.parsed:
    if ws is not None and ws.files:
        st.warning("No file could be parsed. See the sidebar for reasons.")
    st.markdown("Load CIF files from the sidebar, or start with the demo set.")
    if st.button("Load demo set", type="primary"):
        load(SOURCES[0])
        st.rerun()
    st.stop()


# ---------------------------------------------------------------- selection row
parsed = ws.parsed
file_labels = {f.key: f"{f.blocks[0].formula} — {f.name}" for f in parsed}
s1, s2 = st.columns([1.3, 1])
file_key = s1.selectbox("Structure", list(file_labels), format_func=file_labels.get, key="file_key")
input_file = next(f for f in parsed if f.key == file_key)
block_names = [b.block_name for b in input_file.blocks]
if st.session_state.get("block_name") not in block_names:
    st.session_state.pop("block_name", None)
if len(block_names) > 1:
    block_name = s1.selectbox("Data block", block_names, key="block_name")
else:
    block_name = block_names[0]
block = next(b for b in input_file.blocks if b.block_name == block_name)
sites = api.site_table(block)
site_ids = [r["site_id"] for r in sites]
site_lookup = {r["site_id"]: r for r in sites}
if st.session_state.get("site_id") not in site_ids:
    st.session_state.pop("site_id", None)
site_id = s2.selectbox("Center site", site_ids, key="site_id",
                       format_func=lambda s: f"{site_lookup[s]['label']}  ({site_lookup[s]['species']}, "
                                             f"mult. {site_lookup[s]['multiplicity']})")

state("tie_rel", BASELINE.neighbor_tie_relative)
state("plane_rel", BASELINE.plane_membership_relative)
settings = current_settings()
try:
    analysis = get_analysis(input_file, block, site_id, settings)
except Exception as exc:
    show_exception("This site could not be analyzed", exc)
    st.stop()

sel = analysis.selection
geometry = analysis.cn12.geometry if analysis.cn12 is not None else None
hull = analysis.hull
elements = sorted({c.element for s in block.physical_sites for c in s.components})


# ---------------------------------------------------------------- sidebar: display + analysis settings
with st.sidebar:
    st.header("Display")
    style = st.selectbox("Style", list(STYLES), key="style",
                         help="Classic uses the colours of the original CN12 PyVista program.")
    opts = DisplayOptions(
        style=style,
        atom_scale=st.slider("Atom size", 0.3, 2.5, 1.0, 0.1, key="atom_scale"),
        atom_opacity=st.slider("Atom opacity", 0.1, 1.0, 0.95, 0.05, key="atom_opacity"),
        height=st.select_slider("Plot height", [420, 520, 620, 720, 860], 620, key="plot_h"),
    )
    d1, d2 = st.columns(2)
    opts.show_labels = d1.toggle("Labels", False, key="labels")
    opts.show_legend = d2.toggle("Legend", True, key="legend")
    opts.show_edges = d1.toggle("Edges", True, key="edges")
    opts.show_faces = d2.toggle("Faces", True, key="faces")
    opts.show_contacts = d1.toggle("Center lines", False, key="contacts",
                                   help="Center–neighbor lines (geometric, not bonds).")
    opts.show_axes = d2.toggle("Axes", True, key="axes")
    opts.orthographic = d1.toggle("Orthographic", False, key="ortho")
    opts.blend_mixed = d2.toggle("Blend mixed", True, key="blend",
                                 help="Colour mixed-occupancy sites by occupancy-weighted blend.")
    st.toggle("Scroll wheel zooms 3D", False, key="scroll_zoom",
              help="Off: the wheel scrolls the page. Zoom with pinch, or the zoom tool in the plot toolbar.")
    show_dist = st.toggle("Distance chart", True, key="show_dist")
    with st.expander("Colours"):
        p1, p2 = st.columns(2)
        opts.plane_color = p1.color_picker("Plane", STYLES[style]["plane"], key=f"plane_c_{style}")
        opts.face_opacity = p2.slider("Face opacity", 0.0, 0.8, STYLES[style]["face_opacity"], 0.02,
                                      key=f"face_op_{style}")
        opts.plane_opacity = p2.slider("Plane opacity", 0.05, 0.9, STYLES[style]["plane_opacity"], 0.05,
                                       key=f"plane_op_{style}")
        cols = st.columns(3)
        for i, e in enumerate(elements):
            opts.color_overrides[e] = cols[i % 3].color_picker(e, element_color(e), key=f"col_{e}")

    st.header("Analysis")
    st.select_slider("Neighbor tie tolerance (rel.)", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3], key="tie_rel",
                     format_func=lambda v: f"{v:g}")
    st.select_slider("Plane tolerance (× L)", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], key="plane_rel",
                     format_func=lambda v: f"{v:g}")
    if settings != BASELINE:
        def _restore():
            st.session_state["tie_rel"] = BASELINE.neighbor_tie_relative
            st.session_state["plane_rel"] = BASELINE.plane_membership_relative
        st.warning("Non-baseline settings (recorded in exports).")
        st.button("Restore baseline", on_click=_restore, width="stretch")
    st.divider()
    st.button("Reset", on_click=reset_dataset, width="stretch")


# Plotly's 3D scenes swallow wheel events even with scrollZoom off. Unless the user enables
# wheel zoom, stop wheel events at the window (capture phase) before they reach a plot, so
# the page keeps scrolling past 3D models.
components.html(f"""<script>
const w = window.parent;
w.__aeScrollZoom = {str(bool(st.session_state.get("scroll_zoom"))).lower()};
if (!w.__aeWheelGuard) {{
  w.__aeWheelGuard = true;
  w.addEventListener("wheel", e => {{
    if (!w.__aeScrollZoom && e.target.closest && e.target.closest(".js-plotly-plot")) e.stopPropagation();
  }}, {{capture: true, passive: true}});
}}
</script>""", height=0)


# ---------------------------------------------------------------- result card
gap = ((sel.d13_angstrom - sel.d12_angstrom) / sel.d12_angstrom * 100
       if sel.d12_angstrom and sel.d13_angstrom else None)
if geometry is not None:
    q_label, q_class, tone = QUALITY.get(geometry.geometry_quality, ("Other", "q-none", "bad"))
    family = geometry.classification if geometry.geometry_quality in (
        "ideal_within_numerical_tolerance", "accepted_distorted_geometry") else (
        "Other" if geometry.geometry_quality != "ambiguous" else "Unclassified")
    if family == "Other" and geometry.candidate_family not in (None, "", "none", "other"):
        q_label = f"{geometry.candidate_family} connectivity, {q_label.lower()}" \
            if geometry.geometry_quality == "strongly_distorted" else "none of the four families"
    elif family == "Other":
        q_label = "none of the four families"
    reason = ""
else:
    family, q_label, q_class, tone = "Not classified", f"shell {STATUS.get(sel.status, sel.status)}", "q-none", "warn"
    reason = analysis.verdict.replace("not classified: ", "").replace(" A ", " Å ")
faces_txt = ""
if hull is not None:
    names = {3: "tri", 4: "quad", 5: "pent", 6: "hex"}
    faces_txt = " · ".join(f"{v} {names.get(int(k), f'{k}-gon')}"
                           for k, v in sorted(hull.face_size_counts.items(), key=lambda kv: int(kv[0])))
stats = [("d₁₂", f"{fmt(sel.d12_angstrom)} Å"), ("d₁₃", f"{fmt(sel.d13_angstrom)} Å"),
         ("Shell gap", f"{fmt(gap, '.2f')} %"),
         ("Hex planes", geometry.qualifying_bisecting_hexagonal_planes if geometry else "—"),
         ("Shape RMSD", fmt(geometry.normalized_shape_rmsd if geometry else None)),
         ("Faces", faces_txt or "—"), ("CN estimate", sel.cn_estimate if sel.cn_estimate is not None else "—")]
stat_html = "".join(f'<div class="ae-stat"><div class="k">{k}</div><div class="v">{escape(str(v))}</div></div>'
                    for k, v in stats)
st.markdown(
    f'<div class="ae-card {tone}"><div><div class="ae-family">{escape(family.capitalize())}'
    f'<span class="ae-quality {q_class}">{escape(q_label)}</span></div>'
    f'<div class="ae-sub">{escape(analysis.center.display_label)} in {escape(block.formula)} · '
    f'{escape(block.space_group)} (No. {block.space_group_number}) · CN12</div></div>'
    f'<div class="ae-stats">{stat_html}</div>'
    + (f'<div class="ae-reason">{escape(reason)}</div>' if reason else "") + '</div>',
    unsafe_allow_html=True)
if sel.cn_estimate is not None and sel.cn_estimate != CN and sel.status == "unambiguous":
    st.caption(f"The distance-gap heuristic suggests CN {sel.cn_estimate} for this site; "
               "the CN12 shell above is still reported as computed.")

tabs = st.tabs(["Polyhedron", "Planes", "Plane extension", "Neighbors", "Structure", "Dataset & export"])


# ---------------------------------------------------------------- Polyhedron
with tabs[0]:
    if hull is None:
        st.info("No polyhedron: the nearest-12 shell is not unique. Neighbors are shown as a preview.")
    left, right = st.columns([2.3, 1])
    with left:
        chart(environment_figure(analysis, opts), "poly_plot", "polyhedron")
    with right:
        if hull is not None:
            st.markdown(f"**{hull.vertex_count}** vertices · **{hull.edge_count}** edges · "
                        f"**{hull.face_count}** faces  \nEuler V−E+F = {hull.euler_characteristic}  \n"
                        f"Vertex degrees: {', '.join(map(str, sorted(set(hull.vertex_degrees))))}  \n"
                        f"Center inside: {'yes' if hull.center_inside else 'no'}  \n"
                        f"Max face residual: {hull.max_face_planarity_residual_angstrom:.1e} Å")
        if geometry is not None:
            st.markdown(f"Radial spread: {fmt(geometry.center_distance_spread)}  \n"
                        f"Edge spread: {fmt(geometry.edge_length_spread)}")
            with st.expander("Why this classification"):
                for r in geometry.classification_reasons:
                    st.markdown(f"- {r}")
    if hull is not None:
        labels = {n.neighbor_id: f"N{i:02d}" for i, n in enumerate(analysis.neighbors, 1)}
        f1, f2 = st.columns(2)
        with f1.expander(f"Faces ({hull.face_count})"):
            st.dataframe(pd.DataFrame([{
                "face": f.face_id, "sides": f.polygon_size,
                "corners": " ".join(labels.get(i, "?") for i in f.vertex_ids),
                "edge lengths (Å)": ", ".join(f"{v:.4f}" for v in f.edge_lengths_angstrom),
                "planarity (Å)": float(f"{f.max_planarity_residual_angstrom:.2e}"),
            } for f in hull.faces]), hide_index=True, width="stretch")
            if getattr(hull, "face_adjacency_counts", None):
                st.caption(f"Face-type adjacency: {hull.face_adjacency_counts}")
        with f2.expander(f"Edges ({hull.edge_count})"):
            st.dataframe(pd.DataFrame([{"edge": f"{labels.get(a, '?')}–{labels.get(b, '?')}",
                                        "length (Å)": round(l, 5)}
                                       for (a, b), l in zip(hull.edges, hull.edge_lengths_angstrom)]),
                         hide_index=True, width="stretch")


# ---------------------------------------------------------------- Planes
hex_planes = [p.plane_id for p in geometry.planes if p.is_bisecting_hexagon] if geometry else []
with tabs[1]:
    if geometry is None:
        st.info("Plane analysis needs a unique nearest-12 shell.")
    else:
        all_ids = [p.plane_id for p in geometry.planes]
        if not set(st.session_state.get("planes_on") or []) <= set(all_ids):
            st.session_state.pop("planes_on", None)
        c1, c2 = st.columns([3, 1])
        planes_on = c1.multiselect(
            "Planes", all_ids, default=None if "planes_on" in st.session_state else hex_planes[:1],
            key="planes_on", format_func=lambda p: f"{p} ⬡" if p in hex_planes else p,
            help="⬡ = bisecting hexagonal plane (six members forming a convex hexagon around the center).")
        isolate = c2.toggle("Isolate planes", True, key="isolate",
                            help="Hide the polyhedron and atoms not on the selected planes.")
        chart(environment_figure(analysis, opts, plane_ids=planes_on, isolate_planes=isolate and bool(planes_on)),
              "plane_plot", "planes")
        st.caption(f"{len(hex_planes)} bisecting hexagonal planes of {geometry.total_distinct_center_planes} "
                   f"distinct center planes. Membership tolerance {analysis.settings.plane_membership_relative:g} × L, "
                   f"L = {geometry.characteristic_length_angstrom:.4f} Å.")
        rows = []
        for p in geometry.planes:
            hkl, d = api.miller_indices(block, p.normal)
            rows.append({"plane": p.plane_id, "hexagon": "⬡" if p.is_bisecting_hexagon else "",
                         "members": p.member_count,
                         "parallel to (hkl)": f"({' '.join(map(str, hkl))})" if hkl else "—",
                         "d_hkl (Å)": round(d, 4) if d else None,
                         "max residual (Å)": float(f"{p.max_fit_residual_angstrom:.2e}"),
                         "normal": "(" + ", ".join(f"{x:.4f}" for x in p.normal) + ")",
                         "rejected because": "; ".join(p.rejection_reasons) or ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        index = {n.neighbor_id: i + 1 for i, n in enumerate(analysis.neighbors)}
        with st.expander("Signed distances for selected planes"):
            for p in geometry.planes:
                if p.plane_id in (planes_on or []):
                    members = set(p.member_ids)
                    st.markdown(f"**{p.plane_id}** · d = {p.offset_angstrom:.5f} Å")
                    st.dataframe(pd.DataFrame([{"neighbor": f"N{index.get(nid, 0):02d}", "member": nid in members,
                                                "signed distance (Å)": float(f"{dist:.3e}")}
                                               for nid, dist in zip(p.neighbor_ids, p.signed_distances_angstrom)]),
                                 hide_index=True, width="stretch")


# ---------------------------------------------------------------- Plane extension
with tabs[2]:
    if geometry is None or not geometry.planes:
        st.info("Plane extension needs a unique nearest-12 shell with detected center planes.")
    else:
        ids = hex_planes + [p.plane_id for p in geometry.planes if p.plane_id not in hex_planes]
        if st.session_state.get("ext_plane") not in ids:
            st.session_state.pop("ext_plane", None)
        c1, c2, c3, c4 = st.columns([1.2, 2, 1.2, 1.2])
        pid = c1.selectbox("Plane", ids, key="ext_plane", format_func=lambda p: f"{p} ⬡" if p in hex_planes else p)
        n_atoms = c2.slider("In-plane atoms", 20, api.PLANE_EXTENSION_CAP, 100, 10, key="ext_n")
        tol = c3.number_input("Tolerance (Å)", 0.001, 0.5, 0.05, 0.01, format="%.3f", key="ext_tol",
                              help="Maximum |distance| from the plane for an atom to count as in-plane.")
        view = c4.radio("View", ["Both", "Face-on", "3D"], horizontal=True, key="ext_view")
        try:
            ext = api.plane_extension(analysis, pid, n_atoms, tol)
        except Exception as exc:
            show_exception("Plane extension failed", exc)
            ext = None
        if ext is not None and ext.atoms:
            shells = api.in_plane_shells(ext)
            hkl = f"({' '.join(map(str, ext.miller))})" if ext.miller else "irrational"
            elems_on = sorted({e for a in ext.atoms for e, _ in a.species_occupancies})
            area_r = ext.atoms[-1].distance_angstrom
            density = len(ext.atoms) / (3.141592653589793 * area_r ** 2) if area_r else None
            pstats = [("Parallel to", hkl),
                      ("d_hkl", f"{ext.d_spacing_angstrom:.4f} Å" if ext.d_spacing_angstrom else "—"),
                      ("In-plane NN", f"{shells[0]['distance (Å)']:.4f} Å"),
                      ("Areal density", f"{density:.4f} Å⁻²" if density else "—"),
                      ("Shells shown", len(shells)), ("Species", ", ".join(elems_on))]
            st.markdown('<div class="ae-stats" style="margin:4px 0 8px 0">' + "".join(
                f'<div class="ae-stat"><div class="k">{k}</div><div class="v">{escape(str(v))}</div></div>'
                for k, v in pstats) + "</div>", unsafe_allow_html=True)
            context = None
            if view != "Face-on" and st.toggle("Show off-plane atoms (faded)", False, key="ext_ctx"):
                context, _ = api.extended_region(analysis, api.EXTENDED_ATOM_CAP)
            if view == "Both":
                a, b = st.columns(2)
                with a:
                    chart(plane_map_figure(analysis, ext, opts, [s["distance (Å)"] for s in shells[:3]]),
                          "ext_map", f"plane_{pid}_map")
                with b:
                    chart(plane_extension_figure(analysis, ext, opts, context), "ext_3d", f"plane_{pid}_3d")
            elif view == "Face-on":
                chart(plane_map_figure(analysis, ext, opts, [s["distance (Å)"] for s in shells[:3]]),
                      "ext_map", f"plane_{pid}_map")
            else:
                chart(plane_extension_figure(analysis, ext, opts, context), "ext_3d", f"plane_{pid}_3d")
            t1, t2 = st.columns([1, 1.6])
            with t1:
                st.markdown("**In-plane shells**")
                st.dataframe(pd.DataFrame(shells)[["shell", "distance (Å)", "d / d₁", "count", "species"]],
                             hide_index=True, width="stretch", height=300)
            atom_rows = [{"#": i + 1, "label": "+".join(a.source_labels),
                          "species": ", ".join(f"{e} {o:g}" for e, o in a.species_occupancies),
                          "distance (Å)": round(a.distance_angstrom, 5),
                          "u (Å)": round(float(ext.in_plane_xy[i, 0]), 5), "v (Å)": round(float(ext.in_plane_xy[i, 1]), 5),
                          "off-plane (Å)": float(f"{ext.off_plane_angstrom[i]:.2e}"),
                          "hexagon member": a.neighbor_id in ext.shell_member_ids,
                          "image": str(a.image_translation)} for i, a in enumerate(ext.atoms)]
            with t2:
                st.markdown("**In-plane positions**")
                st.dataframe(pd.DataFrame(atom_rows), hide_index=True, width="stretch", height=300)
            stem = f"{Path(input_file.name).stem}_{analysis.center.display_label}_{pid}".replace("+", "-")
            st.download_button("Download in-plane positions (CSV)", pd.DataFrame(atom_rows).to_csv(index=False),
                               f"{stem}_plane.csv", "text/csv")
        elif ext is not None:
            st.warning("No periodic positions found on this plane within the search limit. Increase the tolerance.")


# ---------------------------------------------------------------- Neighbors
with tabs[3]:
    if show_dist:
        chart(distance_figure(analysis, options=opts), "dist_plot", "neighbor_distances")
    boundary = set(sel.boundary_shell_neighbor_ids)
    st.dataframe(pd.DataFrame([{
        "#": i + 1, "label": "+".join(n.source_labels),
        "species": ", ".join(f"{e} {o:g}" for e, o in n.species_occupancies),
        "distance (Å)": round(n.distance_angstrom, 6),
        "x (Å)": round(n.relative_angstrom[0], 5), "y (Å)": round(n.relative_angstrom[1], 5),
        "z (Å)": round(n.relative_angstrom[2], 5), "image": str(n.image_translation),
        "boundary shell": n.neighbor_id in boundary, "mixed": n.mixed_occupancy,
    } for i, n in enumerate(analysis.neighbors)]), hide_index=True, width="stretch")
    st.caption(f"{sel.rationale} {sel.candidate_count} candidates searched. "
               f"CN estimate {sel.cn_estimate}: {sel.cn_estimate_method}.")
    if analysis.warnings:
        with st.expander(f"Warnings ({len(set(analysis.warnings))})"):
            for w in dict.fromkeys(analysis.warnings):
                st.markdown(f"- {w}")


# ---------------------------------------------------------------- Structure
with tabs[4]:
    c1, c2 = st.columns([1, 2])
    view = c1.radio("View", ["Unit cell", "Local region"], horizontal=True, key="struct_view")
    if view == "Local region":
        n_region = c2.slider("Atoms", CN + 1, api.EXTENDED_ATOM_CAP, 80, key="region_n")
        try:
            region, _ = api.extended_region(analysis, n_region)
            o = replace(opts, show_contacts=False, show_faces=False)
            chart(environment_figure(analysis, o, extended=region), "region_plot", "local_region")
        except Exception as exc:
            show_exception("Region view failed", exc)
    else:
        positions, note = api.unit_cell_positions(block)
        if note:
            st.caption(note)
        chart(unit_cell_figure(block, positions, opts, analysis.center.position_id), "cell_plot", "unit_cell")
    a, b, c = block.cell_lengths_angstrom
    al, be, ga = block.cell_angles_degrees
    st.markdown(f"a = {a:.5g} Å, b = {b:.5g} Å, c = {c:.5g} Å · α = {al:.5g}°, β = {be:.5g}°, γ = {ga:.5g}° · "
                f"{block.space_group} (No. {block.space_group_number}) · block `{block.block_name}`")
    st.dataframe(pd.DataFrame(sites).drop(columns=["site_id"]), hide_index=True, width="stretch")
    if input_file.entry_reference or input_file.record_id:
        st.caption(f"{input_file.record_id or input_file.name} — {input_file.entry_reference or ''} "
                   f"{input_file.license or ''}")
    with st.expander("Provenance"):
        st.json(input_file.provenance(), expanded=False)


# ---------------------------------------------------------------- Dataset & export
with tabs[5]:
    record = feature_record(analysis)
    stem = f"{Path(input_file.name).stem}_{analysis.center.display_label}_CN12".replace("+", "-")
    st.markdown("**This site**")
    e1, e2, e3 = st.columns(3)
    e1.download_button("Record (JSON)", records_to_json([record]), f"{stem}.json", "application/json", width="stretch")
    e2.download_button("Row (CSV)", records_to_csv([record]), f"{stem}.csv", "text/csv", width="stretch")
    e3.download_button("Feature schema", json.dumps(schema_document(), indent=2), "feature_schema.json",
                       "application/json", width="stretch")
    with st.expander("All features for this site"):
        st.dataframe(pd.DataFrame([{"feature": k, "value": str(v)} for k, v in flat_row(record).items()]),
                     hide_index=True, width="stretch", height=360)

    st.markdown("**All loaded sites**")
    jobs = [(f, b, s.site_id) for f in parsed for b in f.blocks for s in b.center_sites]
    if st.button(f"Analyze all {len(jobs)} sites", type="primary"):
        records, summary, failures = [], [], []
        progress = st.progress(0.0)
        for i, (f, b, s) in enumerate(jobs, 1):
            try:
                a = get_analysis(f, b, s, settings)
                records.append(feature_record(a))
                g = a.cn12.geometry if a.cn12 else None
                summary.append({"file": f.name, "formula": b.formula, "space group": b.space_group,
                                "site": a.center.display_label, "result": a.verdict,
                                "hex planes": g.qualifying_bisecting_hexagonal_planes if g else None,
                                "RMSD": round(g.normalized_shape_rmsd, 4) if g and g.normalized_shape_rmsd is not None else None,
                                "d12 (Å)": round(a.selection.d12_angstrom, 4) if a.selection.d12_angstrom else None,
                                "d13 (Å)": round(a.selection.d13_angstrom, 4) if a.selection.d13_angstrom else None,
                                "CN est.": a.selection.cn_estimate})
            except Exception as exc:
                failures.append({"file": f.name, "block": b.block_name, "site": s, "error": str(exc)})
            progress.progress(i / len(jobs))
        progress.empty()
        st.session_state["dataset_export"] = (records, summary, failures)
    export = st.session_state.get("dataset_export")
    if export:
        records, summary, failures = export
        st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch")
        counts = pd.Series([r["result"] for r in summary]).value_counts()
        st.caption(" · ".join(f"{v} × {k}" for k, v in counts.items()))
        x1, x2, x3 = st.columns(3)
        x1.download_button("All sites (JSON)", records_to_json(records), "atomenv_sites_CN12.json",
                           "application/json", width="stretch")
        x2.download_button("All sites (CSV)", records_to_csv(records), "atomenv_sites_CN12.csv", "text/csv",
                           width="stretch")
        x3.download_button("Summary (CSV)", pd.DataFrame(summary).to_csv(index=False), "atomenv_summary_CN12.csv",
                           "text/csv", width="stretch")
        if failures:
            st.error(f"{len(failures)} site(s) failed")
            st.dataframe(pd.DataFrame(failures), hide_index=True, width="stretch")
    with st.expander("Python API"):
        st.code(f'''import sys; sys.path.insert(0, "02_code")
from atomenv import api
records = api.analyze_cif("{input_file.name}", site="{analysis.center.display_label}", cn=12)''', language="python")
