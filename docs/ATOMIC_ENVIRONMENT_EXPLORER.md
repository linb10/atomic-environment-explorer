# Atomic Environment Explorer

A local Streamlit app for inspecting crystal structures (CIF), local atomic environments, planes and polyhedra, and for exporting documented per-site features. It is built on the project's CN12 analysis (`02_code/cn12/`, v0.2), which it reuses without changes.

It is an analysis tool. It does not do machine learning, simulation or property prediction. Its labels come from the program and have no independent ground truth (see *Limitations*).

## Launch

Double-click **`Run Atomic Environment Explorer.bat`** in the project folder. The app opens at http://localhost:8501. Close the console window to stop it.

Command line (PowerShell, from the project folder):

```powershell
.\.venv\Scripts\python.exe -m streamlit run 02_code\atomenv_app.py
```

### Opening it on a phone (same Wi-Fi)

1. Double-click **`Run Atomic Environment Explorer (phone on Wi-Fi).bat`**. The console prints addresses such as `http://192.168.0.110:8501`.
2. The first time, Windows Defender Firewall may ask whether Python can accept connections. Allow **Private networks** only. Do not allow Public.
3. On the phone, connect to the same Wi-Fi and open the printed address in a browser. Tap **›** at the top left to open the *Input* sidebar.

If the phone cannot connect, check three things: the PC's network is set to *Private* (Settings → Network → Wi-Fi → network properties); the network does not isolate clients (guest and campus networks often do); and the firewall allows inbound connections on port 8501. To add that rule yourself, run this in an administrator PowerShell:

```powershell
New-NetFirewallRule -DisplayName "Atomic Environment Explorer" -Direction Inbound -Protocol TCP -LocalPort 8501 -Profile Private -Action Allow
```

While phone mode is running, anyone on that network can open the app. It has no login. Use phone mode only on a network you trust, and close it when you are done.

## Setup on another computer

Requires Python 3.12. From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r 02_code\requirements-app.txt
```

`requirements-app.txt` includes the CN12 pins in `requirements.txt`, plus Streamlit 1.50.0, Plotly 6.3.0 and requests.

## Workflow

| Stage | Where | What you do |
| --- | --- | --- |
| 1. Load structures | Sidebar *Input* | Pick **Demo structures (COD, offline)**, **Upload .cif files**, **Upload .zip** or **COD record (online)**, then press **Load / analyze CIFs**. The sidebar shows file, parsed, block and site counts, and lists every skipped or failed file with its reason. |
| 2. Choose structure and site | Top of page | Pick a file, a crystal block (only shown when a file has several), N (4–16) and a center site (one representative per symmetry orbit). Metrics show the selection status, d_N, d_N+1 and the shell gap, followed by the program result. |
| 3. Inspect the structure | **Structure** tab | *Unit cell* shows the cell box, the sites, faded images of sites on faces/edges/corners, and a ring around the selected center. *Local region* shows the nearest 13–100 periodic positions (default 80), with the selected shell drawn opaque. |
| 4. Atomic environment | **Environment** tab | 3D shell around the center, a bar chart of distance by rank with the N boundary marked, the neighbor table (species, occupancy, image translation, boundary-shell flag) and selection diagnostics. |
| 5. Planes and polyhedra | **Planes & polyhedron** tab | Hull faces, edges, vertices and Euler number. For CN12: the plane table (members, residuals, normals, rejection reasons), a toggle for each plane on the 3D view, signed distances, and *Why this label*. |
| 6. Features and export | **Features & export** tab | Feature table; JSON and CSV downloads for the site or for every site in the dataset; the schema file; the equivalent Python call; technical settings. |

Every 3D view has *Display settings*: atom size, opacity and per-element color; labels; center–neighbor lines; polyhedron edges and faces; face opacity; plot height; and, on the Planes tab, plane color and opacity. Distinctions do not rely on color alone. The center is a diamond, selected neighbors are filled circles, and a neighbor set taken from a tied (non-unique) shell is drawn as open circles. Lines are labeled *not bonds*: they are geometric hull edges or center-to-neighbor guides, not assigned chemical bonds. The camera icon in each plot's toolbar saves a PNG.

**Clear selection** (sidebar) resets the file and site choices. **Reset all** also unloads the data and deletes its temporary copies.

## Data sources and licensing

- **Demo set** (`03_data/demo_cod/`, 5 files, raw and unmodified): COD 9008468 Cu (fcc), 9008506 Mg (hcp), 9008558 W (bcc, nearest-12 shell tied), 1524810 MgCu2 (Laves), 1000041 NaCl (CN6). COD states that its data are dedicated to the public domain under **CC0** (https://www.crystallography.net/cod/, checked 2026-10-18). `manifest.json` records each file's SHA-256, source URL and retrieval date. Please also credit the original authors; the app shows each entry's reference.
- **COD citation:** Gražulis, S. et al. (2012). Crystallography Open Database (COD): an open-access collection of crystal structures and platform for world-wide collaboration. *Nucleic Acids Research* 40, D420–D427.
- **COD retrieval** uses only the single-entry URL `https://www.crystallography.net/cod/<7-digit ID>.cif`, which the COD wiki page *howtoquerycod* documents. The app calls no search API, sends only the ID, and records the retrieval time.
- **Pearson's Crystal Data** files (`03_data/cif_samples/`, `old/cifkit/`) are licensed to Hunter College. The app does not bundle, list or expose them. A user may still upload their own files, which are processed only on that computer.
- **Uploads** are copied into a private temporary folder (`%TEMP%\atomenv_*`), named by hash. The folder is deleted on *Reset all* or when a new dataset is loaded. If the app is killed without a reset, the folder can remain in `%TEMP%`. Nothing is sent to any external service.

## Input safety

| Check | Limit or behavior |
| --- | --- |
| File type | `.cif` name and a `data_` header are required. Content with NUL bytes is rejected. |
| Size | 20 MB per CIF, 100 MB per ZIP, 300 MB uncompressed per ZIP, 2,000 ZIP entries. |
| ZIP paths | Absolute and `..` members are skipped as unsafe. `__MACOSX` and `._*` entries are skipped. Archive paths are never used as file-system paths. |
| Duplicates | A second file with identical bytes is skipped with a reason. |
| Failures | Every skipped file and parse error is listed with its reason; nothing is dropped silently. Analysis errors appear as a message with an expandable traceback. |
| Caching | Results are cached per session, keyed by file SHA-256, block, site, N and all settings. |

## Supported analyses

| N | Neighbor shell and tie diagnostics | Convex-hull faces and edges | Plane enumeration and family label |
| --- | --- | --- | --- |
| 12 | yes | yes | **yes**: cuboctahedron, anticuboctahedron, icosahedron, hexagonal prism (the cn12 method) |
| 4–11, 13–16 | yes | yes | **planned, not implemented**; no label is produced |

- **Neighbor rule:** the nearest N distinct periodic physical positions, excluding the center, from the existing complete periodic search. A shell is *tied* when d_N and d_N+1 differ by no more than the tie tolerance (baseline 1e-5 relative). In that case hull and family results are withheld, and the displayed neighbors are a deterministic preview only.
- **CN12:** runs `cn12.pipeline.analyze_parsed_block` unchanged, with the definitions and tolerances in `00_project_notes/CN12_METHOD_AND_VALIDATION.md`. Planes are fitted through the actual center (`n·x + d = 0`, `|n| = 1`). The residual columns separate exact coplanarity (residual ~1e-16 Å) from approximate fits accepted within 1e-5 × L.
- **Other N:** the same neighbor search with `count=N`. When N+1 ≥ 16, the tie-tolerance median uses N+1 candidates instead of 16. If the shell is unique, `cn12.geometry.reconstruct_hull` runs on its own.
- **Registry:** `02_code/atomenv/registry.py` maps each N to a family plugin. Adding a CN-specific analysis means registering a plugin there and handling it in `atomenv.api.analyze_site`.
- **Settings:** the *Analysis settings* expander exposes the tie tolerance and the plane-membership tolerance. Non-baseline values are flagged and written into every export. Changing them changes the scientific definitions.

## Feature schema (`atomenv.site_features/1.0`)

One record per analyzed site. JSON nests the full record; CSV holds one flat row per site. Full field list: `docs/examples/feature_schema.json`. Examples: `docs/examples/example_Mg_CN12.json` and `example_NaCl_CN6.csv`.

| Block | Contents |
| --- | --- |
| `provenance` | source, filename, archive, record ID, retrieval time, SHA-256, size, parse status/message, license, citation, entry reference |
| `structure` | block, formula as written in the CIF, space group, cell lengths (Å), angles (°), lattice column matrix (Å), parser |
| `center` | site ID, label, species/occupancies, multiplicity, fractional (wrapped to [0,1)), Cartesian (Å) |
| `selection` | requested N, rule, status, d_N, d_N+1, relative gap, tie counts, boundary-shell IDs, candidate count, search radius, heuristic CN estimate, `neighbors_authoritative` |
| `neighbors[]` | rank, stable ID, labels, species, image translation, unwrapped fractional, Cartesian, center-relative (Å), distance, mixed-occupancy and boundary-shell flags |
| `hull` | face-size counts, faces, edges, vertex degrees, Euler characteristic, center inside, volume (Å³), planarity; or `status: not_computed` with a reason |
| `cn12` | plane counts, plane records, spreads, shape RMSD, candidate family, classification, quality, reasons, `label_is_program_output: true`; or `not_applicable` / withheld |
| `capability`, `settings`, `software`, `warnings` | what was implemented for N, all tolerances, package versions, messages |

**Frame:** `r = L @ f`, where `L` is `structure.lattice_columns_angstrom` (Gemmi orthogonalization: a along x, b in the xy plane). Neighbor fractional coordinates are unwrapped (site + `image_translation`), so `relative = L @ (f_neighbor − f_center)`. Cartesian values depend on this convention and are not rotation-invariant; distances, counts and residuals are invariant. **Missing values** are `null` in JSON and empty in CSV, always next to a status or reason. Nothing is imputed.

## Python API

The app and scripts call the same functions in `02_code/atomenv/api.py`:

```python
import sys; sys.path.insert(0, "02_code")
from atomenv import api, sources
from atomenv.features import records_to_csv

# One call: parse a local CIF and get a feature record per site.
records = api.analyze_cif("03_data/demo_cod/9008506.cif", cn=12)
print(records[0]["cn12"]["verdict"])            # anticuboctahedron (slightly distorted)

# Step by step, the way the app does it (ZIPs and demo data too).
ws = sources.Workspace()
sources.load_demo(ws)                            # or ws.add_zip(name, bytes), sources.fetch_cod(ws, "9008468")
f = ws.parsed[0]
block = f.blocks[0]
analysis = api.analyze_site(block, block.center_sites[0].site_id, cn=12, input_file=f)
record = api.feature_record(analysis)
open("sites.csv", "w").write(records_to_csv([record]))
ws.close()                                       # deletes temporary copies
```

A test (`test_record_serializes_and_api_matches_app_path`) checks that `api.analyze_cif` and the app's code path produce identical records, apart from the creation timestamp. No HTTP API is provided: the local Python API covers the need without opening another network service.

## 3D technology

The in-app 3D views use Plotly, which renders in any browser, phones included. PyVista is not embedded. On this computer, starting its Trame server inside a kernel ended the kernel (CN12 method note, v0.2 section). The verified PyVista paths are still available through the existing program: `.\.venv\Scripts\python.exe 02_code\run_cn12.py view --file X.cif --site S` opens a desktop window, and `analyze` saves the HTML/PNG model. The app has no screen-recording feature; to share the view, share the browser window.

## Limitations

- No independently checked labels exist, so there is no accuracy figure. Labels are program outputs.
- Coordinate perturbations of about 1 part in 100,000 can change ideal-shape labels (`04_results/research_review_20261009`).
- Nearest-N selection is a geometric choice, not proof of physical coordination. The CN estimate is a heuristic.
- Partial occupancies describe an average structure. Mixed sites are shown with all species, and no ordered configuration is assumed.
- Family analysis exists only for CN12.
- Display caps: the local region shows at most 100 positions. In unit cells with more than 600 displayed positions, boundary images are hidden.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `.venv was not found` | Follow *Setup*. |
| Port 8501 already in use | Close the other app window, or run `streamlit run 02_code\atomenv_app.py --server.port 8502`. |
| COD record fails | Check the network. The demo set and uploads work offline. |
| "no unique nearest-N shell" | Expected for symmetric shells (W, NaCl at N=12). Choose another N. |
| Phone cannot connect | See *Opening it on a phone*. |

## Code layout

| File | Role |
| --- | --- |
| `02_code/atomenv_app.py` | Streamlit UI only |
| `02_code/atomenv/sources.py` | input validation, ZIP, COD, demo, provenance, temporary storage |
| `02_code/atomenv/api.py` | analysis service (reuses `cn12.crystal`, `cn12.geometry`, `cn12.pipeline`) |
| `02_code/atomenv/registry.py` | CN 4–16 capabilities and family plugins |
| `02_code/atomenv/features.py` | schema, JSON record, CSV row |
| `02_code/atomenv/viz.py` | Plotly figures (display only) |
| `02_code/tests/test_atomenv.py` | 22 checks, including a headless run of the app |
| `.streamlit/config.toml` | theme, upload limit, telemetry off |

Tests: `.\.venv\Scripts\python.exe -m unittest discover -s 02_code\tests -v` (40 tests: 18 CN12, 22 app).
