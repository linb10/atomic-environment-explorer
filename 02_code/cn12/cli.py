"""Command-line entry point.

    python run_cn12.py                    choose CIF file(s) in a window, then analyze
    python run_cn12.py FILE.cif [...]     analyze the given file(s)
    python run_cn12.py analyze --file ... one file, with options
    python run_cn12.py view --file ...    open one site's 3D model in a window
    python run_cn12.py batch              every CIF in the project

Every distinct center site in a file is analyzed automatically; ``--site`` is
only needed to restrict a run to one of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .export import export_analysis, export_batch_summary
from .files import (
    choose_from_list,
    expand_cif_inputs,
    find_project_root,
    list_cif_files,
    open_in_default_app,
    pick_cif_files,
)
from .pipeline import (
    AnalysisSettings,
    analyze_all_sites,
    analyze_batch,
    analyze_file,
    describe_result,
    plain_verdict,
)

COMMANDS = {"analyze", "batch", "pick", "view"}


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    results = project_root / "04_results"
    parser = argparse.ArgumentParser(
        prog="run_cn12.py",
        description="Classify CN12 coordination geometry from CIF files. "
                    "Run with no arguments to choose files in a window.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_output_options(command: argparse.ArgumentParser) -> None:
        command.add_argument("--output", type=Path, default=results,
                             help="Folder that receives a new run folder per site (default: 04_results).")
        command.add_argument("--no-save", action="store_true", help="Print the result only; write no files.")
        command.add_argument("--no-html", action="store_true", help="Skip the interactive 3D model file.")
        command.add_argument("--no-image", action="store_true", help="Skip the PNG picture.")
        command.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of text.")

    pick = sub.add_parser("pick", help="Choose CIF file(s) in a window and analyze every site (the default).")
    add_output_options(pick)
    pick.add_argument("--no-open", action="store_true", help="Do not open the 3D model when finished.")

    single = sub.add_parser("analyze", help="Analyze CIF file(s) given on the command line.")
    single.add_argument("--file", required=True, type=Path, action="append",
                        help="CIF file; repeat --file for several.")
    single.add_argument("--site", default=None,
                        help="Restrict to one center label or site_id. Default: every site in the file.")
    single.add_argument("--block", default=None, help="Restrict to one CIF data block. Default: every block.")
    single.add_argument("--open", action="store_true", help="Open the 3D model when finished.")
    single.add_argument("--settings", default=None,
                        help="JSON object of AnalysisSettings values to change, e.g. "
                             "{\"plane_membership_relative\": 1e-4}.")
    add_output_options(single)

    view = sub.add_parser("view", help="Open one site's 3D model in its own window.")
    view.add_argument("--file", required=True, type=Path)
    view.add_argument("--site", default=None, help="Center label or site_id; needed if the file has several.")
    view.add_argument("--block", default=None)
    view.add_argument("--settings", default=None, help="JSON object of AnalysisSettings values to change.")
    view.add_argument("--sphere-size", type=float, default=0.45, help="Drawing scale of the atom spheres.")

    batch = sub.add_parser("batch", help="Analyze every CIF in the project and write two summary tables.")
    batch.add_argument("--input", action="append", type=Path,
                       help="File or folder; repeatable. Default: every CIF in the project.")
    batch.add_argument("--output", type=Path, default=results)
    return parser


def _normalize_arguments(argv: list[str]) -> list[str]:
    """Allow the two shortest forms: no arguments, or just file paths."""
    if not argv:
        return ["pick"]
    if argv[0] in COMMANDS or argv[0] in {"-h", "--help"}:
        return argv
    # Leading plain arguments are files (this is what drag-and-drop onto the
    # launcher produces); anything from the first option onward passes through.
    split = next((i for i, item in enumerate(argv) if item.startswith("-")), len(argv))
    expanded = ["analyze", "--open"]
    for item in argv[:split]:
        expanded += ["--file", item]
    return expanded + argv[split:]


def _choose_files(project_root: Path) -> list[Path]:
    samples = project_root / "03_data" / "cif_samples"
    try:
        return pick_cif_files(samples if samples.is_dir() else project_root)
    except RuntimeError as exc:
        print(exc)
        return choose_from_list(list_cif_files(project_root), project_root)


def _analyze_files(files: list[Path], args, settings: AnalysisSettings, open_viewer: bool) -> int:
    site = getattr(args, "site", None)    # the "pick" command has neither option
    block = getattr(args, "block", None)
    all_complete = True
    first_viewer: Path | None = None
    first_folder: Path | None = None
    records = []
    for path in files:
        try:
            if site is None:
                results = analyze_all_sites(path, block, settings)
            else:
                results = [analyze_file(path, site, block, settings)]
        except Exception as exc:
            all_complete = False
            records.append({"file": path.name, "status": "error", "error": str(exc)})
            if not args.json:
                print(f"\n{path.name}\n  could not be analyzed: {exc}")
            continue
        if not args.json:
            print(f"\n{path.name}")
        for result in results:
            all_complete = all_complete and result.status == "complete"
            record = describe_result(result)
            record["verdict"] = plain_verdict(result)
            if not args.no_save:
                folder = export_analysis(
                    result, args.output,
                    include_interactive_html=not args.no_html,
                    include_static_png=not args.no_image,
                )
                record["result_directory"] = str(folder)
                first_folder = first_folder or folder
                viewer = folder / "coordination_model.html"
                if first_viewer is None and viewer.is_file():
                    first_viewer = viewer
            records.append(record)
            if not args.json:
                _print_record(record)
    if args.json:
        print(json.dumps(records, indent=2))
    if open_viewer and (first_viewer or first_folder):
        target = first_viewer or first_folder
        if open_in_default_app(target) and not args.json:
            what = "the 3D model (drag to rotate, scroll to zoom)" if first_viewer else "the results folder"
            print(f"\nOpened {what}.")
    return 0 if all_complete else 2


def _print_record(record: dict) -> None:
    print(f"  center {record['center']}: {record['verdict']}")
    details = []
    if record.get("hexagonal_planes") is not None:
        details.append(f"{record['hexagonal_planes']} central hexagonal plane(s)")
    if record.get("faces"):
        details.append(record["faces"])
    if record.get("d12_angstrom") is not None and record.get("d13_angstrom") is not None:
        details.append(f"12th neighbor {record['d12_angstrom']:.4f} A, 13th {record['d13_angstrom']:.4f} A")
    if details:
        print("    " + "; ".join(details))
    if record.get("result_directory"):
        print(f"    saved in {record['result_directory']}")


def main(argv: list[str] | None = None) -> int:
    project_root = find_project_root()
    arguments = _normalize_arguments(list(sys.argv[1:] if argv is None else argv))
    args = build_parser(project_root).parse_args(arguments)
    overrides = json.loads(args.settings) if getattr(args, "settings", None) else {}
    settings = AnalysisSettings(**overrides)

    if args.command == "view":
        result = analyze_file(args.file, args.site, args.block, settings)
        if result.geometry is None:
            print(f"No 3D model: {plain_verdict(result)}")
            return 2
        from .plotting import show_desktop_window   # imported late: PyVista is slow to load
        show_desktop_window(result, args.sphere_size,
                            title=f"CN12: {args.file.name}, center {result.center_label}")
        return 0

    if args.command == "pick":
        files = _choose_files(project_root)
        if not files:
            print("No file chosen; nothing was analyzed.")
            return 0
        return _analyze_files(files, args, settings, open_viewer=not args.no_open)

    if args.command == "analyze":
        try:
            files = expand_cif_inputs(args.file)
        except FileNotFoundError as exc:
            print(f"File not found: {exc}")
            return 1
        if not files:
            print("No CIF files were found in the given --file arguments.")
            return 1
        return _analyze_files(files, args, settings, open_viewer=args.open)

    if args.command == "batch":
        files = expand_cif_inputs(args.input) if args.input else list_cif_files(project_root)
        batch_result = analyze_batch(files, settings)
        out = export_batch_summary(batch_result, args.output)
        report = {key: value for key, value in batch_result.items()
                  if key not in {"file_rows", "site_rows"}}
        report["result_directory"] = str(out)
        print(json.dumps(report, indent=2))
        return 0 if batch_result["file_parse_errors"] == 0 else 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
