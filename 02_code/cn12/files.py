"""Finding and choosing CIF files, so nothing has to be typed into the code."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import webbrowser


# Folders that never hold input data.
_SKIPPED_FOLDERS = {".venv", "04_results", "macos_metadata", "__pycache__", ".git", ".ipynb_checkpoints"}


def find_project_root(start: str | Path | None = None) -> Path:
    """Walk upward from ``start`` (default: this file) to the project folder."""
    origin = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for candidate in (origin, *origin.parents):
        if (candidate / "02_code" / "cn12").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find the project folder (the one that contains 02_code/cn12) "
        f"starting from {origin}."
    )


def is_real_cif(path: Path) -> bool:
    """True for an actual CIF; false for macOS '._name.cif' sidecar files."""
    return path.suffix.lower() == ".cif" and not path.name.startswith("._")


def list_cif_files(project_root: str | Path) -> list[Path]:
    """Every CIF in the project, samples first, then the rest in name order."""
    root = Path(project_root).resolve()
    found = []
    for folder, subfolders, names in os.walk(root):
        subfolders[:] = sorted(name for name in subfolders if name not in _SKIPPED_FOLDERS)
        found.extend(Path(folder) / name for name in names if is_real_cif(Path(name)))
    samples = root / "03_data"

    def order(path: Path) -> tuple:
        relative = path.relative_to(root)
        return (samples not in path.parents, str(relative.parent).lower(), _natural_key(path.name))

    return sorted(found, key=order)


def _natural_key(name: str) -> tuple:
    """Sort '251905.cif' before '1042013.cif' and '2-W' before '10-X'."""
    digits = ""
    for character in name:
        if not character.isdigit():
            break
        digits += character
    return (int(digits) if digits else float("inf"), name.lower())


def expand_cif_inputs(items: list[str | Path]) -> list[Path]:
    """Turn a mix of CIF files and folders into a sorted list of CIF files."""
    files: set[Path] = set()
    for item in items:
        path = Path(item)
        if path.is_dir():
            files.update(p for p in path.rglob("*.cif")
                         if is_real_cif(p) and "macos_metadata" not in p.parts)
        elif path.is_file() and is_real_cif(path):
            files.add(path)
        elif not path.exists():
            raise FileNotFoundError(path)
    return sorted(files)


def pick_cif_files(start_folder: str | Path | None = None, multiple: bool = True) -> list[Path]:
    """Open the normal Windows 'choose a file' window and return the chosen CIFs.

    Returns an empty list when the window is cancelled. Raises RuntimeError when
    no window can be shown (for example over a remote text-only connection).
    """
    try:
        import tkinter
        from tkinter import filedialog
    except Exception as exc:  # tkinter is optional in some Python installs
        raise RuntimeError(f"The file-chooser window is unavailable: {exc}") from exc
    try:
        window = tkinter.Tk()
    except Exception as exc:
        raise RuntimeError(f"The file-chooser window is unavailable: {exc}") from exc
    try:
        window.withdraw()
        window.attributes("-topmost", True)  # keep the chooser in front of the editor
        window.update()
        options = {
            "parent": window,
            "title": "Choose CIF file(s) to analyze" if multiple else "Choose a CIF file to analyze",
            "filetypes": [("CIF files", "*.cif"), ("All files", "*.*")],
        }
        if start_folder is not None and Path(start_folder).is_dir():
            options["initialdir"] = str(start_folder)
        if multiple:
            chosen = filedialog.askopenfilenames(**options)
        else:
            single = filedialog.askopenfilename(**options)
            chosen = (single,) if single else ()
    finally:
        window.destroy()
    return [Path(item) for item in chosen if item]


def choose_from_list(files: list[Path], project_root: Path) -> list[Path]:
    """Text fallback for the chooser: print a numbered list and read a choice."""
    if not files:
        return []
    print("CIF files in this project:")
    for number, path in enumerate(files, 1):
        print(f"  {number:>3}  {path.relative_to(project_root)}")
    answer = input("Type a number (or several separated by spaces), or press Enter to cancel: ")
    chosen = []
    for token in answer.replace(",", " ").split():
        if token.isdigit() and 1 <= int(token) <= len(files):
            chosen.append(files[int(token) - 1])
        else:
            print(f"  ignored {token!r}")
    return chosen


def open_in_default_app(path: str | Path) -> bool:
    """Open a file or folder the way a double-click would. Never raises."""
    target = Path(path)
    try:
        if target.suffix.lower() in {".html", ".htm"}:
            return bool(webbrowser.open(target.resolve().as_uri()))
        if sys.platform.startswith("win"):
            os.startfile(str(target))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["xdg-open", str(target)])
        return True
    except Exception:
        return False
