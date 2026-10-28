"""Input handling: demo files, uploads, ZIP archives and single COD records.

All inputs are treated as untrusted. Bytes are size-checked and content-checked,
then written under a private temporary directory using a hash-derived file name
(archive member paths are never used as file-system paths), parsed with the
existing Gemmi-based ``cn12.crystal.parse_cif_blocks``, and every outcome,
including skips and parse errors, is recorded with provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

from cn12.crystal import parse_cif_blocks
from cn12.models import CrystalBlock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = PROJECT_ROOT / "03_data" / "demo_cod"

MAX_CIF_BYTES = 20 * 1024 * 1024
MAX_ZIP_BYTES = 100 * 1024 * 1024
MAX_ZIP_MEMBERS = 2000
MAX_ZIP_UNCOMPRESSED = 300 * 1024 * 1024

COD_LICENSE = "CC0 1.0 (Crystallography Open Database; public domain dedication)"
COD_CITATION = ("Gražulis, S. et al. (2012). Crystallography Open Database (COD): an open-access "
                "collection of crystal structures and platform for world-wide collaboration. "
                "Nucleic Acids Research 40, D420-D427. https://www.crystallography.net/cod/")
COD_CIF_URL = "https://www.crystallography.net/cod/{cod_id}.cif"
COD_ID_PATTERN = re.compile(r"^[1-9]\d{6}$")


@dataclass
class InputFile:
    """One input file and everything known about where it came from."""

    name: str
    source: str  # "demo", "upload", "zip", "cod", "local"
    sha256: str
    size_bytes: int
    status: str  # "parsed", "skipped", "parse_error"
    reason: str = ""
    path: Path | None = None
    record_id: str | None = None
    archive: str | None = None
    retrieved_utc: str | None = None
    license: str | None = None
    citation: str | None = None
    entry_reference: str | None = None
    blocks: list[CrystalBlock] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.sha256[:12]}:{self.name}"

    def provenance(self) -> dict:
        return {
            "source": self.source,
            "filename": self.name,
            "archive": self.archive,
            "record_id": self.record_id,
            "retrieved_utc": self.retrieved_utc,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "parse_status": self.status,
            "parse_message": self.reason or None,
            "license": self.license,
            "citation": self.citation,
            "entry_reference": self.entry_reference,
        }


class Workspace:
    """Private temporary storage for one loaded dataset; ``close`` deletes it."""

    def __init__(self) -> None:
        self.directory = Path(tempfile.mkdtemp(prefix="atomenv_"))
        self.files: list[InputFile] = []

    def close(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)

    # ---- adding inputs -------------------------------------------------
    def add_bytes(self, name: str, data: bytes, source: str, **meta) -> InputFile:
        digest = sha256(data).hexdigest()
        record = InputFile(name=_display_name(name), source=source, sha256=digest,
                           size_bytes=len(data), status="skipped", **meta)
        problem = validate_cif_bytes(record.name, data)
        if problem:
            record.reason = problem
            self.files.append(record)
            return record
        if any(f.sha256 == digest and f.status == "parsed" for f in self.files):
            record.reason = "duplicate content of a file already loaded (same SHA-256)"
            self.files.append(record)
            return record
        path = self.directory / f"{digest[:16]}.cif"
        path.write_bytes(data)
        record.path = path
        _parse_into(record)
        if record.source == "cod" or record.source == "demo":
            record.entry_reference = record.entry_reference or _entry_reference(data)
        self.files.append(record)
        return record

    def add_zip(self, name: str, data: bytes) -> list[InputFile]:
        archive = _display_name(name)
        added: list[InputFile] = []
        if len(data) > MAX_ZIP_BYTES:
            added.append(self._skip(archive, data, "zip", f"archive larger than {MAX_ZIP_BYTES // 2**20} MB"))
            return added
        try:
            bundle = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as exc:
            added.append(self._skip(archive, data, "zip", f"not a readable ZIP archive ({exc})"))
            return added
        members = bundle.infolist()
        if len(members) > MAX_ZIP_MEMBERS:
            added.append(self._skip(archive, data, "zip", f"more than {MAX_ZIP_MEMBERS} entries"))
            return added
        total = 0
        for info in members:
            if info.is_dir():
                continue
            member = info.filename
            unsafe = _unsafe_member(member)
            if unsafe:
                added.append(self._skip(member, b"", "zip", unsafe, archive=archive))
                continue
            if not member.lower().endswith(".cif"):
                added.append(self._skip(member, b"", "zip", "not a .cif file", archive=archive))
                continue
            if info.file_size > MAX_CIF_BYTES:
                added.append(self._skip(member, b"", "zip", "member larger than the per-file limit", archive=archive))
                continue
            total += info.file_size
            if total > MAX_ZIP_UNCOMPRESSED:
                added.append(self._skip(member, b"", "zip", "archive expands beyond the total size limit; remaining members skipped", archive=archive))
                break
            try:
                with bundle.open(info) as stream:
                    content = stream.read(MAX_CIF_BYTES + 1)
            except Exception as exc:  # encrypted, corrupt, unsupported compression
                added.append(self._skip(member, b"", "zip", f"could not read member ({exc})", archive=archive))
                continue
            added.append(self.add_bytes(member, content, "zip", archive=archive))
        if not added:
            added.append(self._skip(archive, data, "zip", "archive contains no files"))
        return added

    def add_path(self, path: str | Path, source: str = "local", **meta) -> InputFile:
        path = Path(path)
        try:
            size = path.stat().st_size
            if size > MAX_CIF_BYTES:
                return self._skip(path.name, b"", source, "file larger than the per-file limit")
            data = path.read_bytes()
        except OSError as exc:
            return self._skip(path.name, b"", source, f"could not read file ({exc})")
        return self.add_bytes(path.name, data, source, **meta)

    def _skip(self, name: str, data: bytes, source: str, reason: str, **meta) -> InputFile:
        record = InputFile(name=_display_name(name), source=source, sha256=sha256(data).hexdigest(),
                           size_bytes=len(data), status="skipped", reason=reason, **meta)
        self.files.append(record)
        return record

    # ---- summaries -------------------------------------------------------
    @property
    def parsed(self) -> list[InputFile]:
        return [f for f in self.files if f.status == "parsed"]

    def counts(self) -> dict:
        return {
            "inputs": len(self.files),
            "parsed": len(self.parsed),
            "skipped": sum(f.status == "skipped" for f in self.files),
            "parse_errors": sum(f.status == "parse_error" for f in self.files),
            "blocks": sum(len(f.blocks) for f in self.parsed),
            "sites": sum(len(b.center_sites) for f in self.parsed for b in f.blocks),
        }


def validate_cif_bytes(name: str, data: bytes) -> str:
    """Return a reason the content cannot be a CIF, or an empty string."""
    if not name.lower().endswith(".cif"):
        return "not a .cif file name"
    if len(data) == 0:
        return "empty file"
    if len(data) > MAX_CIF_BYTES:
        return f"file larger than {MAX_CIF_BYTES // 2**20} MB"
    if b"\x00" in data[:65536]:
        return "binary content (NUL bytes); not a text CIF"
    if not re.search(rb"(?im)^\s*data_", data):
        return "no data_ block header found"
    return ""


def _parse_into(record: InputFile) -> None:
    try:
        record.blocks = parse_cif_blocks(record.path)
        record.status = "parsed"
        warnings = sorted({w for b in record.blocks for w in b.warnings})
        if warnings:
            record.reason = "; ".join(warnings)
    except Exception as exc:
        record.status = "parse_error"
        record.reason = str(exc).replace(str(record.path), record.name)


def _display_name(name: str) -> str:
    return PurePosixPath(str(name).replace("\\", "/")).name or "unnamed"


def _unsafe_member(member: str) -> str:
    normalized = member.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized) or ".." in parts:
        return "unsafe archive path (absolute or parent-directory reference)"
    if parts and (parts[0] == "__MACOSX" or PurePosixPath(normalized).name.startswith("._")):
        return "macOS archive metadata, not a CIF"
    return ""


def _entry_reference(data: bytes) -> str | None:
    """Original-publication reference from the CIF, for author acknowledgement."""
    try:
        import gemmi

        block = gemmi.cif.read_string(data.decode("utf-8", errors="replace")).sole_block()
        authors = [a.strip("'\" ") for a in block.find_values("_publ_author_name")]
        journal = (block.find_value("_journal_name_full") or "").strip("'\" ")
        year = (block.find_value("_journal_year") or "").strip("'\" ")
        title = (block.find_value("_publ_section_title") or "").strip("'\";\n ").replace("\n", " ")
        parts = [", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""), year, title, journal]
        text = ". ".join(p for p in parts if p)
        return text or None
    except Exception:
        return None


# ---- demo data -----------------------------------------------------------
def demo_manifest() -> dict:
    return json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))


def load_demo(workspace: Workspace) -> list[InputFile]:
    manifest = demo_manifest()
    out = []
    for entry in manifest["files"]:
        out.append(workspace.add_path(
            DEMO_DIR / entry["file"], source="demo", record_id=f"COD {entry['cod_id']}",
            retrieved_utc=entry["retrieved_utc"], license=COD_LICENSE, citation=COD_CITATION,
        ))
    return out


# ---- COD -------------------------------------------------------------------
def fetch_cod(workspace: Workspace, cod_id: str, timeout: float = 20.0) -> InputFile:
    """Download one COD entry by its 7-digit ID using the documented URL pattern.

    ``https://www.crystallography.net/cod/<ID>.cif`` is the coordinates URL given on
    the COD wiki "howtoquerycod" page (checked 2026-10-18). Only this single-record
    pattern is used; no search API is called and no user data is sent.
    """
    cod_id = str(cod_id).strip()
    if not COD_ID_PATTERN.match(cod_id):
        return workspace._skip(f"{cod_id or 'empty'}.cif", b"", "cod",
                               "COD IDs are 7-digit numbers, e.g. 9008468")
    import requests

    url = COD_CIF_URL.format(cod_id=cod_id)
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        response = requests.get(url, timeout=timeout, stream=True,
                                headers={"User-Agent": "AtomicEnvironmentExplorer/1.0 (research)"})
        if response.status_code == 404:
            return workspace._skip(f"{cod_id}.cif", b"", "cod", f"COD has no entry {cod_id} (HTTP 404)")
        response.raise_for_status()
        data = response.raw.read(MAX_CIF_BYTES + 1, decode_content=True)
    except Exception as exc:
        return workspace._skip(f"{cod_id}.cif", b"", "cod",
                               f"COD download failed; check the network connection ({type(exc).__name__}: {exc})")
    return workspace.add_bytes(f"{cod_id}.cif", data, "cod", record_id=f"COD {cod_id}",
                               retrieved_utc=retrieved, license=COD_LICENSE, citation=COD_CITATION)
