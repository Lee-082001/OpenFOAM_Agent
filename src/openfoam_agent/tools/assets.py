"""User-authorized binary/text assets, independent of LLM case authoring."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
from pathlib import Path
from openfoam_agent.tools.workspace import WorkspaceSafetyError


def _regular_source(path: Path) -> Path:
    path = path.expanduser().absolute()
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise WorkspaceSafetyError("Asset sources must not contain symlink components.")
    if not path.is_file():
        raise WorkspaceSafetyError("Asset source is not a regular file.")
    return path


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def basic_quality(path: Path):
    result = {"level": "format_only", "watertightness_verified": False,
              "units_verified": False, "coordinate_frame_verified": False}
    if path.suffix.lower() == ".stl":
        with path.open("rb") as stream:
            header = stream.read(84)
            count = struct.unpack("<I", header[80:84])[0] if len(header) == 84 else 0
            if count and path.stat().st_size == 84 + 50 * count:
                low = [float("inf")] * 3; high = [float("-inf")] * 3
                for _ in range(count):
                    values = struct.unpack("<12fH", stream.read(50))[:12]
                    if not all(math.isfinite(v) for v in values):
                        raise WorkspaceSafetyError("Binary STL contains non-finite coordinates/normals.")
                    for vertex in (values[3:6], values[6:9], values[9:12]):
                        low = [min(a,b) for a,b in zip(low,vertex)]
                        high = [max(a,b) for a,b in zip(high,vertex)]
                result.update(format="binary_stl", triangles=count, bounds=[low, high])
                return result
        vertices = 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if len(line) > 65536:
                    raise WorkspaceSafetyError("STL line exceeds input bound.")
                if line.strip().startswith("vertex "):
                    values = [float(x) for x in line.split()[1:]]
                    if len(values) != 3 or not all(math.isfinite(x) for x in values):
                        raise WorkspaceSafetyError("Malformed ASCII STL vertex.")
                    vertices += 1
        if not vertices or vertices % 3:
            raise WorkspaceSafetyError("ASCII STL has no complete triangle vertices.")
        result.update(format="ascii_stl", triangles=vertices//3)
    elif path.suffix.lower() == ".csv":
        rows = 0; width = None
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) > 1024 or sum(len(x) for x in row) > 65536:
                    raise WorkspaceSafetyError("CSV row exceeds input bound.")
                width = len(row) if width is None else width
                if len(row) != width:
                    raise WorkspaceSafetyError("CSV has inconsistent column counts.")
                rows += 1
        result.update(format="csv", rows=rows, columns=width)
    else:
        result.update(format=path.suffix.lstrip(".") or "openfoam_dictionary",
                      note="No geometry/physical quality validation is claimed for this format.")
    return result


class AssetRegistry:
    def __init__(self, workspace, *, max_asset_bytes=256*1024*1024):
        self.workspace = workspace
        self.max_asset_bytes = max_asset_bytes
        self.path = workspace.root / "asset-registry.json"
        self.records = []
        if self.path.exists():
            if self.path.is_symlink():
                raise WorkspaceSafetyError("Asset registry may not be a symlink.")
            self.records = json.loads(self.path.read_text(encoding="utf-8"))
            for record in self.records:
                target = workspace.resolve_case_path(record["case_path"], must_exist=True)
                if _sha(target) != record["copy_sha256"]:
                    raise WorkspaceSafetyError("Previously registered asset copy changed.")
                workspace._asset_paths.add(record["case_path"])

    def import_file(self, source, case_path, *, approved_sources, units=None, coordinate_frame=None,
                    region_names=None, bindings=None):
        source = _regular_source(Path(source))
        approved = {Path(x).expanduser().absolute() for x in approved_sources}
        if source not in approved:
            raise WorkspaceSafetyError("Asset source was not explicitly authorized by the user.")
        if source.stat().st_size > self.max_asset_bytes:
            raise WorkspaceSafetyError("Asset exceeds the configured input size limit.")
        target = self.workspace.resolve_case_path(case_path)
        source_hash = _sha(source)
        for record in self.records:
            if record["case_path"] == case_path:
                if record["source_sha256"] != source_hash or record["copy_sha256"] != _sha(target):
                    raise WorkspaceSafetyError("Asset identity changed; explicit replacement/fork is required.")
                return record
        if target.exists():
            raise WorkspaceSafetyError("Asset import refuses to overwrite an existing case input.")
        quality = basic_quality(source)
        # Imported OpenFOAM dictionaries must pass the same executable-content policy.
        if source.suffix.lower() not in {".stl", ".obj", ".off", ".ply", ".csv", ".dat", ".table", ".emesh"}:
            if source.stat().st_size > self.workspace.max_file_bytes:
                raise WorkspaceSafetyError("Imported dictionary exceeds text-policy inspection limit.")
            self.workspace._validate_content(source.read_text(encoding="utf-8"), case_path)
        current_bytes = sum(x.size_bytes for x in self.workspace.execution_file_seals())
        if current_bytes + source.stat().st_size > self.workspace.max_execution_bytes:
            raise WorkspaceSafetyError("Asset would exceed the total input byte limit.")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".asset-", dir=target.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as dest, source.open("rb") as src:
                shutil.copyfileobj(src, dest, length=1024*1024)
                dest.flush(); os.fsync(dest.fileno())
            if _sha(temporary) != source_hash or _sha(source) != source_hash:
                raise WorkspaceSafetyError("Asset changed during import.")
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        record = {"source_path": str(source), "source_sha256": source_hash,
                  "case_path": case_path, "copy_sha256": source_hash,
                  "size_bytes": target.stat().st_size, "units": units,
                  "coordinate_frame": coordinate_frame, "region_names": region_names or [],
                  "bindings": bindings or [], "quality": quality}
        self.records.append(record); self.workspace._asset_paths.add(case_path)
        self._save()
        return record

    def import_case_tree(self, source, *, approved: bool):
        source = Path(source).expanduser().absolute()
        if not approved or source.is_symlink() or not source.is_dir():
            raise WorkspaceSafetyError("Existing case import requires an explicitly authorized directory.")
        files = []
        for name in ("0", "constant", "system"):
            base = source / name
            if base.is_symlink():
                raise WorkspaceSafetyError("Case import rejects symlink trees.")
            if base.exists():
                for path in sorted(base.rglob("*")):
                    if path.is_symlink():
                        raise WorkspaceSafetyError("Case import rejects symlink trees.")
                    if path.is_file():
                        files.append(path)
                    if len(files) > self.workspace.max_execution_files:
                        raise WorkspaceSafetyError("Case import exceeds input file-count limit.")
        if not files:
            raise WorkspaceSafetyError("No case inputs were found under 0/constant/system.")
        # Import only immutable inputs; existing time results/logs are deliberately excluded.
        for path in files:
            self.import_file(path, path.relative_to(source).as_posix(), approved_sources=files)
        return self.records

    def _save(self):
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.records, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, 0o600); os.replace(temporary, self.path)

    def public_summary(self):
        return [{key: value for key, value in row.items() if key != "source_path"} for row in self.records]


def ingest_request_assets(state, workspace):
    registry = AssetRegistry(workspace)
    request = state.user_request
    for source in request.geometry_files:
        registry.import_file(source, "constant/triSurface/" + Path(source).name,
                             approved_sources=request.geometry_files)
    for source in request.additional_files:
        registry.import_file(source, "constant/inputData/" + Path(source).name,
                             approved_sources=request.additional_files)
    if request.existing_case:
        registry.import_case_tree(request.existing_case, approved=True)
    state.assets = registry.public_summary()
    return registry
