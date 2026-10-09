import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
import zlib
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import geopandas as gpd
import numpy as np
import rasterio
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from waterpath_data_service.settings import settings
from waterpath_data_service.services.model_grid import align_model_grid


MONTHLY_VARIABLES = (
    "runoff", "discharge", "river_depth", "river_restime", "ssrd", "river_temperature",
)
REQUIRED_FILES = {
    f"{variable}/{variable}_m{month:02d}.tif"
    for variable in MONTHLY_VARIABLES for month in range(1, 13)
} | {"routing/flowdir.tif", "routing/flowacc.tif", "doc.tif"}
UNITS = {
    "runoff": "mm/day", "discharge": "m3/s", "river_depth": "m",
    "river_restime": "days", "ssrd": "kJ/m2/day", "river_temperature": "degC",
    "doc": "mg/L", "flowacc": "upstream_cell_count", "flowdir": "ESRI_D8",
}
PROVENANCE = "source.json"


def checked_session(root: Path, session_id: str) -> Path:
    if not session_id or session_id in {".", ".."} or any(
        character in session_id for character in "/\\:"
    ):
        raise HTTPException(422, "Invalid session_id.")
    session = root / session_id
    if not session.is_dir() or session.resolve().parent != root.resolve():
        raise HTTPException(404, "Session ID not found.")
    return session


@contextmanager
def session_lock(session: Path):
    lock = session / ".input-write-lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise HTTPException(409, "Another input operation is in progress for this session.")
    try:
        yield
    finally:
        lock.rmdir()


def source_info(folder: Path) -> dict:
    path = folder / PROVENANCE
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def replace_folder(staged: Path, destination: Path) -> None:
    def rename(source: Path, target: Path) -> None:
        for attempt in range(3):
            try:
                source.rename(target)
                return
            except PermissionError:
                if attempt == 2:
                    raise

    backup = staged.parent / "previous"
    existed = destination.exists()
    if existed:
        rename(destination, backup)
    try:
        rename(staged, destination)
    except Exception:
        if existed:
            rename(backup, destination)
        raise
    if existed:
        shutil.rmtree(backup)


def _extract(archive: Path, staging: Path) -> None:
    allowed = {f"hydrology/{name}" for name in REQUIRED_FILES} | {"hydrology/metadata.json"}
    directories = {"hydrology", *{str(PurePosixPath(name).parent) for name in allowed}}
    try:
        with zipfile.ZipFile(archive) as zipped:
            members = zipped.infolist()
            if len(members) > 100:
                raise HTTPException(413, "Too many ZIP entries.")
            seen = set()
            expanded = 0
            for member in members:
                name = member.filename
                normalized = name.rstrip("/")
                if (
                    "\\" in name or ":" in name or name.startswith("/")
                    or ".." in PurePosixPath(name).parts
                    or normalized.casefold() in seen
                    or stat.S_ISLNK(member.external_attr >> 16)
                    or member.flag_bits & 1
                    or member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or (member.is_dir() and normalized not in directories)
                    or (not member.is_dir() and name not in allowed)
                ):
                    raise HTTPException(422, f"Unsafe, duplicate or unexpected ZIP entry: {name!r}")
                seen.add(normalized.casefold())
                expanded += member.file_size
                if expanded > settings.hydrology_upload_expanded_bytes:
                    raise HTTPException(413, "Expanded ZIP exceeds configured limit.")
            missing = sorted({f"hydrology/{name}" for name in REQUIRED_FILES} - seen)
            if missing:
                raise HTTPException(422, {"message": "Missing hydrology rasters.", "files": missing})
            written = 0
            for member in members:
                if member.is_dir():
                    continue
                target = staging / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        if written > settings.hydrology_upload_expanded_bytes:
                            raise HTTPException(413, "Expanded ZIP exceeds configured limit.")
                        destination.write(chunk)
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError, zlib.error) as exc:
        raise HTTPException(422, "Invalid or unsupported hydrology ZIP.") from exc


def _validate(folder: Path, session: Path, target: Path) -> dict:
    from waterpath_data_service.services.hydrology import _d8_network

    shapefile = session / "baseline" / "geodata" / "geodata.shp"
    if not shapefile.is_file():
        raise HTTPException(422, "Generate baseline study-area geography before uploading hydrology.")
    bounds = gpd.read_file(shapefile).to_crs(4326).total_bounds
    signature = None
    grid = None
    decoded = 0
    for relative in sorted(REQUIRED_FILES):
        try:
            with rasterio.open(folder / relative) as source:
                transform = source.transform
                if (
                    source.driver != "GTiff" or source.count != 1
                    or source.crs is None or source.crs.to_epsg() != 4326
                    or not np.isfinite(tuple(transform)).all()
                    or transform.a <= 0 or transform.e >= 0
                    or transform.b != 0 or transform.d != 0
                    or not np.isclose(transform.a, -transform.e, rtol=1e-9, atol=0)
                    or np.dtype(source.dtypes[0]).kind not in "iuf"
                ):
                    raise HTTPException(422, f"Invalid single-band EPSG:4326 square grid: {relative}")
                pixels = source.width * source.height
                decoded += pixels * np.dtype(source.dtypes[0]).itemsize
                if pixels > settings.hydrology_upload_max_pixels or decoded > settings.hydrology_upload_expanded_bytes:
                    raise HTTPException(413, f"Decoded raster size exceeds configured limit: {relative}")
                current = (source.crs, transform, source.width, source.height)
                if signature is None:
                    signature = current
                    grid = {"crs": "EPSG:4326", "resolution": list(source.res),
                            "width": source.width, "height": source.height,
                            "bounds": list(source.bounds)}
                    if (source.bounds.left > bounds[0] + 1e-8 or source.bounds.bottom > bounds[1] + 1e-8
                            or source.bounds.right < bounds[2] - 1e-8 or source.bounds.top < bounds[3] - 1e-8):
                        raise HTTPException(422, f"Hydrology grid does not cover the study extent: {relative}")
                elif current != signature:
                    raise HTTPException(422, f"Hydrology grid mismatch: {relative}")
                valid_count = 0
                for _, window in source.block_windows(1):
                    values = source.read(1, window=window, masked=True).compressed()
                    if not np.isfinite(values).all():
                        raise HTTPException(422, f"Unmasked non-finite values: {relative}")
                    valid_count += values.size
                    if not relative.startswith("river_temperature/") and np.any(values < 0):
                        raise HTTPException(422, f"Negative values: {relative}")
                    if relative == "routing/flowdir.tif" and not np.isin(values, [0, 1, 2, 4, 8, 16, 32, 64, 128]).all():
                        raise HTTPException(422, f"Invalid ESRI D8 codes: {relative}")
                    if relative == "routing/flowacc.tif" and np.any(values != np.floor(values)):
                        raise HTTPException(422, f"Flow accumulation must be an upstream-cell count: {relative}")
                if not valid_count:
                    raise HTTPException(422, f"Raster has no valid data: {relative}")
        except rasterio.errors.RasterioError as exc:
            raise HTTPException(422, f"Unreadable GeoTIFF: {relative}") from exc
    with rasterio.open(folder / "routing/flowdir.tif") as source:
        flow = source.read(1, masked=True)
        try:
            _d8_network(flow.filled(0), ~np.ma.getmaskarray(flow))
        except ValueError as exc:
            raise HTTPException(422, "routing/flowdir.tif contains a routing cycle.") from exc
    declared = {}
    metadata = folder / "metadata.json"
    if metadata.is_file():
        if metadata.stat().st_size > 65536:
            raise HTTPException(413, "metadata.json exceeds 64 KiB.")
        try:
            declared = json.loads(metadata.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, "Invalid metadata.json.") from exc
        if (not isinstance(declared, dict) or set(declared) - {"source", "period", "notes", "units"}
                or any(not isinstance(value, str) for key, value in declared.items() if key != "units")
                or ("units" in declared and declared["units"] != UNITS)):
            raise HTTPException(422, "metadata.json must contain source/period/notes strings and optional documented units.")
    warnings = ["Units and scientific calibration are the uploader's responsibility; nodata coverage may vary by variable."]
    reference = (target / "isoraster.tif" if target.name != "baseline"
                 else target / "human_emissions/isoraster.tif")
    if not reference.is_file():
        reference = session / "baseline/human_emissions/isoraster.tif"
    if reference.is_file():
        with rasterio.open(reference) as source:
            if (source.crs, source.transform, source.width, source.height) != signature:
                warnings.append("Hydrology and emissions grids differ. Mixed-grid GloWPa execution is not guaranteed.")
    else:
        warnings.append("No emissions grid is available for a compatibility comparison.")
    return {"source": "custom_uploaded", "grid": grid, "units": UNITS,
            "declared_metadata": declared, "warnings": warnings, "raster_count": 75}


async def install_upload(upload, session: Path, target: Path) -> dict:
    if not (upload.filename or "").lower().endswith(".zip"):
        raise HTTPException(415, "file_id=hydrology requires a .zip archive.")
    with tempfile.TemporaryDirectory(prefix=".hydrology-", dir=session.parent) as temporary:
        staging = Path(temporary)
        archive = staging / "upload.zip"
        total = 0
        digest = hashlib.sha256()
        with archive.open("wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > settings.hydrology_upload_max_bytes:
                    raise HTTPException(413, "Hydrology upload exceeds configured limit.")
                digest.update(chunk)
                destination.write(chunk)
        await run_in_threadpool(_extract, archive, staging)
        report = await run_in_threadpool(_validate, staging / "hydrology", session, target)
        report["sha256"] = digest.hexdigest()
        package = staging / "package"
        await run_in_threadpool(shutil.copytree, target, package)
        if (package / "hydrology").exists():
            shutil.rmtree(package / "hydrology")
        (staging / "hydrology").rename(package / "hydrology")
        # Label the hydrology as custom before alignment, which treats generated hydrology differently.
        (package / "hydrology" / PROVENANCE).write_text(json.dumps(report, indent=2), encoding="utf-8")
        try:
            await run_in_threadpool(align_model_grid, package)
        except ValueError as exc:
            raise HTTPException(422, f"Model input preparation failed: {exc}") from exc
        report["warnings"] = [warning for warning in report["warnings"] if "grids differ" not in warning]
        (package / "hydrology" / PROVENANCE).write_text(json.dumps(report, indent=2), encoding="utf-8")
        replace_folder(package, target)
    return report


def preserve_custom(out_dir: Path, ssp, climate_model, experimental: bool, fine_path) -> dict | None:
    destination = out_dir / "hydrology"
    info = source_info(destination)
    baseline = out_dir.parent.parent / "baseline" / "hydrology" if ssp else None
    inherited = False
    if info.get("source") != "custom_uploaded":
        baseline_info = source_info(baseline) if baseline is not None else {}
        if baseline_info.get("source") == "custom_uploaded":
            info = baseline_info
            inherited = True
        elif info.get("source") != "custom_baseline_reused":
            return None
    if climate_model or experimental or fine_path:
        raise HTTPException(409, "Custom hydrology cannot be combined with climate_model or experimental raster controls.")
    if inherited:
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".hydrology-", dir=out_dir) as temporary:
            staged = Path(temporary) / "hydrology"
            shutil.copytree(baseline, staged)
            info = {**info, "source": "custom_baseline_reused",
                    "assumption": "Hydrology reused unchanged from custom baseline; not climate-projected."}
            (staged / PROVENANCE).write_text(json.dumps(info, indent=2), encoding="utf-8")
            replace_folder(staged, destination)
    return {"hydrology_dir": str(destination), "model": info["source"],
            "variables": {variable: [f"{variable}_m{month:02d}.tif" for month in range(1, 13)]
                          for variable in MONTHLY_VARIABLES},
            "downscaling": "disabled", "source": info}