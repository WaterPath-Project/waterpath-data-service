"""Hydrology input generation for GloWPa.

Clips global hydrology rasters from the bundled static data to the study-area
shapefile so that the resulting files match the GloWPa expected input layout.

Static data layout (under ``static/data/hydrology/``):

    models/
        {model}_{ssp_code}_{start}_{end}/
            runoff/           runoff_m01.tif … runoff_m12.tif
            discharge/        discharge_m01.tif … discharge_m12.tif
            river_depth/      river_depth_m01.tif … river_depth_m12.tif
            river_restime/    river_restime_m01.tif … river_restime_m12.tif
            ssrd/             ssrd_m01.tif … ssrd_m12.tif
            river_temperature/ river_temperature_m01.tif … (optional)
    routing/
        flowdir.tif
        flowacc.tif
    doc/
        doc.tif

Output layout (written under ``out_dir/hydrology/``):

    hydrology/
        runoff/           runoff_m01.tif … (clipped)
        discharge/        …
        river_depth/      …
        river_restime/    …
        ssrd/             …
        river_temperature/ … (when available)
        routing/
            flowdir.tif
            flowacc.tif
        doc.tif

SSP → ssp_code mapping follows the same convention used by temperature.py:

    SSP1 → ssp126,  SSP2 → ssp245,  SSP3 → ssp370,  SSP4/5 → ssp585

Model directory discovery:
    The function scans ``models/`` for directories whose names end with
    ``_{ssp_code}_{start}_{end}`` where ``start ≤ year ≤ end``.
    If no exact match is found, the first available directory is used as
    a fallback (with a warning).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from rasterio.warp import Resampling, reproject

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SSP_CODE_MAP: dict[str, str] = {
    "SSP1": "ssp126",
    "SSP2": "ssp245",
    "SSP3": "ssp370",
    "SSP4": "ssp585",
    "SSP5": "ssp585",
}

# Variable subdirectories inside a model folder (river_temperature is optional)
_MODEL_VARIABLE_DIRS: list[str] = [
    "runoff",
    "discharge",
    "river_depth",
    "river_restime",
    "ssrd",
    "river_temperature",
]

_ROUTING_FILES: list[str] = ["flowdir.tif", "flowacc.tif"]

# Regex to parse model directory names: anything_{ssp_code}_{start}_{end}
_MODEL_DIR_RE = re.compile(r"^.+_(ssp\d+)_(\d{4})_(\d{4})$")

# WorldClim year → period string used in filenames
_WORLDCLIM_YEAR_PERIODS: list[tuple[int, str]] = [
    (2040, "2021-2040"),
    (2060, "2041-2060"),
    (2080, "2061-2080"),
    (2100, "2081-2100"),
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _find_model_dir(static_hydrology_dir: Path, ssp: str, year: int) -> Path:
    """Return the best-matching model directory for *ssp* / *year*.

    Scans ``static_hydrology_dir/models/`` for directories whose name ends
    with ``_{ssp_code}_{start}_{end}`` where ``start ≤ year ≤ end``.
    Falls back to the first available directory when no exact match is found.

    Parameters
    ----------
    static_hydrology_dir:
        Root of the ``hydrology/`` subtree inside ``static/data/``.
    ssp:
        SSP identifier string (e.g. ``"SSP1"``).
    year:
        Target projection year (e.g. 2025).

    Returns
    -------
    Path to the selected model directory.

    Raises
    ------
    FileNotFoundError
        When ``models/`` does not exist or contains no directories.
    """
    ssp_code = _SSP_CODE_MAP.get(ssp.strip().upper(), "ssp126")
    models_dir = static_hydrology_dir / "models"

    if not models_dir.is_dir():
        raise FileNotFoundError(f"Hydrology models directory not found: {models_dir}")

    best: Path | None = None
    for candidate in sorted(models_dir.iterdir()):
        if not candidate.is_dir():
            continue
        match = _MODEL_DIR_RE.match(candidate.name)
        if match:
            dir_ssp, start_yr, end_yr = match.group(1), int(match.group(2)), int(match.group(3))
            if dir_ssp == ssp_code and start_yr <= year <= end_yr:
                best = candidate
                break

    if best is None:
        available = sorted(d for d in models_dir.iterdir() if d.is_dir())
        if not available:
            raise FileNotFoundError(
                f"No hydrology model directories found under {models_dir}."
            )
        best = available[0]
        logger.warning(
            "No hydrology model found for SSP=%s year=%d; falling back to '%s'.",
            ssp,
            year,
            best.name,
        )

    return best


def _baseline_model_dir(static_hydrology_dir: Path) -> Path:
    """Return the first available model directory (used for baseline generation).

    Raises
    ------
    FileNotFoundError
        When no model directories exist.
    """
    models_dir = static_hydrology_dir / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Hydrology models directory not found: {models_dir}")

    available = sorted(d for d in models_dir.iterdir() if d.is_dir())
    if not available:
        raise FileNotFoundError(
            f"No hydrology model directories found under {models_dir}."
        )
    return available[0]


# ---------------------------------------------------------------------------
# Clipping helpers
# ---------------------------------------------------------------------------

def _clip_raster(src_path: Path, shapes: list, out_path: Path, reference_path: Path | None = None) -> None:
    """Clip *src_path* to *shapes* and write a compressed GeoTIFF to *out_path*.

    Parameters
    ----------
    src_path:
        Source GeoTIFF (any CRS, any number of bands).
    shapes:
        List of Shapely geometries in EPSG:4326 used as the clip mask.
    out_path:
        Destination file path; parent directories are created as needed.
    reference_path:
        Optional reference raster.  When supplied the clipped data is
        reprojected / resampled to exactly match the reference grid
        (extent, resolution, CRS) so that all outputs are pixel-aligned.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(src_path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0

        # Reproject shapes to the source CRS when necessary.
        src_crs = src.crs
        # Global climate-model TIFs sometimes ship without an embedded CRS.
        # Fall back to EPSG:4326 (geographic, WGS-84) which is correct for all
        # global hydrology products used here.
        if src_crs is None:
            from rasterio.crs import CRS as _CRS
            src_crs = _CRS.from_epsg(4326)
            logger.debug("No CRS found in %s; assuming EPSG:4326.", src_path)

        if src_crs.to_epsg() != 4326:
            gdf_clip = gpd.GeoDataFrame(geometry=shapes, crs="EPSG:4326").to_crs(src_crs)
            clip_shapes = list(gdf_clip.geometry)
        else:
            clip_shapes = shapes

        clipped, clip_transform = rasterio.mask.mask(
            src,
            clip_shapes,
            crop=True,
            filled=True,
            nodata=nodata,
        )

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": clip_transform,
                "nodata": nodata,
                "compress": "lzw",
                "crs": src_crs,  # stamp assumed CRS so downstream readers see it
            }
        )

        # Align to the reference grid when a template raster is provided.
        if reference_path is not None and reference_path.is_file():
            with rasterio.open(reference_path) as ref:
                ref_crs = ref.crs or src_crs
                n_bands = clipped.shape[0]
                aligned = np.full(
                    (n_bands, ref.height, ref.width),
                    nodata,
                    dtype=clipped.dtype,
                )
                for band_idx in range(n_bands):
                    reproject(
                        source=clipped[band_idx],
                        destination=aligned[band_idx],
                        src_transform=clip_transform,
                        src_crs=src_crs,
                        src_nodata=nodata,
                        dst_transform=ref.transform,
                        dst_crs=ref_crs,
                        dst_nodata=nodata,
                        resampling=Resampling.bilinear,
                    )
                clipped = aligned
                out_meta.update(
                    {
                        "height": ref.height,
                        "width": ref.width,
                        "transform": ref.transform,
                        "crs": ref_crs,
                    }
                )

        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(clipped)


# ---------------------------------------------------------------------------
# River-temperature fallback from WorldClim bioclimatic data
# ---------------------------------------------------------------------------

def _worldclim_bioc_path(static_data_dir: Path, ssp: str | None, year: int | None) -> Path:
    """Return the WorldClim MIROC6 bioc TIF that best matches *ssp* / *year*.

    For baseline runs (``ssp=None`` / ``year=None``) the earliest available
    future file (ssp126, 2021-2040) is used as a near-present approximation.
    """
    wc_dir = static_data_dir / "worldclim_2025"
    if ssp is not None and year is not None:
        ssp_code = _SSP_CODE_MAP.get(ssp.strip().upper(), "ssp126")
        period = next(
            (p for threshold, p in _WORLDCLIM_YEAR_PERIODS if year <= threshold),
            _WORLDCLIM_YEAR_PERIODS[-1][1],
        )
    else:
        ssp_code = "ssp126"
        period = "2021-2040"

    fname = f"wc2.1_2.5m_bioc_MIROC6_{ssp_code}_{period}.tif"
    path = wc_dir / fname
    if not path.is_file():
        raise FileNotFoundError(
            f"WorldClim bioc file not found: {path}\n"
            "Download it from https://worldclim.org/data/cmip6/cmip6_clim2.5m.html "
            "and place it under static/data/worldclim_2025/."
        )
    return path


def _generate_river_temperature_fallback(
    shapes: list,
    static_data_dir: Path,
    out_dir: Path,
    ssp: str | None,
    year: int | None,
    reference_path: Path | None = None,
) -> list[str]:
    """Generate monthly river-temperature TIFs from WorldClim surface temperature.

    River temperature is approximated from WorldClim MIROC6 bioclimatic bands:

    * **BIO1** (band 1) – annual mean temperature (°C): used as the seasonal mean.
    * **BIO5** (band 5) – max temperature of the warmest month (°C).
    * **BIO6** (band 6) – min temperature of the coldest month (°C).

    A sinusoidal seasonal model is applied::

        T_river(m) = BIO1 + A · cos(2π · (m − peak_month) / 12)
        A          = (BIO5 − BIO6) / 2

    The peak month is 7 (July) in the Northern hemisphere and 1 (January) in
    the Southern hemisphere, determined from the centroid latitude of *shapes*.

    Output files are named ``triver_monmean_m{month:02d}.tif`` inside
    ``out_dir/river_temperature/`` to match the GloWPa expected layout.

    Parameters
    ----------
    shapes:
        Shapely geometries in EPSG:4326 representing the study area.
    static_data_dir:
        Root of ``static/data/``.
    out_dir:
        Parent output directory; ``river_temperature/`` is created inside it.
    ssp:
        SSP identifier (``"SSP1"`` … ``"SSP5"``).  ``None`` for baseline.
    year:
        Projection year.  ``None`` for baseline.

    Returns
    -------
    List of written filenames (``["triver_monmean_m01.tif", …]``).

    Notes
    -----
    *A* is derived from the difference between the *extreme* monthly
    temperatures (BIO5/BIO6), which tend to overestimate the amplitude of
    monthly *mean* temperatures.  This is a known limitation of the fallback.
    """
    wc_path = _worldclim_bioc_path(static_data_dir, ssp, year)
    logger.info("Using WorldClim fallback for river_temperature: %s", wc_path.name)

    out_var_dir = out_dir / "river_temperature"
    out_var_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(wc_path) as src:
        src_crs = src.crs
        if src_crs is None:
            from rasterio.crs import CRS as _CRS
            src_crs = _CRS.from_epsg(4326)
        if src_crs.to_epsg() != 4326:
            gdf_clip = gpd.GeoDataFrame(geometry=shapes, crs="EPSG:4326").to_crs(src_crs)
            clip_shapes = list(gdf_clip.geometry)
        else:
            clip_shapes = shapes

        nodata_val = src.nodata if src.nodata is not None else -9999.0

        # Read all 19 bands at once so we only clip the raster once.
        all_bands, clip_transform = rasterio.mask.mask(
            src,
            clip_shapes,
            crop=True,
            filled=True,
            nodata=nodata_val,
        )

        out_meta = src.meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "count": 1,
                "height": all_bands.shape[1],
                "width": all_bands.shape[2],
                "transform": clip_transform,
                "nodata": nodata_val,
                "compress": "lzw",
                "dtype": "float32",
            }
        )

    # WorldClim bands are 1-indexed; numpy arrays are 0-indexed.
    bio1 = all_bands[0].astype(np.float64)  # Annual mean temp (°C)
    bio5 = all_bands[4].astype(np.float64)  # Max temp of warmest month
    bio6 = all_bands[5].astype(np.float64)  # Min temp of coldest month

    valid = (bio1 != nodata_val) & (bio5 != nodata_val) & (bio6 != nodata_val)
    amplitude = np.where(valid, (bio5 - bio6) / 2.0, 0.0)

    # Determine seasonal phase from hemisphere (centroid latitude).
    centroid_lat = sum(geom.centroid.y for geom in shapes) / len(shapes)
    peak_month = 7 if centroid_lat >= 0 else 1

    # Pre-read reference grid dimensions once (avoids opening the file 12 times).
    ref_transform = ref_crs_val = ref_height = ref_width = None
    if reference_path is not None and reference_path.is_file():
        with rasterio.open(reference_path) as ref:
            ref_transform = ref.transform
            ref_crs_val = ref.crs or src_crs
            ref_height = ref.height
            ref_width = ref.width

    written: list[str] = []
    for month in range(1, 13):
        phase = 2.0 * np.pi * (month - peak_month) / 12.0
        t_monthly = np.where(valid, bio1 + amplitude * np.cos(phase), nodata_val).astype(np.float32)

        if ref_transform is not None:
            aligned = np.full((ref_height, ref_width), nodata_val, dtype=np.float32)
            reproject(
                source=t_monthly,
                destination=aligned,
                src_transform=clip_transform,
                src_crs=src_crs,
                src_nodata=nodata_val,
                dst_transform=ref_transform,
                dst_crs=ref_crs_val,
                dst_nodata=nodata_val,
                resampling=Resampling.bilinear,
            )
            write_arr = aligned
            write_meta = {
                **out_meta,
                "height": ref_height,
                "width": ref_width,
                "transform": ref_transform,
                "crs": ref_crs_val,
            }
        else:
            write_arr = t_monthly
            write_meta = out_meta

        out_fname = f"triver_monmean_m{month:02d}.tif"
        out_path = out_var_dir / out_fname
        with rasterio.open(out_path, "w", **write_meta) as dst:
            dst.write(write_arr[np.newaxis])
        written.append(out_fname)

    logger.info(
        "River-temperature fallback: wrote %d monthly files to '%s'.",
        len(written),
        out_var_dir,
    )
    return written


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_hydrology_inputs(
    shapefile_path: str | Path,
    static_data_dir: str | Path,
    out_dir: str | Path,
    ssp: str | None = None,
    year: int | None = None,
    reference_raster_path: str | Path | None = None,
) -> dict:
    """Clip hydrology inputs to the study area and write to ``out_dir/hydrology/``.

    Called both for baseline generation (``ssp=None``, ``year=None``) and for
    scenario projections (``ssp`` and ``year`` provided).

    Parameters
    ----------
    shapefile_path:
        Study-area polygon shapefile (any CRS).
    static_data_dir:
        Root of ``static/data/`` (must contain a ``hydrology/`` subdirectory).
    out_dir:
        Parent output directory; a ``hydrology/`` sub-folder is created inside it.
    ssp:
        SSP identifier (``"SSP1"`` … ``"SSP5"``).  When *None* the first
        available model directory is used (suitable for baseline runs).
    year:
        Projection year.  When *None* the first available model directory is used.

    Returns
    -------
    dict with keys:

    * ``hydrology_dir`` – absolute path of the written ``hydrology/`` directory.
    * ``model``         – name of the model directory that was used.
    * ``variables``     – mapping of variable name → list of clipped filenames.
    * ``routing``       – mapping of routing filename → absolute output path.
    * ``doc``           – absolute path of ``doc.tif``, or ``None``.
    """
    shapefile_path = Path(shapefile_path)
    static_data_dir = Path(static_data_dir)
    out_dir = Path(out_dir)
    reference_raster_path = Path(reference_raster_path) if reference_raster_path else None

    hydrology_static_dir = static_data_dir / "hydrology"

    # Resolve model directory.
    if ssp is not None and year is not None:
        model_dir = _find_model_dir(hydrology_static_dir, ssp, year)
    else:
        model_dir = _baseline_model_dir(hydrology_static_dir)

    logger.info("Generating hydrology inputs from model '%s'.", model_dir.name)

    # Read shapefile shapes (EPSG:4326) for masking.
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    shapes = list(gdf.geometry)

    out_hydrology_dir = out_dir / "hydrology"
    out_hydrology_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {
        "hydrology_dir": str(out_hydrology_dir),
        "model": model_dir.name,
        "variables": {},
        "routing": {},
        "doc": None,
    }

    # ------------------------------------------------------------------
    # Clip monthly variable rasters
    # ------------------------------------------------------------------
    for var_dir_name in _MODEL_VARIABLE_DIRS:
        var_src_dir = model_dir / var_dir_name
        if not var_src_dir.is_dir():
            logger.debug(
                "Variable directory '%s' not found in model '%s'; skipping.",
                var_dir_name,
                model_dir.name,
            )
            continue

        var_out_dir = out_hydrology_dir / var_dir_name
        var_out_dir.mkdir(parents=True, exist_ok=True)
        clipped_files: list[str] = []

        for tif_path in sorted(var_src_dir.glob("*.tif")):
            out_tif = var_out_dir / tif_path.name
            try:
                _clip_raster(tif_path, shapes, out_tif, reference_path=reference_raster_path)
                clipped_files.append(tif_path.name)
            except Exception as exc:
                logger.warning("Failed to clip %s: %s", tif_path, exc)

        if clipped_files:
            result["variables"][var_dir_name] = clipped_files

    # ------------------------------------------------------------------
    # River-temperature fallback: use WorldClim surface temperature when
    # the model directory does not contain river_temperature rasters.
    # ------------------------------------------------------------------
    if "river_temperature" not in result["variables"]:
        try:
            fallback_files = _generate_river_temperature_fallback(
                shapes=shapes,
                static_data_dir=static_data_dir,
                out_dir=out_hydrology_dir,
                ssp=ssp,
                year=year,
                reference_path=reference_raster_path,
            )
            if fallback_files:
                result["variables"]["river_temperature"] = fallback_files
                result["river_temperature_source"] = "worldclim_fallback"
        except Exception as exc:
            logger.warning(
                "River-temperature WorldClim fallback failed: %s", exc
            )

    # ------------------------------------------------------------------
    # Clip routing rasters
    # ------------------------------------------------------------------
    routing_src_dir = hydrology_static_dir / "routing"
    if routing_src_dir.is_dir():
        routing_out_dir = out_hydrology_dir / "routing"
        routing_out_dir.mkdir(parents=True, exist_ok=True)
        for fname in _ROUTING_FILES:
            src = routing_src_dir / fname
            if src.is_file():
                out_f = routing_out_dir / fname
                try:
                    _clip_raster(src, shapes, out_f, reference_path=reference_raster_path)
                    result["routing"][fname] = str(out_f)
                except Exception as exc:
                    logger.warning("Failed to clip routing file '%s': %s", src, exc)

    # ------------------------------------------------------------------
    # Clip DOC raster
    # ------------------------------------------------------------------
    doc_src = hydrology_static_dir / "doc" / "doc.tif"
    if doc_src.is_file():
        out_doc = out_hydrology_dir / "doc.tif"
        try:
            _clip_raster(doc_src, shapes, out_doc, reference_path=reference_raster_path)
            result["doc"] = str(out_doc)
        except Exception as exc:
            logger.warning("Failed to clip DOC raster: %s", exc)

    logger.info(
        "Hydrology inputs written to '%s' (variables: %s).",
        out_hydrology_dir,
        list(result["variables"]),
    )
    return result
