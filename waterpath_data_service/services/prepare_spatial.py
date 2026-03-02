"""
prepare_spatial.py
==================
Python equivalent of the R prepare.R spatial functions for generating:

  - ``isoraster.tif``        – integer raster where every pixel holds the
                               ``iso`` value of the polygon it falls in,
                               as recorded in ``isodata.csv``.  Downstream
                               code joins raster pixels to CSV rows via
                               ``isodata["iso"] == pixel_value``, then
                               resolves the string area id from ``isodata["gid"]``.
  - ``human/pop_urban.tif``  – gridded urban population count.
  - ``human/pop_rural.tif``  – gridded rural population count.

The R workflow these functions replace uses:
  - ``geodata::gadm`` / ``geodata::world`` for boundaries
    → here the user supplies the shapefile from the case study folder.
  - ``WUP2018`` spreadsheets for urban fraction per country
    → here taken directly from ``isodata.csv``, which already contains
    ``fraction_urban_pop`` and ``population`` per polygon.
  - A gridded population raster (WorldPop or equivalent) for spatial
    disaggregation of population within polygons.

Gridded population source
--------------------------
The function accepts any population raster via ``pop_raster_path``.  The only
requirement is that pixel values represent **population counts per cell** (not
density) in WGS-84 (EPSG:4326).  The raster is converted internally to
population density (pop / km²) before bilinear resampling to the target
resolution, then back to counts – identical to the R logic.

Recommended sources by use-case
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
+----------------------------+----------+----------------------------------+
| Product                    | Native   | Suitable target ``res``          |
|                            | res      |                                  |
+============================+==========+==================================+
| WorldPop 2018 1 km         | ~0.0083° | 0.05° – 0.5° (city to national)  |
| ``ppp_2018_1km_            |          |                                  |
| Aggregated.tif``           |          |                                  |
+----------------------------+----------+----------------------------------+
| WorldPop 2020 100 m        | ~0.0009° | 0.01° – 0.1° (city / district)   |
| (country tiles)            |          |                                  |
+----------------------------+----------+----------------------------------+
| GHS-POP R2023A 1 km        | ~0.0083° | 0.05° – 0.5°                     |
| (GHSL, recommended for     |          |                                  |
| global runs)               |          |                                  |
+----------------------------+----------+----------------------------------+
| GPW v4 (2.5 / 5 / 10 arc- | 0.042° – | 0.1° – 1° (regional / global)    |
| minute tiles)              | 0.167°   |                                  |
+----------------------------+----------+----------------------------------+

Rule of thumb: the source raster resolution should be **≤ target ``res``**.
Using WorldPop 1 km at 0.5° is fine; using it at 1° or globally starts to
over-smooth small urban concentrations.  For large domains (> continental)
prefer GPW or GHS-POP at a matching coarser resolution.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List

import fiona
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds as window_from_bounds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prepare_spatial_inputs(
    geodata_path: str,
    isodata_path: str,
    pop_raster_path: str,
    out_dir: str,
    res: float | None = None,
) -> Dict[str, str]:
    """Generate ``isoraster.tif``, ``pop_urban.tif`` and ``pop_rural.tif``.

    Parameters
    ----------
    geodata_path:
        Path to the case-study ``.shp`` file (GADM-style, any admin level).
        Each feature is matched to ``isodata.csv`` via its GID property; the
        matching row's ``iso`` integer becomes the raster burn value.
    isodata_path:
        Path to the case-study ``isodata.csv``.  Must contain at minimum:
        - ``iso``                – polygon identifier matching the shapefile's
                                   highest-level GID field (e.g. ``GID_3``).
        - ``fraction_urban_pop`` – fraction of population that is urban (0–1).
        - ``population``         – total population for the polygon.
    pop_raster_path:
        Path to a gridded population raster in WGS-84 (EPSG:4326) where pixel
        values are **population counts per cell**.  See module docstring for
        recommended sources at different scales.
        Constraint: native resolution must be ≤ target ``res``.
    out_dir:
        Directory where output files are written.
    res:
        Target raster resolution in decimal degrees.  When ``None`` (default)
        the resolution is auto-selected so that the smallest dimension of the
        study-area bounding box spans at least 20 pixels, clamped to the range
        [0.001°, 0.5°].  Pass an explicit value to override.
        Typical values:
          - 0.5°  → ~55 km at equator  (national / global)
          - 0.1°  → ~11 km             (regional)
          - 0.01° → ~1 km              (city / district)

    Returns
    -------
    dict
        ``{"isoraster": path, "pop_urban": path, "pop_rural": path}``
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 – Read the shapefile
    # ------------------------------------------------------------------
    with fiona.open(geodata_path) as src:
        features = list(src)

    n_features = len(features)
    logger.info("Shapefile: %d features loaded from %s", n_features, geodata_path)

    # ------------------------------------------------------------------
    # Step 2 – Load isodata.csv.
    #
    # The iso column holds an integer zone index that matches the pixel
    # values burned into isoraster.tif.  The gid column holds the string
    # GID identifier.  This lets downstream code join the raster back to
    # the CSV using isodata["iso"] == raster_pixel_value, then resolve to
    # the string area id via isodata["gid"].
    # ------------------------------------------------------------------
    isodata_df = pd.read_csv(isodata_path, dtype=str)
    isodata_df["fraction_urban_pop"] = pd.to_numeric(
        isodata_df["fraction_urban_pop"], errors="coerce"
    )

    # gid string → iso integer (the value to burn into the raster)
    gid_to_iso: Dict[str, int] = {}
    for _, row in isodata_df.iterrows():
        try:
            gid_to_iso[str(row["gid"]).strip()] = int(row["iso"])
        except (ValueError, TypeError):
            pass

    # gid string → urban fraction (for poulating furban_arr in step 5)
    gid_to_furban: Dict[str, float] = {}
    for _, row in isodata_df.iterrows():
        frac = row["fraction_urban_pop"]
        if pd.notna(frac):
            gid_to_furban[str(row["gid"]).strip()] = float(frac)

    # Per shapefile feature: resolve GID string, look up iso integer burn value.
    # Priority: GID_3 > GID_2 > GID_1 > GID_0 (highest admin level present).
    iso_ids: List[int] = []
    id_to_gid: Dict[int, str] = {}
    for idx, feat in enumerate(features):
        props = feat["properties"]
        gid = str(
            props.get("GID_3")
            or props.get("GID_2")
            or props.get("GID_1")
            or props.get("GID_0")
            or str(idx + 1)
        ).strip()
        # Burn the iso integer from isodata.csv; fall back to feature order
        # if this polygon's GID is not present in isodata.csv.
        iso_val = gid_to_iso.get(gid, idx + 1)
        iso_ids.append(iso_val)
        id_to_gid[iso_val] = gid

    # ------------------------------------------------------------------
    # Step 3 – Determine target grid extent and transform
    # ------------------------------------------------------------------
    all_bounds = [fiona.bounds(f) for f in features]
    xmin_data = min(b[0] for b in all_bounds)
    ymin_data = min(b[1] for b in all_bounds)
    xmax_data = max(b[2] for b in all_bounds)
    ymax_data = max(b[3] for b in all_bounds)

    # Auto-select resolution when not supplied: aim for ≥ 20 pixels across the
    # smallest extent dimension, clamped to [0.001°, 0.5°].
    if res is None:
        extent_x = xmax_data - xmin_data
        extent_y = ymax_data - ymin_data
        min_extent = min(extent_x, extent_y)
        auto_res = min_extent / 20.0
        res = float(max(0.001, min(0.5, auto_res)))
        logger.info(
            "Auto-selected resolution: %.5f° (extent %.4f° × %.4f°)",
            res, extent_x, extent_y,
        )

    # Pad by one cell on every side (mirrors R's ``padding = 1``)
    xmin = max(math.floor(xmin_data / res) * res - res, -180.0)
    ymin = max(math.floor(ymin_data / res) * res - res,  -90.0)
    xmax = min(math.ceil(xmax_data  / res) * res + res,  180.0)
    ymax = min(math.ceil(ymax_data  / res) * res + res,   90.0)

    width  = round((xmax - xmin) / res)
    height = round((ymax - ymin) / res)
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    crs_out = "EPSG:4326"

    logger.info(
        "Target grid: %d × %d px at %.4f° | extent (%.4f, %.4f, %.4f, %.4f)",
        width, height, res, xmin, ymin, xmax, ymax,
    )

    # ------------------------------------------------------------------
    # Step 4 – Rasterize polygons → isoraster.tif
    # ------------------------------------------------------------------
    shapes = [(feat["geometry"], iso_ids[i]) for i, feat in enumerate(features)]
    isoraster_arr = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=True,  # matches R's ``touches = TRUE``
    )

    iso_path = out_dir / "isoraster.tif"
    _write_tif(iso_path, isoraster_arr, transform, crs_out, nodata=0, dtype=np.int32)
    logger.info("Written: %s", iso_path)

    domain_mask = isoraster_arr > 0

    # ------------------------------------------------------------------
    # Step 5 – Paint urban-fraction per pixel from isodata.csv
    # ------------------------------------------------------------------
    furban_arr = np.full((height, width), np.nan, dtype=np.float32)
    missing_gids = set()
    for iso_id, gid in id_to_gid.items():
        frac = gid_to_furban.get(gid)  # look up via string gid, not iso int
        if frac is not None and np.isfinite(frac):
            furban_arr[isoraster_arr == iso_id] = float(frac)
        else:
            missing_gids.add(gid)

    if missing_gids:
        logger.warning(
            "No fraction_urban_pop in isodata.csv for GID(s): %s", missing_gids
        )

    # ------------------------------------------------------------------
    # Step 6 – Resample population raster to target grid
    # ------------------------------------------------------------------
    pop_arr = _resample_pop_raster(pop_raster_path, transform, height, width, crs_out)
    pop_arr[~domain_mask] = np.nan

    # ------------------------------------------------------------------
    # Step 7 – Split into urban / rural
    # ------------------------------------------------------------------
    NODATA_F = -9999.0

    # Within the domain, treat NaN population (e.g. water pixels in the
    # source raster) as zero so no polygon ends up with nodata holes.
    pop_arr_domain = np.where(domain_mask & ~np.isfinite(pop_arr), 0.0, pop_arr)

    # A pixel is valid when both urb-fraction and population are known.
    valid = np.isfinite(furban_arr) & np.isfinite(pop_arr_domain)

    pop_urban_arr = np.where(valid, furban_arr * pop_arr_domain, NODATA_F).astype(np.float32)
    pop_rural_arr = np.where(valid, (1.0 - furban_arr) * pop_arr_domain, NODATA_F).astype(np.float32)

    pop_urban_path = out_dir / "pop_urban.tif"
    pop_rural_path = out_dir / "pop_rural.tif"
    _write_tif(pop_urban_path, pop_urban_arr, transform, crs_out, nodata=NODATA_F)
    _write_tif(pop_rural_path, pop_rural_arr, transform, crs_out, nodata=NODATA_F)
    logger.info("Written: %s", pop_urban_path)
    logger.info("Written: %s", pop_rural_path)

    return {
        "isoraster": str(iso_path),
        "pop_urban": str(pop_urban_path),
        "pop_rural": str(pop_rural_path),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cell_area_km2(
    transform: rasterio.transform.Affine, height: int, width: int
) -> np.ndarray:
    """Return a ``(height, width)`` array of cell areas in km² for a WGS-84
    lat/lon raster.  Area varies with latitude (cosine-dependence on cell width).

    Mirrors R's ``terra::cellSize(rast, unit="km")``.
    """
    a  = 6378.137          # WGS-84 semi-major axis (km)
    b  = 6356.752314245    # WGS-84 semi-minor axis (km)
    e2 = 1.0 - (b / a) ** 2

    res_lon_deg = abs(transform.a)
    res_lat_deg = abs(transform.e)

    lat_centers = transform.f + transform.e * (np.arange(height) + 0.5)
    lat_rad = np.radians(lat_centers)

    N = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)   # prime-vertical radius
    M = a * (1.0 - e2) / (1.0 - e2 * np.sin(lat_rad) ** 2) ** 1.5  # meridional

    cell_w = N * np.cos(lat_rad) * np.radians(res_lon_deg)
    cell_h = M * np.radians(res_lat_deg)

    return np.broadcast_to((cell_w * cell_h)[:, np.newaxis], (height, width)).copy()


def _resample_pop_raster(
    pop_raster_path: str,
    dst_transform: rasterio.transform.Affine,
    dst_height: int,
    dst_width: int,
    crs: str,
) -> np.ndarray:
    """Resample any population-count raster to the target grid.

    Converts source counts → density (pop / km²) → average resample →
    counts per target cell.  ``Resampling.average`` is used instead of
    bilinear because it skips NaN source pixels when computing each
    destination pixel; bilinear would propagate a single nodata/edge NaN
    to all four neighbouring destination pixels and blank out the domain.
    """
    dst_xmin, dst_ymin, dst_xmax, dst_ymax = rasterio.transform.array_bounds(
        dst_height, dst_width, dst_transform
    )

    with rasterio.open(pop_raster_path) as src:
        window = window_from_bounds(dst_xmin, dst_ymin, dst_xmax, dst_ymax, src.transform)
        # boundless=True ensures the read array always matches the window
        # dimensions, even when the window extends outside the source raster.
        # fill_value=0 keeps uninhabited out-of-extent cells as zero population
        # (they will be masked to NaN by the nodata step below or via nan
        # propagation in the density conversion).
        src_data = src.read(
            1, window=window, boundless=True, fill_value=0
        ).astype(np.float32)
        src_transform = src.window_transform(window)
        src_nodata = src.nodata

    if src_nodata is not None:
        # Use a tolerance-based comparison (float equality is unreliable) and
        # also catch common large-negative sentinel values.
        nodata_mask = (
            np.isclose(src_data, float(src_nodata), rtol=1e-5, atol=0)
            | (src_data < -1e10)
        )
        src_data[nodata_mask] = np.nan

    src_h, src_w = src_data.shape
    src_cell_area = _cell_area_km2(src_transform, src_h, src_w)
    with np.errstate(invalid="ignore", divide="ignore"):
        src_density = np.where(src_cell_area > 0, src_data / src_cell_area, np.nan)

    density_resampled = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=src_density,
        destination=density_resampled,
        src_transform=src_transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        # Resampling.average skips NaN source pixels when computing each dest
        # pixel, so a single nodata/edge NaN in the source does NOT propagate
        # to destroy its four neighbouring dest pixels (bilinear would do that).
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    dst_cell_area = _cell_area_km2(dst_transform, dst_height, dst_width)
    with np.errstate(invalid="ignore"):
        pop = density_resampled * dst_cell_area

    return pop.astype(np.float32)


def _write_tif(
    path: Path,
    arr: np.ndarray,
    transform: rasterio.transform.Affine,
    crs: str,
    nodata=None,
    dtype=None,
) -> None:
    """Write a single-band GeoTIFF with LZW compression."""
    if dtype is None:
        dtype = arr.dtype
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(arr.astype(dtype), 1)
