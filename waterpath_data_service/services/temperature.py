"""Temperature raster clipping using WorldClim MIROC6 bioclimatic TIFs.

Static data layout (under static/data/):

  vic_watch/
      1970_2000_Tair_year.asc        # Baseline mean annual temperature (~1970-2000)

  worldclim_2025/
      wc2.1_2.5m_bioc_MIROC6_ssp126_2021-2040.tif   # 19 BIO bands; BIO1 (band 1) = mean annual temp (°C)
      wc2.1_2.5m_bioc_MIROC6_ssp126_2041-2060.tif
      wc2.1_2.5m_bioc_MIROC6_ssp126_2061-2080.tif
      wc2.1_2.5m_bioc_MIROC6_ssp126_2081-2100.tif
      (ssp245 / ssp370 / ssp585 — same four periods each)

Source downloads:
  Baseline – https://worldclim.org/data/worldclim21.html
  Future   – https://worldclim.org/data/cmip6/cmip6_clim2.5m.html  (MIROC6)
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSP / year → WorldClim MIROC6 file-naming
# ---------------------------------------------------------------------------

_SSP_MAP: dict[str, str] = {
    "SSP1": "ssp126",
    "SSP2": "ssp245",
    "SSP3": "ssp370",
    "SSP4": "ssp585",   # SSP4-6.0 not in WorldClim CMIP6; fall back to ssp585
    "SSP5": "ssp585",
}

_YEAR_TO_PERIOD: dict[int, str] = {
    2030: "2021-2040",
    2050: "2041-2060",
    2100: "2081-2100",
}

# Band 1 in the WorldClim multi-band bioclimatic TIFs = BIO1 = mean annual temperature.
_BIO1_BAND = 1


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _baseline_temperature_path(static_data_dir: Path) -> Path:
    """Return the path to the vic_watch baseline temperature ASC."""
    path = static_data_dir / "vic_watch" / "1970_2000_Tair_year.asc"
    if not path.is_file():
        raise FileNotFoundError(
            f"Baseline temperature raster not found: {path}\n"
            "Expected vic_watch/1970_2000_Tair_year.asc under static/data/."
        )
    return path


def _future_temperature_path(static_data_dir: Path, ssp: str, year: int) -> Path:
    """Return the WorldClim MIROC6 multi-band TIF for *ssp* / *year*."""
    period = _YEAR_TO_PERIOD.get(year)
    if period is None:
        raise ValueError(
            f"No WorldClim future period defined for year {year}. "
            f"Supported years: {sorted(_YEAR_TO_PERIOD)}."
        )
    ssp_code = _SSP_MAP.get(ssp.strip().upper(), "ssp585")
    fname = f"wc2.1_2.5m_bioc_MIROC6_{ssp_code}_{period}.tif"
    path = static_data_dir / "worldclim_2025" / fname
    if not path.is_file():
        raise FileNotFoundError(
            f"WorldClim temperature TIF not found: {path}\n"
            f"Download {fname} from https://worldclim.org/data/cmip6/cmip6_clim2.5m.html "
            f"and place it under static/data/worldclim_2025/."
        )
    return path


# ---------------------------------------------------------------------------
# Core clip function
# ---------------------------------------------------------------------------

def generate_temperature_tif(
    shapefile_path: str | Path,
    source_raster_path: str | Path,
    out_dir: str | Path,
    template_raster_path: str | Path | None = None,
    band: int = _BIO1_BAND,
) -> Path:
    """Clip a temperature raster to the shapefile extent and write a GeoTIFF.

    Output: ``<out_dir>/temperature/temperature.tif``

    For the WorldClim multi-band bioclimatic TIFs, *band* selects which
    variable to extract (default 1 = BIO1 = mean annual temperature in °C).
    The baseline vic_watch ASC is single-band; the default ``band=1`` is
    correct for it as well.

    Parameters
    ----------
    shapefile_path:
        Polygon shapefile defining the study area (any CRS; reprojected
        internally to match the source raster).
    source_raster_path:
        Source temperature raster (ASC or GeoTIFF, single- or multi-band).
    out_dir:
        Parent output directory; ``temperature/`` subfolder is created.
    template_raster_path:
        Optional template raster. When provided, the clipped temperature is
        resampled/reprojected to this exact grid (extent, resolution, CRS).
    band:
        1-based band index to read from the source raster.

    Returns
    -------
    Path to the written ``temperature.tif``.
    """
    shapefile_path = Path(shapefile_path)
    source_raster_path = Path(source_raster_path)
    out_dir = Path(out_dir)
    template_raster_path = Path(template_raster_path) if template_raster_path else None

    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")

    out_tif_dir = out_dir / "temperature"
    out_tif_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_tif_dir / "temperature.tif"

    with rasterio.open(source_raster_path) as src:
        src_crs = src.crs or CRS.from_epsg(4326)

        # Reproject shapes into the source raster's CRS for masking.
        if src_crs.to_epsg() != 4326:
            shapes = list(gdf.to_crs(src_crs).geometry)
        else:
            shapes = list(gdf.geometry)

        fill_nodata = src.nodata if src.nodata is not None else -9999.0

        clipped, clip_transform = rasterio.mask.mask(
            src,
            shapes,
            crop=True,
            filled=True,
            indexes=band,
            nodata=fill_nodata,
        )
        # rasterio.mask with a scalar band index returns shape (rows, cols).
        if clipped.ndim == 3:
            clipped = clipped[0]

        write_arr = clipped.astype(np.float32)
        write_transform = clip_transform
        write_crs = src_crs
        write_width = write_arr.shape[1]
        write_height = write_arr.shape[0]

        # Keep temperature aligned with the model domain raster when provided.
        if template_raster_path and template_raster_path.is_file():
            with rasterio.open(template_raster_path) as tpl:
                tpl_crs = tpl.crs or src_crs
                aligned = np.full((tpl.height, tpl.width), fill_nodata, dtype=np.float32)
                reproject(
                    source=write_arr,
                    destination=aligned,
                    src_transform=write_transform,
                    src_crs=write_crs,
                    src_nodata=fill_nodata,
                    dst_transform=tpl.transform,
                    dst_crs=tpl_crs,
                    dst_nodata=fill_nodata,
                    resampling=Resampling.bilinear,
                )

                # If the template is an isoraster, keep temperature nodata where
                # the template has no modeled area.
                template_vals = tpl.read(1, masked=True)
                outside = np.zeros((tpl.height, tpl.width), dtype=bool)
                if np.ma.isMaskedArray(template_vals):
                    outside |= np.ma.getmaskarray(template_vals)
                if "isoraster" in tpl.name.lower():
                    outside |= np.asarray(template_vals.filled(0) <= 0)

                aligned[outside] = fill_nodata
                write_arr = aligned
                write_transform = tpl.transform
                write_crs = tpl_crs
                write_width = tpl.width
                write_height = tpl.height

        profile = {
            "driver": "GTiff",
            "dtype": "float32",
            "width": write_width,
            "height": write_height,
            "count": 1,
            "crs": write_crs,
            "transform": write_transform,
            "nodata": fill_nodata,
            "compress": "lzw",
            "predictor": 2,
        }

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(write_arr, 1)

    logger.info("Temperature TIF written \u2192 %s", out_path)
    return out_path
