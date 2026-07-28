"""
qmra.py
=======
Generate QMRA (Quantitative Microbial Risk Assessment) spatial inputs for the
`GloWPaQMRA <https://github.com/microstijn/GloWPaQMRA>`_ R library.

The only *spatial data* input the QMRA engine derives from GDP is the
**treatment raster** (a "drinking-water map").  Its pixel values are
World-Bank income-group codes that ``qmra_ras_batch_fast()`` maps to a
treatment train:

======  ======================  =========================================
 code    income group            treatment regime (GloWPaQMRA)
======  ======================  =========================================
  5      low                     traditional
  6      lower-middle            conventional 1
  7      upper-middle            conventional 2
  8      high                    advanced
======  ======================  =========================================

The reclassification thresholds mirror ``data_generation_code`` (R):

    GDP per capita < 1045          -> 5
                   1045 .. 4126    -> 6
                   4126 .. 12736   -> 7
                   > 12736         -> 8

Baseline vs. projections
------------------------
Baseline GDP per capita comes from the Kummu et al. (Zenodo v4) gridded
dataset (``rast_adm2_gdp_perCapita_1990_2024.tif``, 5 arc-min, one band per
year 1990-2024), bundled under ``static/data/gdp_kummu_2025/``.  The dataset
carries no SSP futures, so projections apply a *national growth factor* from a
hosted ``gdp_future.csv`` (alpha3 x SSP x year) to the 2024 grid:

    projected_gdp(zone) = kummu_gdp_2024(zone) * gdp_future[a3, ssp, year]
                                               / gdp_future[a3, ssp, 2024]

This keeps the Kummu spatial pattern and absolute level (anchored to 2024) and
scales it by the country-level SSP trajectory, so baseline and projections
stay correlated.  The baseline product (nominally 2025) is the 2024 grid grown
one year under SSP2 ("middle of the road").

All outputs are aligned to the session ``isoraster.tif`` grid so the treatment
raster overlays the hydrology / pathogen-concentration rasters the QMRA run
consumes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds as window_from_bounds

from waterpath_data_service.services.prepare_spatial import _write_tif

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reclassification constants
# ---------------------------------------------------------------------------

#: World-Bank income-group upper thresholds -> GloWPaQMRA treatment code.
#: A GDP-per-capita value below a threshold takes that threshold's code;
#: values above the last threshold take :data:`_HIGH_INCOME_CODE`.
#: Thresholds mirror ``data_generation_code`` (World Bank Atlas groups).
INCOME_GROUP_THRESHOLDS: list[tuple[float, int]] = [
    (1045.0, 5),
    (4126.0, 6),
    (12736.0, 7),
]
_HIGH_INCOME_CODE = 8

#: Latest year present in the bundled Kummu v4 raster; the anchor band used
#: for SSP growth scaling.
BASE_YEAR = 2024
_GDP_FIRST_BAND_YEAR = 1990  # band 1 of the multiband GeoTIFF

#: Bundled baseline GDP raster (see module docstring).
_GDP_DATA_SUBDIR = "gdp_kummu_2025"
_GDP_RASTER_NAME = "rast_adm2_gdp_perCapita_1990_2024.tif"

_NODATA_INT = 0
_NODATA_FLOAT = -9999.0

# ---------------------------------------------------------------------------
# Static QMRA run configuration (from GloWPaQMRA/R/qmra_run.R)
# ---------------------------------------------------------------------------
# Everything the R engine needs beyond the treatment raster and the (external,
# GloWPa-produced) monthly concentration rasters is static run configuration.
# It is emitted as qmra_config.json so an R run is fully reproducible.

DEFAULT_QMRA_CONFIG: dict = {
    "pathogen": "rotavirus",             # 'cryptosporidium' | 'rotavirus'
    "model": "bp",                       # 'exp' | 'bp'
    "quantiles": [0.025, 0.5, 0.975],
    "mci": 10000,                        # Monte-Carlo iterations
    "routes": ["drinking"],
    "output_type": "monthly",            # 'monthly' | 'daily'
    "include_boiling": False,
    "boiling_lrv": {"min": 6, "max": 9},
    "include_immunological": False,
    "random_seed": 100,
    # Treatment codes present in the drinking-water map; the engine maps each to
    # a treatment train internally.
    "treatment_codes": [5, 6, 7, 8],
    # Beta-Poisson dose-response parameters.
    "bp_params": {
        "cryptosporidium": {
            "muw": -1.323, "muz": -0.206, "varw": 0.294, "varz": 1.054, "cov": -0.0625,
        },
        "rotavirus": {
            "muw": 0.571, "muz": -5.093, "varw": 0.677, "varz": 28.180, "cov": -2.728,
        },
    },
    # Daily volume / frequency parameters per exposure route.
    "exposure_groups": [
        {"name": "drinking", "route": "drinking", "type": "poisson",
         "lambda": 3.49, "glass": 250, "frequency": 365},
        {"name": "swimming", "route": "swimming", "type": "triangular",
         "min": 20, "mode": 35, "max": 50,
         "frequency": {"dist": "nbinom", "size": 0.4, "prob": 0.11}},
        {"name": "flooding", "route": "flooding", "type": "triangular",
         "min": 10, "mode": 100, "max": 300,
         "frequency": {"dist": "poisson", "lambda": 1}},
        {"name": "open_drain", "route": "open_drain", "type": "triangular",
         "min": 0.5, "mode": 3, "max": 20,
         "frequency": {"dist": "poisson", "lambda": 200}},
        {"name": "playing", "route": "playing", "type": "triangular",
         "min": 1, "mode": 10, "max": 50,
         "frequency": {"dist": "poisson", "lambda": 30}},
        {"name": "washing_clothes", "route": "washing_clothes", "type": "triangular",
         "min": 0.1, "mode": 1, "max": 5,
         "frequency": {"dist": "poisson", "lambda": 200}},
    ],
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def income_group_code(gdp_per_capita: float) -> int:
    """Map a GDP-per-capita value to a GloWPaQMRA treatment code (5-8).

    Returns :data:`_NODATA_INT` (0) for missing / non-finite input.
    """
    if gdp_per_capita is None or not np.isfinite(gdp_per_capita):
        return _NODATA_INT
    for threshold, code in INCOME_GROUP_THRESHOLDS:
        if gdp_per_capita < threshold:
            return code
    return _HIGH_INCOME_CODE


def gdp_raster_path(static_data_dir: str | Path) -> Path:
    """Absolute path to the bundled Kummu GDP-per-capita GeoTIFF."""
    return Path(static_data_dir) / _GDP_DATA_SUBDIR / _GDP_RASTER_NAME


def _band_for_year(year: int) -> int:
    """1-based band index for *year* in the multiband Kummu GeoTIFF."""
    return year - _GDP_FIRST_BAND_YEAR + 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_qmra_inputs(
    *,
    isoraster_path: str | Path,
    isodata_path: str | Path,
    static_data_dir: str | Path,
    out_dir: str | Path,
    gdp_future_df: pd.DataFrame | None = None,
    ssp: str,
    year: int,
    base_year: int = BASE_YEAR,
    config: dict | None = None,
    write_config: bool = False,
) -> dict:
    """Generate the QMRA drinking-water map and companion files.

    Parameters
    ----------
    isoraster_path:
        Session / scenario ``isoraster.tif`` (integer zone-index raster).  All
        outputs are written on this grid.
    isodata_path:
        ``population.csv`` (baseline) or scenario ``isodata.csv``.  Provides the
        ``iso`` -> ``gid`` mapping and the parent-country alpha3 (``iso_country``
        or ``gid[:3]``).
    static_data_dir:
        Root of ``static/data`` (locates the bundled Kummu GDP raster).
    out_dir:
        Destination ``qmra/`` folder.
    gdp_future_df:
        Country-level SSP GDP-per-capita projections filtered to *ssp*, with
        columns ``alpha3``, ``year``, ``gdp_per_capita``.  When ``None`` the raw
        Kummu ``base_year`` values are used (no growth scaling).
    ssp, year:
        Scenario labels; ``year`` selects the target growth year.
    base_year:
        Anchor year (must exist as a band in the Kummu raster and, for scaling,
        as a row in *gdp_future_df*).
    config:
        Overrides merged onto :data:`DEFAULT_QMRA_CONFIG` for ``qmra_config.json``.
    write_config:
        Write ``qmra_config.json``. This is enabled for baseline generation only.

    Returns
    -------
    dict with keys ``qmra_dir``, ``treatment_raster``, ``gdp_raster``,
    ``gdp_csv``, ``config``, ``treatment_codes`` and ``scaled`` (bool).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Read the zone-index raster (defines the output grid).
    # ------------------------------------------------------------------
    with rasterio.open(isoraster_path) as src:
        zones = src.read(1)
        transform = src.transform
        crs = src.crs
    height, width = zones.shape

    # ------------------------------------------------------------------
    # 2. iso (zone integer) -> gid, and gid -> alpha3, from isodata.
    # ------------------------------------------------------------------
    isodata_df = pd.read_csv(isodata_path, dtype=str)
    if "iso" not in isodata_df.columns or "gid" not in isodata_df.columns:
        raise ValueError(
            f"isodata at {isodata_path} must contain 'iso' and 'gid' columns."
        )
    alpha3_col = next(
        (c for c in ("iso_country", "alpha3") if c in isodata_df.columns), None
    )

    iso_to_gid: dict[int, str] = {}
    iso_to_alpha3: dict[int, str] = {}
    for _, row in isodata_df.iterrows():
        try:
            iso_val = int(row["iso"])
        except (ValueError, TypeError):
            continue
        gid = str(row["gid"]).strip()
        iso_to_gid[iso_val] = gid
        a3 = str(row[alpha3_col]).strip() if alpha3_col else gid[:3]
        iso_to_alpha3[iso_val] = (a3 or gid[:3])[:3].upper()

    # ------------------------------------------------------------------
    # 3. Resample the Kummu base-year GDP grid onto the zone grid.
    # ------------------------------------------------------------------
    gdp_path = gdp_raster_path(static_data_dir)
    if not gdp_path.is_file():
        raise FileNotFoundError(
            f"Baseline GDP raster not found at {gdp_path}. "
            "Bundle the Kummu v4 GeoTIFF under static/data/gdp_kummu_2025/."
        )
    gdp_grid = _resample_gdp_raster(
        gdp_path, _band_for_year(base_year), transform, height, width, crs
    )

    # ------------------------------------------------------------------
    # 4. Zonal mean GDP per admin area (base year).
    # ------------------------------------------------------------------
    zone_gdp_base: dict[int, float] = {}
    for iso_val in iso_to_gid:
        mask = zones == iso_val
        if not mask.any():
            continue
        vals = gdp_grid[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            zone_gdp_base[iso_val] = float(vals.mean())

    # ------------------------------------------------------------------
    # 5. Per-alpha3 national growth factor (target / base year).
    # ------------------------------------------------------------------
    growth = _build_growth_factors(gdp_future_df, base_year, year)
    scaled = bool(growth)

    # ------------------------------------------------------------------
    # 6. Project GDP, reclassify, paint onto the grid.
    # ------------------------------------------------------------------
    code_arr = np.full((height, width), _NODATA_INT, dtype=np.int32)
    gdp_out = np.full((height, width), _NODATA_FLOAT, dtype=np.float32)
    per_admin_rows: list[dict] = []

    for iso_val, gid in iso_to_gid.items():
        a3 = iso_to_alpha3.get(iso_val, gid[:3])
        base_gdp = zone_gdp_base.get(iso_val, np.nan)
        factor = growth.get(a3, 1.0)
        proj_gdp = base_gdp * factor if np.isfinite(base_gdp) else np.nan
        code = income_group_code(proj_gdp)

        mask = zones == iso_val
        if mask.any():
            code_arr[mask] = code
            if np.isfinite(proj_gdp):
                gdp_out[mask] = np.float32(proj_gdp)

        per_admin_rows.append({
            "gid": gid,
            "alpha3": a3,
            "gdp_per_capita": round(float(proj_gdp), 2) if np.isfinite(proj_gdp) else "",
            "income_group_code": code,
        })

    # ------------------------------------------------------------------
    # 7. Write outputs.
    # ------------------------------------------------------------------
    treatment_path = out_dir / "treatment.tif"
    gdp_tif_path = out_dir / "gdp_per_capita.tif"
    _write_tif(treatment_path, code_arr, transform, crs, nodata=_NODATA_INT, dtype=np.int32)
    _write_tif(gdp_tif_path, gdp_out, transform, crs, nodata=_NODATA_FLOAT, dtype=np.float32)

    gdp_csv_path = out_dir / "gdp_per_capita.csv"
    pd.DataFrame(per_admin_rows).to_csv(gdp_csv_path, index=False)

    present_codes = sorted(int(c) for c in np.unique(code_arr) if c != _NODATA_INT)
    config_path = out_dir / "qmra_config.json"
    if write_config:
        merged_config = {**DEFAULT_QMRA_CONFIG, **(config or {})}
        merged_config.update({
            "ssp": ssp,
            "year": year,
            "base_year": base_year,
            "growth_applied": scaled,
            "treatment_codes": present_codes or DEFAULT_QMRA_CONFIG["treatment_codes"],
            "treatment_raster": treatment_path.name,
            "income_group_thresholds": {str(t): c for t, c in INCOME_GROUP_THRESHOLDS},
        })
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(merged_config, fh, indent=2)
    else:
        config_path.unlink(missing_ok=True)

    logger.info(
        "QMRA inputs written to %s (ssp=%s year=%s, growth=%s, codes=%s)",
        out_dir, ssp, year, scaled, present_codes,
    )

    return {
        "qmra_dir": str(out_dir),
        "treatment_raster": str(treatment_path),
        "gdp_raster": str(gdp_tif_path),
        "gdp_csv": str(gdp_csv_path),
        "config": str(config_path) if write_config else None,
        "treatment_codes": present_codes,
        "scaled": scaled,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_growth_factors(
    gdp_future_df: pd.DataFrame | None,
    base_year: int,
    target_year: int,
) -> dict[str, float]:
    """Return ``{alpha3: target_gdp / base_gdp}`` from *gdp_future_df*.

    Countries missing either year (or with a non-positive base value) are
    omitted, so callers fall back to a factor of 1.0.
    """
    if gdp_future_df is None or gdp_future_df.empty:
        return {}
    required = {"alpha3", "year", "gdp_per_capita"}
    if not required.issubset(gdp_future_df.columns):
        logger.warning(
            "gdp_future_df missing columns %s; skipping growth scaling.",
            required - set(gdp_future_df.columns),
        )
        return {}

    df = gdp_future_df.copy()
    df["alpha3"] = df["alpha3"].astype(str).str.strip().str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["gdp_per_capita"] = pd.to_numeric(df["gdp_per_capita"], errors="coerce")

    pivot = df.pivot_table(
        index="alpha3", columns="year", values="gdp_per_capita", aggfunc="first"
    )
    growth: dict[str, float] = {}
    for a3, row in pivot.iterrows():
        base = row.get(base_year)
        tgt = row.get(target_year)
        if (
            base is not None and tgt is not None
            and np.isfinite(base) and np.isfinite(tgt) and base > 0
        ):
            growth[str(a3)] = float(tgt) / float(base)
    return growth


def _resample_gdp_raster(
    gdp_path: Path,
    band: int,
    dst_transform: rasterio.transform.Affine,
    dst_height: int,
    dst_width: int,
    dst_crs,
) -> np.ndarray:
    """Bilinearly resample one band of the GDP raster to the target grid.

    GDP per capita is an *intensive* quantity, so a plain bilinear average of
    values (no density conversion) is appropriate.  Source nodata becomes NaN.
    """
    dst_xmin, dst_ymin, dst_xmax, dst_ymax = rasterio.transform.array_bounds(
        dst_height, dst_width, dst_transform
    )
    with rasterio.open(gdp_path) as src:
        window = window_from_bounds(dst_xmin, dst_ymin, dst_xmax, dst_ymax, src.transform)
        src_data = src.read(
            band, window=window, boundless=True, fill_value=np.nan
        ).astype(np.float32)
        src_transform = src.window_transform(window)
        src_nodata = src.nodata
        src_crs = src.crs

    if src_nodata is not None:
        src_data[
            np.isclose(src_data, float(src_nodata), rtol=1e-5, atol=0)
            | (src_data < -1e10)
        ] = np.nan
    src_data[~np.isfinite(src_data)] = np.nan

    dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
    reproject(
        source=src_data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs or dst_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst
