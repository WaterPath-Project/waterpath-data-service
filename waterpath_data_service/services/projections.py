from __future__ import annotations

import io
import json
import logging
import shutil
from pathlib import Path

import geopandas as gpd
import httpx
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from waterpath_data_service.services.prepare_spatial import prepare_spatial_inputs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Remote dataset URLs
# ---------------------------------------------------------------------------
_URBANIZATION_LEVEL0_FUTURE_URL = (
    "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data"
    "/refs/heads/main/world_admin_units_urbanisation_degree/data/world_urbanisation_level0_future.csv"
)
_POPULATION_FUTURE_URL = (
    "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data"
    "/refs/heads/main/world_population/data/world-population-future.csv"
)
_HDI_FUTURE_URL = (
    "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data"
    "/refs/heads/main/hdi/data/hdi_future.csv"
)
TREATMENT_FRACTIONS_URL = (
    "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data"
    "/refs/heads/main/treatment_fractions/data/treatment.csv"
)
_SANITATION_FUTURE_URL = (
    "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data"
    "/refs/heads/main/jmp_household_surveys/data/sanitation_combined_future.csv"
)
_TREATMENT_FUTURE_URL = (
    "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data"
    "/refs/heads/main/treatment_fractions/data/treatment_future.csv"
)
_LIVESTOCK_FUTURE_URL = (
    "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data"
    "/refs/heads/main/livestock_projections/data/livestock_future.csv"
)

_ASSUMPTIONS_URLS: dict[str, str] = {
    "urbanization": (
        "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data"
        "/refs/heads/main/world_admin_units_urbanisation_degree/data/assumptions.csv"
    ),
    "population": (
        "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data"
        "/refs/heads/main/world_population/data/assumptions.csv"
    ),
    "hdi": (
        "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data"
        "/refs/heads/main/hdi/data/assumptions.csv"
    ),
    "treatment_fractions": (
        "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data"
        "/refs/heads/main/treatment_fractions/data/assumptions.csv"
    ),
    "sanitation": (
        "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data"
        "/refs/heads/main/jmp_household_surveys/data/assumptions.csv"
    ),
}



# ---------------------------------------------------------------------------
# Remote data helpers
# ---------------------------------------------------------------------------

async def fetch_treatment_fractions_csv(alpha3_list: list[str]) -> pd.DataFrame:
    """Fetch the country-level treatment fractions dataset from GitHub.

    Columns: ``alpha3``, ``FractionPrimarytreatment``,
    ``FractionSecondarytreatment``, ``FractionTertiarytreatment``,
    ``FractionQuaternarytreatment``.

    ``FractionQuaternarytreatment`` is derived as
    ``max(0, 1 - Primary - Secondary - Tertiary)`` when absent from the source
    data, mirroring the same logic applied to future projections.

    Countries in *alpha3_list* that are absent from the remote CSV receive a
    fallback row with all fractions set to zero and
    ``FractionQuaternarytreatment = 1.0`` (conservative "no treatment data"
    assumption).

    The returned DataFrame is filtered to the requested *alpha3_list*.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(TREATMENT_FRACTIONS_URL, timeout=30)
        r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df[df["alpha3"].isin(alpha3_list)].copy()

    # Ensure FractionQuaternarytreatment column exists.
    col_lower = {c.lower(): c for c in df.columns}
    # Normalise any existing variant of the column name to the canonical spelling.
    for variant in ("fractionquartenarytreatment", "fractionquarternarytreatment"):
        if variant in col_lower and col_lower[variant] != "FractionQuaternarytreatment":
            df = df.rename(columns={col_lower[variant]: "FractionQuaternarytreatment"})
            col_lower = {c.lower(): c for c in df.columns}
    if "fractionquaternarytreatment" not in col_lower:
        primary   = col_lower.get("fractionprimarytreatment")
        secondary = col_lower.get("fractionsecondarytreatment")
        tertiary  = col_lower.get("fractiontertiarytreatment")
        if primary and secondary and tertiary:
            df["FractionQuaternarytreatment"] = (
                1.0
                - pd.to_numeric(df[primary],   errors="coerce").fillna(0)
                - pd.to_numeric(df[secondary], errors="coerce").fillna(0)
                - pd.to_numeric(df[tertiary],  errors="coerce").fillna(0)
            ).clip(lower=0.0)
        else:
            df["FractionQuaternarytreatment"] = 0.0

    # Add fallback rows for countries absent from the remote dataset so the
    # returned DataFrame always contains one row per requested alpha3.
    missing = set(alpha3_list) - set(df["alpha3"])
    if missing:
        logger.warning(
            "fetch_treatment_fractions_csv: no source data for %s — "
            "using fallback (FractionQuaternarytreatment=1.0, all others 0).",
            sorted(missing),
        )
        fallback = pd.DataFrame({
            "alpha3":                        sorted(missing),
            "FractionPrimarytreatment":      0.0,
            "FractionSecondarytreatment":    0.0,
            "FractionTertiarytreatment":     0.0,
            "FractionQuaternarytreatment":   1.0,
        })
        df = pd.concat([df, fallback], ignore_index=True)

    return df


async def fetch_treatment_future_csv(
    alpha3_list: list[str],
    ssp: str,
    year: int,
) -> pd.DataFrame:
    """Fetch projected treatment fractions from ``treatment_future.csv``.

    Filters to *alpha3_list*, *ssp*, and *year*.  If ``FractionQuaternarytreatment``
    is absent it is added as ``max(0, 1 - Primary - Secondary - Tertiary)``.

    Returns a DataFrame keyed by ``alpha3``.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(_TREATMENT_FUTURE_URL, timeout=30)
        r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    alpha3_col = next((c for c in df.columns if c.lower() == "alpha3"), None)
    scenario_col = next((c for c in df.columns if c.lower() in ("scenario", "ssp")), None)
    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    if alpha3_col is None:
        raise ValueError("treatment_future.csv is missing an alpha3 column.")

    ssp_norm = ssp.strip().upper()
    if scenario_col is not None:
        df = df[df[scenario_col].astype(str).str.strip().str.upper() == ssp_norm]
    if year_col is not None:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df = df[df[year_col] == year]
    df = df[df[alpha3_col].isin(alpha3_list)].copy()
    if alpha3_col != "alpha3":
        df = df.rename(columns={alpha3_col: "alpha3"})

    # Ensure FractionQuaternarytreatment column exists.
    col_lower = {c.lower(): c for c in df.columns}
    # Normalise any existing variant of the column name to the canonical spelling.
    for variant in ("fractionquartenarytreatment", "fractionquarternarytreatment"):
        if variant in col_lower and col_lower[variant] != "FractionQuaternarytreatment":
            df = df.rename(columns={col_lower[variant]: "FractionQuaternarytreatment"})
            col_lower = {c.lower(): c for c in df.columns}
    if "fractionquaternarytreatment" not in col_lower:
        primary   = col_lower.get("fractionprimarytreatment")
        secondary = col_lower.get("fractionsecondarytreatment")
        tertiary  = col_lower.get("fractiontertiarytreatment")
        if primary and secondary and tertiary:
            df["FractionQuaternarytreatment"] = (
                1.0
                - pd.to_numeric(df[primary],   errors="coerce").fillna(0)
                - pd.to_numeric(df[secondary], errors="coerce").fillna(0)
                - pd.to_numeric(df[tertiary],  errors="coerce").fillna(0)
            ).clip(lower=0.0)
        else:
            df["FractionQuaternarytreatment"] = 0.0

    drop_cols = [c for c in [scenario_col, year_col] if c in df.columns]
    return df.drop(columns=drop_cols)


async def fetch_livestock_future_csv(
    alpha3_list: list[str],
    ssp: str,
    year: int,
) -> pd.DataFrame:
    """Fetch projected livestock data from ``livestock_future.csv``.

    Filters to *alpha3_list*, *ssp*, and *year*.  Returns a DataFrame with
    ``alpha3`` plus all available projection columns (animal counts, manure
    fractions, production system fractions).

    The scenario and year columns are dropped from the result.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(_LIVESTOCK_FUTURE_URL, timeout=30)
        r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    alpha3_col = next((c for c in df.columns if c.lower() == "alpha3"), None)
    scenario_col = next((c for c in df.columns if c.lower() in ("scenario", "ssp")), None)
    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    if alpha3_col is None:
        raise ValueError("livestock_future.csv is missing an alpha3 column.")

    ssp_norm = ssp.strip().upper()
    if scenario_col is not None:
        df = df[df[scenario_col].astype(str).str.strip().str.upper() == ssp_norm]
    if year_col is not None:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df = df[df[year_col] == year]
    df = df[df[alpha3_col].isin(alpha3_list)].copy()
    if alpha3_col != "alpha3":
        df = df.rename(columns={alpha3_col: "alpha3"})

    drop_cols = [c for c in [scenario_col, year_col] if c in df.columns]
    return df.drop(columns=drop_cols)


async def fetch_sanitation_projection(
    alpha3_list: list[str],
    ssp: str,
    year: int,
) -> pd.DataFrame:
    """Fetch projected sanitation fractions for the given countries, SSP, and year.

    Source: ``sanitation_combined_future.csv``

    Returns a DataFrame filtered to *alpha3_list*, *ssp*, and *year* with the
    scenario and year columns dropped.  The result contains ``alpha3`` plus all
    sanitation fraction columns from the source file.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(_SANITATION_FUTURE_URL, timeout=30)
        r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))

    alpha3_col = next((c for c in df.columns if c.lower() == "alpha3"), None)
    scenario_col = next((c for c in df.columns if c.lower() in ("scenario", "ssp")), None)
    year_col = next((c for c in df.columns if c.lower() == "year"), None)

    if alpha3_col is None or scenario_col is None or year_col is None:
        raise ValueError(
            "sanitation_combined_future.csv is missing expected columns "
            "(alpha3, scenario/ssp, year). "
            f"Found: {df.columns.tolist()}"
        )

    ssp_norm = ssp.strip().upper()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")

    mask = (
        df[alpha3_col].isin(alpha3_list)
        & (df[scenario_col].astype(str).str.strip().str.upper() == ssp_norm)
        & (df[year_col] == year)
    )
    result = df[mask].drop(columns=[scenario_col, year_col]).copy()
    if alpha3_col != "alpha3":
        result = result.rename(columns={alpha3_col: "alpha3"})
    return result


async def fetch_urbanization_level0_projection(
    alpha3_list: list[str],
    ssp: str,
    year: int,
) -> dict[str, float]:
    """Return ``{alpha3: fraction_urban}`` for the given SSP/year.

    Source: ``world_urbanisation_level0_future.csv``
    Long-format CSV with columns (possibly no header): alpha3, scenario, year, fractionUrban.
    For sub-national analyses, urbanization is kept as-is (baseline); only call this for
    country-level (admin level 0) requests.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(_URBANIZATION_LEVEL0_FUTURE_URL, timeout=30)
        r.raise_for_status()
    # Detect whether the CSV has a header row.
    first_line = r.text.lstrip().split("\n")[0]
    has_header = first_line.lower().startswith("alpha3")
    if has_header:
        df = pd.read_csv(io.StringIO(r.text))
        # Normalise column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        fraction_col = next((c for c in df.columns if "urban" in c), df.columns[-1])
        scenario_col = next((c for c in df.columns if "scenario" in c or "ssp" in c), "scenario")
    else:
        df = pd.read_csv(
            io.StringIO(r.text),
            header=None,
            names=["alpha3", "scenario", "year", "fractionUrban"],
        )
        fraction_col = "fractionUrban"
        scenario_col = "scenario"
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    ssp_norm = ssp.strip().upper()
    mask = (
        df["alpha3"].isin(alpha3_list)
        & (df[scenario_col].astype(str).str.strip().str.upper() == ssp_norm)
        & (df["year"] == year)
    )
    sub = df[mask]
    return dict(zip(sub["alpha3"].astype(str), pd.to_numeric(sub[fraction_col], errors="coerce")))


async def fetch_fraction_under_five_projection(
    alpha3_list: list[str],
    year: int,
) -> dict[str, float]:
    """Return ``{alpha3: fraction_pop_under5}`` for the given year.

    Source: ``world-population-future.csv`` (SSP-independent, single projection trajectory).
    Columns: name, alpha3, totalPopulation, fractionUrban, year, fractionUnderFive.
    Only the *fractionUnderFive* column is used here.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(_POPULATION_FUTURE_URL, timeout=30)
        r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    mask = df["alpha3"].isin(alpha3_list) & (df["year"] == year)
    sub = df[mask]
    # Drop duplicates (there should be only one row per alpha3+year here)
    sub = sub.drop_duplicates(subset=["alpha3"])
    return dict(zip(sub["alpha3"].astype(str), pd.to_numeric(sub["fractionUnderFive"], errors="coerce")))


async def fetch_hdi_projection(
    alpha3_list: list[str],
    ssp: str,
    year: int,
) -> dict[str, float]:
    """Return ``{alpha3: hdi}`` for the given SSP/year.

    Source: ``hdi_future.csv`` – wide format:
    ``alpha3, scenario, 2025, 2030, 2050, 2100``.
    Country-level values are used for both national and sub-national analyses.
    """
    async with httpx.AsyncClient() as client:
        r = await client.get(_HDI_FUTURE_URL, timeout=30)
        r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    year_col = str(year)
    if year_col not in df.columns:
        logger.warning("Year column '%s' not found in hdi_future.csv; available: %s", year_col, df.columns.tolist())
        return {}
    ssp_norm = ssp.strip().upper()
    scenario_col = next((c for c in df.columns if c.lower() in ("scenario", "ssp")), "scenario")
    mask = (
        df["alpha3"].isin(alpha3_list)
        & (df[scenario_col].astype(str).str.strip().str.upper() == ssp_norm)
    )
    sub = df[mask].drop_duplicates(subset=["alpha3"])
    return dict(zip(sub["alpha3"].astype(str), pd.to_numeric(sub[year_col], errors="coerce")))


async def fetch_assumptions(dataset_keys: list[str]) -> list[dict]:
    """Fetch and merge assumption records from the requested dataset keys.

    Each assumptions.csv is semicolon-delimited with columns:
    ``id, scenario, year, admin_level, pathogen, assumption``

    Returns a de-duplicated list of assumption dicts.
    """
    seen_ids: set[str] = set()
    results: list[dict] = []
    async with httpx.AsyncClient() as client:
        for key in dataset_keys:
            url = _ASSUMPTIONS_URLS.get(key)
            if url is None:
                continue
            try:
                r = await client.get(url, timeout=30)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text), sep=";")
                for _, row in df.iterrows():
                    row_id = str(row.get("id", "")).strip()
                    if row_id and row_id not in seen_ids:
                        seen_ids.add(row_id)
                        results.append({
                            "id": row_id,
                            "scenario": str(row.get("scenario", "")).strip(),
                            "year": str(row.get("year", "")).strip(),
                            "admin_level": str(row.get("admin_level", "")).strip(),
                            "pathogen": str(row.get("pathogen", "")).strip(),
                            "assumption": str(row.get("assumption", "")).strip(),
                        })
            except Exception as exc:
                logger.warning("Could not fetch assumptions for '%s': %s", key, exc)
    return results


def read_schema_field_names(schema_path: Path) -> set[str]:
    """Return the set of field names defined in a Frictionless Table Schema JSON file.

    This is used to constrain which columns from a remote projection CSV are
    applied when updating a particular output file, ensuring outputs conform to
    their declared schemas.
    """
    with open(schema_path, encoding="utf-8") as fh:
        schema_doc = json.load(fh)
    return {field["name"] for field in schema_doc.get("fields", [])}


async def update_isodata_projected_variables(
    isodata_path: Path,
    ssp: str,
    year: int,
    is_country_level: bool,
) -> list[dict]:
    """Update fraction_urban_pop, fraction_pop_under5, and hdi in isodata.csv.

    - **fraction_urban_pop**: For country-level (admin 0) analysis, fetched from
      ``world_urbanisation_level0_future.csv`` for the given SSP/year.
      For sub-national analysis, the baseline degree is kept unchanged.
    - **fraction_pop_under5**: Fetched from ``world-population-future.csv`` (SSP-independent),
      filtered by year.  Mapped via the parent-country alpha3 (``iso_country`` or ``gid[:3]``).
    - **hdi**: Fetched from ``hdi_future.csv`` for the given SSP/year.  Country-level
      values are used for both national and sub-national analyses.

    The CSV at *isodata_path* is updated in-place.

    Returns a list of assumption dicts to include in the summary.
    """
    df = pd.read_csv(isodata_path)

    # Identify the alpha3 column (used to join projection datasets).
    alpha3_col: str | None = next(
        (c for c in ["iso_country", "alpha3"] if c in df.columns), None
    )
    if alpha3_col is None:
        # Fall back: if gids are 3-char they ARE alpha3.
        gid_col = next((c for c in ["gid"] if c in df.columns), None)
        if gid_col and df[gid_col].astype(str).str.len().max() <= 3:
            alpha3_col = gid_col
        else:
            # For sub-national GIDs like "UGA.1_1", take first 3 chars.
            if gid_col:
                df["_alpha3_tmp"] = df[gid_col].astype(str).str[:3]
                alpha3_col = "_alpha3_tmp"
    if alpha3_col is None:
        logger.warning("update_isodata_projected_variables: no alpha3 column found in %s", isodata_path)
        return []

    alpha3_list = df[alpha3_col].astype(str).str.strip().unique().tolist()

    dataset_keys = ["hdi", "population"]

    # --- HDI ------------------------------------------------------------
    if "hdi" in df.columns:
        hdi_map = await fetch_hdi_projection(alpha3_list, ssp, year)
        if hdi_map:
            updated = df[alpha3_col].astype(str).str.strip().map(hdi_map)
            df["hdi"] = updated.combine_first(df["hdi"])
            logger.info("Updated hdi for %d areas", updated.notna().sum())

    # --- fraction_pop_under5 -------------------------------------------
    fuf_col = next((c for c in ["fraction_pop_under5", "fractionUnderFive"] if c in df.columns), None)
    if fuf_col:
        fuf_map = await fetch_fraction_under_five_projection(alpha3_list, year)
        if fuf_map:
            updated = df[alpha3_col].astype(str).str.strip().map(fuf_map)
            df[fuf_col] = updated.combine_first(df[fuf_col])
            logger.info("Updated %s for %d areas", fuf_col, updated.notna().sum())

    # --- fraction_urban_pop (country-level only) -----------------------
    urb_col = next((c for c in ["fraction_urban_pop", "fractionUrban"] if c in df.columns), None)
    if urb_col:
        if is_country_level:
            dataset_keys.append("urbanization")
            urb_map = await fetch_urbanization_level0_projection(alpha3_list, ssp, year)
            if urb_map:
                updated = df[alpha3_col].astype(str).str.strip().map(urb_map)
                df[urb_col] = updated.combine_first(df[urb_col])
                logger.info("Updated %s for %d areas (country-level)", urb_col, updated.notna().sum())
        else:
            # Sub-national: keep baseline urbanization intact.
            dataset_keys.append("urbanization")

    # Drop temporary column if we added one.
    if "_alpha3_tmp" in df.columns:
        df = df.drop(columns=["_alpha3_tmp"])

    df.to_csv(isodata_path, index=False)

    return await fetch_assumptions(dataset_keys)


def _session_shapefile_path(session_dir: Path) -> Path:
    # Prefer the existing baseline layout used by the service.
    candidates = [
        session_dir / "baseline" / "geodata" / "geodata.shp",
        session_dir / "geodata" / "geodata.shp",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Session shapefile not found. Expected geodata.shp under "
        f"{candidates[0].parent} (or {candidates[1].parent})."
    )


def _population_tif_path(static_data_dir: Path, ssp: str, year: int) -> Path:
    """Return the path to the population GeoTIFF for the given SSP and year.

    Expected naming convention: ``FuturePop_<SSP>_<year>_1km_v0_2.tif``
    e.g. ``FuturePop_SSP3_2050_1km_v0_2.tif``
    """
    ssp_upper = ssp.strip().upper()
    if not ssp_upper.startswith("SSP"):
        raise ValueError("SSP must look like 'SSP1'..'SSP5'.")

    tif_name = f"FuturePop_{ssp_upper}_{year}_1km_v0_2.tif"
    tif_path = static_data_dir / "worldpop_2025" / tif_name
    if not tif_path.is_file():
        raise FileNotFoundError(
            f"Population TIF not found: {tif_path}. "
            f"Expected a file named '{tif_name}' in {static_data_dir}."
        )
    return tif_path


def generate_population_isoraster(
    *,
    session_dir: Path,
    scenario_dir: Path,
    static_data_dir: Path,
    ssp: str,
    year: int,
) -> Path:
    """Generate spatial population rasters for a scenario.

    Selects ``FuturePop_<SSP>_<year>_1km_v0_2.tif`` from ``static_data_dir``,
    then delegates to :func:`prepare_spatial_inputs` to produce:

    - ``<scenario_dir>/isoraster.tif``        – polygon-index raster
    - ``<scenario_dir>/human/pop_urban.tif``  – gridded urban population
    - ``<scenario_dir>/human/pop_rural.tif``  – gridded rural population

    The ``isodata.csv`` is read from the session's baseline ``population.csv``
    (``<session_dir>/baseline/human_emissions/population.csv``).

    Returns the path to ``isoraster.tif``.
    """
    tif_path = _population_tif_path(static_data_dir, ssp, year)
    shp_path = _session_shapefile_path(session_dir)

    isodata_path = session_dir / "baseline" / "human_emissions" / "population.csv"
    if not isodata_path.is_file():
        raise FileNotFoundError(
            f"Baseline population.csv not found at {isodata_path}. "
            "Run /input/generate first."
        )

    scenario_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating spatial rasters for %s/%s using %s",
        ssp, year, tif_path.name,
    )

    paths = prepare_spatial_inputs(
        geodata_path=str(shp_path),
        isodata_path=str(isodata_path),
        pop_raster_path=str(tif_path),
        out_dir=str(scenario_dir),
    )

    return Path(paths["isoraster"])


def generate_baseline_csv_projection(
    *,
    session_dir: Path,
    schema: str,
    scenario_dir: Path,
) -> Path:
    """Placeholder projection for tabular schemas.

    Copies the session baseline CSV from `baseline/human_emissions/<schema>.csv` into the scenario
    directory.
    """

    schema_norm = schema.strip().lower()
    if schema_norm not in {"sanitation", "treatment", "population"}:
        raise ValueError("Unsupported schema for CSV projection.")

    baseline_path = session_dir / "baseline" / "human_emissions" / f"{schema_norm}.csv"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Baseline CSV not found: {baseline_path}")

    scenario_dir.mkdir(parents=True, exist_ok=True)
    out_path = scenario_dir / f"{schema_norm}.csv"
    shutil.copyfile(baseline_path, out_path)
    return out_path


def calculate_zonal_population(
    raster_path: Path,
    shapefile_path: Path,
    gid_column: str = "gid",
) -> dict[str, float]:
    """Calculate total population from raster for each feature in shapefile.
    
    Args:
        raster_path: Path to the isoraster.tif file
        shapefile_path: Path to the session geodata shapefile
        gid_column: Name of the column containing area identifiers (default: "gid")
    
    Returns:
        Dictionary mapping gid -> total population (sum of raster pixels)
    """
    
    gdf = gpd.read_file(shapefile_path)
    if gdf.empty:
        raise ValueError("Shapefile contains no features.")
    
    # Auto-detect identifier column if gid_column not found
    if gid_column not in gdf.columns:
        # Try common identifier column names (including all GID levels)
        possible_columns = ["gid", "GID", "GID_4", "GID_3", "GID_2", "GID_1", "GID_0", 
                           "alpha3", "iso3", "ISO3", "HASC_1", "uid"]
        found_column = None
        for col in possible_columns:
            if col in gdf.columns:
                found_column = col
                break
        
        if found_column is None:
            # Fall back to using the index if no identifier column found
            gdf["_index_id"] = gdf.index.astype(str)
            gid_column = "_index_id"
        else:
            gid_column = found_column
    
    results = {}
    
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)
        raster_transform = src.transform
        raster_crs = src.crs
        nodata = src.nodata
        
        # Reproject shapefile to raster CRS if needed
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
        
        # Filter valid raster data once
        if nodata is not None:
            if abs(nodata) > 1e10:
                valid_mask = raster_data < 1e10
            else:
                valid_mask = ~np.isclose(raster_data, nodata, rtol=1e-5)
            valid_mask &= np.isfinite(raster_data)
        else:
            valid_mask = np.isfinite(raster_data)
        
        # Get pixel dimensions
        pixel_width = abs(raster_transform.a)
        pixel_height = abs(raster_transform.e)
        
        # Build an index of which features touch which pixels
        # For each valid pixel, calculate the proportion of each polygon
        height, width = raster_data.shape
        
        for idx, row in gdf.iterrows():
            gid = row[gid_column]
            geom = row.geometry
            
            if geom is None or geom.is_empty:
                results[str(gid)] = 0.0
                continue
            
            # Get bounding box of this geometry in pixel coordinates
            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            inv_transform = ~raster_transform
            col_min, row_max = inv_transform * (bounds[0], bounds[1])
            col_max, row_min = inv_transform * (bounds[2], bounds[3])
            
            col_min = max(0, int(np.floor(col_min)))
            row_min = max(0, int(np.floor(row_min)))
            col_max = min(width, int(np.ceil(col_max)) + 1)
            row_max = min(height, int(np.ceil(row_max)) + 1)
            
            total_pop = 0.0
            
            # For each pixel that might intersect this geometry
            for row_idx in range(row_min, row_max):
                for col_idx in range(col_min, col_max):
                    if not valid_mask[row_idx, col_idx]:
                        continue
                    
                    # Create pixel bounding box using the transform
                    pixel_minx = raster_transform.c + col_idx * raster_transform.a
                    pixel_maxy = raster_transform.f + row_idx * raster_transform.e
                    pixel_maxx = pixel_minx + raster_transform.a
                    pixel_miny = pixel_maxy + raster_transform.e  # e is negative
                    
                    pixel_box = box(pixel_minx, pixel_miny, pixel_maxx, pixel_maxy)
                    
                    # Calculate intersection area
                    try:
                        intersection = geom.intersection(pixel_box)
                        if intersection.is_empty:
                            continue
                        
                        intersection_area = intersection.area
                        pixel_area = pixel_box.area
                        
                        # Proportion of pixel covered by this geometry
                        proportion = intersection_area / pixel_area if pixel_area > 0 else 0.0
                        
                        # Add proportional population from this pixel
                        pixel_population = raster_data[row_idx, col_idx]
                        total_pop += pixel_population * proportion
                    except Exception:
                        # If intersection fails, skip this pixel
                        continue
            
            # Round to nearest integer since population represents people
            results[str(gid)] = round(total_pop)
    
    return results


def update_human_emissions_population(
    human_emissions_path: Path,
    isoraster_path: Path,
    scenario_dir: Path,
    shapefile_path: Path,
) -> None:
    """Update the population column in isodata.csv with values from pop_urban + pop_rural.

    Uses isoraster.tif as a zone mask: for each zone ID, sums all pixels across
    pop_urban.tif and pop_rural.tif that share that zone ID.  The zone-ID-to-GID
    mapping is derived from isodata.csv row order (zone ID == 1-based row index).

    Args:
        human_emissions_path: Path to the isodata.csv file to update.
        isoraster_path:        Path to isoraster.tif (integer zone-index raster).
        scenario_dir:          Directory containing pop_urban.tif and pop_rural.tif.
        shapefile_path:        Path to the session geodata shapefile (unused, kept for
                               API compatibility).
    """
    pop_urban_path = scenario_dir / "pop_urban.tif"
    pop_rural_path = scenario_dir / "pop_rural.tif"
    if not pop_urban_path.is_file() or not pop_rural_path.is_file():
        raise FileNotFoundError(
            f"pop_urban.tif / pop_rural.tif not found in {scenario_dir}. "
            "Run population projection first."
        )

    # ------------------------------------------------------------------
    # Build zone_id → GID mapping.
    # isodata.csv stores the integer raster zone index in the ``iso`` column
    # and the string area identifier in the ``gid`` column.  Map pixel value
    # directly to gid string so zonal sums can be written back to the CSV.
    # ------------------------------------------------------------------
    df_pre = pd.read_csv(human_emissions_path)
    if "iso" not in df_pre.columns:
        raise ValueError("isodata.csv missing required 'iso' column (integer zone index).")
    gid_col = "gid" if "gid" in df_pre.columns else next(
        (c for c in ["alpha3", "iso_country", "iso3"] if c in df_pre.columns), None
    )
    if gid_col is None:
        raise ValueError("isodata.csv missing a string identifier column (gid/alpha3/iso_country/iso3).")
    zone_to_gid: dict[int, str] = {}
    for _, row in df_pre.iterrows():
        try:
            zone_to_gid[int(row["iso"])] = str(row[gid_col]).strip()
        except (ValueError, TypeError):
            pass

    # ------------------------------------------------------------------
    # Read rasters and compute zonal sums.
    # ------------------------------------------------------------------
    with rasterio.open(isoraster_path) as src:
        zones = src.read(1).astype(np.int32)

    def _read_pop(path: Path) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            nodata = src.nodata
        if nodata is not None:
            arr[np.isclose(arr, nodata, rtol=1e-5) | (arr < -1e10)] = np.nan
        return arr

    total_pop = _read_pop(pop_urban_path)
    rural = _read_pop(pop_rural_path)
    # Sum urban + rural, treating NaN as 0
    total_pop = np.where(np.isfinite(total_pop), total_pop, 0.0) + \
                np.where(np.isfinite(rural), rural, 0.0)

    # Aggregate by zone ID
    unique_zones = np.unique(zones[zones > 0])
    zone_population: dict[str, int] = {}
    for zone_id in unique_zones:
        mask = zones == zone_id
        zone_sum = float(np.nansum(total_pop[mask]))
        gid = zone_to_gid.get(int(zone_id))
        if gid is not None:
            zone_population[gid] = round(zone_sum)

    # ------------------------------------------------------------------
    # Update the CSV.
    # ------------------------------------------------------------------
    df = pd.read_csv(human_emissions_path)
    csv_gid_column = next(
        (c for c in ["gid", "alpha3", "iso_country", "iso3"] if c in df.columns), None
    )
    if csv_gid_column is None:
        raise ValueError("isodata.csv missing required identifier column (gid/alpha3/iso_country/iso3)")
    if "population" not in df.columns:
        raise ValueError("isodata.csv missing 'population' column")

    # Capture baseline values before overwriting.  The CSV was copied from
    # the baseline directory immediately before this function was called, so
    # df["population"] is still the baseline population at this point.
    baseline_pop = df["population"].copy()
    df["population"] = df[csv_gid_column].astype(str).map(zone_population)
    # Zones with 0 projected population indicate a rasterization gap (tiny
    # polygon overwritten by neighbours) or a water/nodata TIF cell (e.g.
    # river islands).  Both are data failures – fall back to the baseline
    # value rather than writing a misleading zero.
    zero_mask = df["population"].isna() | (df["population"] == 0)
    if zero_mask.any():
        n_fallback = int(zero_mask.sum())
        logger.warning(
            "Projected population is 0 for %d area(s) (raster gap or water "
            "coverage) – retaining baseline values: %s",
            n_fallback,
            df.loc[zero_mask, csv_gid_column].tolist(),
        )
        df.loc[zero_mask, "population"] = baseline_pop[zero_mask]
    df["population"] = df["population"].fillna(0).astype(int)
    df.to_csv(human_emissions_path, index=False)
    logger.info(
        "Updated population in %s for %d areas (total: %s)",
        human_emissions_path.name, len(zone_population),
        f"{sum(zone_population.values()):,}",
    )

