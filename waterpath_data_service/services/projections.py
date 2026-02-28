from __future__ import annotations

import logging
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import box

from waterpath_data_service.services.prepare_spatial import prepare_spatial_inputs

logger = logging.getLogger(__name__)


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
    tif_path = static_data_dir / tif_name
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
    mapping is derived from the shapefile feature order (feature i → zone ID i+1).

    Args:
        human_emissions_path: Path to the isodata.csv file to update.
        isoraster_path:        Path to isoraster.tif (integer zone-index raster).
        scenario_dir:          Directory containing pop_urban.tif and pop_rural.tif.
        shapefile_path:        Path to the session geodata shapefile.
    """
    import geopandas as gpd

    pop_urban_path = scenario_dir / "pop_urban.tif"
    pop_rural_path = scenario_dir / "pop_rural.tif"
    if not pop_urban_path.is_file() or not pop_rural_path.is_file():
        raise FileNotFoundError(
            f"pop_urban.tif / pop_rural.tif not found in {scenario_dir}. "
            "Run population projection first."
        )

    # ------------------------------------------------------------------
    # Build zone_id → GID mapping from shapefile feature order.
    # prepare_spatial_inputs assigns iso_ids = [1, 2, …, n] in feature order.
    # ------------------------------------------------------------------
    gdf = gpd.read_file(shapefile_path)
    gid_candidates = ["GID_3", "GID_2", "GID_1", "GID_0", "gid", "alpha3"]
    gid_col = next((c for c in gid_candidates if c in gdf.columns), None)
    if gid_col is None:
        raise ValueError(f"No GID column found in shapefile. Available: {list(gdf.columns)}")
    # Use enumerate so zone_id is always 1,2,...,n regardless of DataFrame index.
    zone_to_gid: dict[int, str] = {
        zone_id: str(row[gid_col]).strip()
        for zone_id, (_, row) in enumerate(gdf.iterrows(), start=1)
    }

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

    df["population"] = df[csv_gid_column].astype(str).map(zone_population)
    if df["population"].isna().any():
        df["population"] = df["population"].fillna(0)
    df["population"] = df["population"].astype(int)
    df.to_csv(human_emissions_path, index=False)
    logger.info(
        "Updated population in %s for %d areas (total: %s)",
        human_emissions_path.name, len(zone_population),
        f"{sum(zone_population.values()):,}",
    )

