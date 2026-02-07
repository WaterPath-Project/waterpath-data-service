from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.features import geometry_mask, rasterize
from shapely.geometry import box


def _session_shapefile_path(session_dir: Path) -> Path:
    # Prefer the existing baseline layout used by the service.
    candidates = [
        session_dir / "default" / "geodata" / "geodata.shp",
        session_dir / "geodata" / "geodata.shp",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Session shapefile not found. Expected geodata.shp under "
        f"{candidates[0].parent} (or {candidates[1].parent})."
    )


def _population_nc4_path(static_data_dir: Path, ssp: str) -> Path:
    # Static files use lowercase ssp, e.g. population_ssp1soc_...
    ssp_lower = ssp.strip().lower()
    if not ssp_lower.startswith("ssp"):
        raise ValueError("SSP must look like 'SSP1'..'SSP5'.")

    nc4_name = f"population_{ssp_lower}soc_2.5min_annual_2006-2100.nc4"
    nc4_path = static_data_dir / nc4_name
    if not nc4_path.is_file():
        raise FileNotFoundError(f"Population nc4 file not found: {nc4_path}")
    return nc4_path


def _rank_netcdf_dataset_paths(nc4_path: Path) -> list[str]:
    """Return a ranked list of dataset paths to try opening.

    NetCDF files frequently contain multiple subdatasets (e.g. lat/lon vectors
    plus the actual data cube). If we naively open the first subdataset we can
    end up clipping a 1xN or Nx1 grid.
    """

    container = rasterio.open(nc4_path)
    subdatasets = list(getattr(container, "subdatasets", []) or [])
    container.close()

    if not subdatasets:
        # No subdatasets means single-layer NetCDF; open the file directly
        return [str(nc4_path)]

    def name_score(path_str: str) -> int:
        lowered = str(path_str).lower()
        score = 0
        # Strongly prefer population data
        if "population" in lowered or ":pop" in lowered or "_pop" in lowered:
            score += 500
        # Avoid coordinate/time vectors
        if "lat" in lowered or "lon" in lowered or "time" in lowered:
            score -= 200
        return score

    return sorted(subdatasets, key=name_score, reverse=True)


def generate_population_isoraster(
    *,
    session_dir: Path,
    scenario_dir: Path,
    static_data_dir: Path,
    ssp: str,
    year: int,
) -> Path:
    """Create a clipped population raster for a session.

    - Selects the correct SSP NetCDF (.nc4) from static/data
    - Selects the band corresponding to `year` (assuming bands map to 2006..2100)
    - Clips/masks by the session shapefile geometry
    - Writes GeoTIFF named `isoraster.tif` into the scenario directory

    Returns the output path.
    """

    if year < 2006 or year > 2100:
        raise ValueError("Population NetCDF supports years 2006..2100.")

    nc4_path = _population_nc4_path(static_data_dir, ssp)
    shp_path = _session_shapefile_path(session_dir)

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Session shapefile contains no features.")

    # Clean up potentially invalid geometries (self-intersections, etc.)
    # which can otherwise result in unexpected clipping artifacts.
    try:
        gdf["geometry"] = gdf.geometry.buffer(0)
    except Exception:
        # If buffer(0) fails for any feature, we'll rely on rasterio's masking.
        pass

    out_path = scenario_dir / "isoraster.tif"

    # NetCDF band index is assumed to map 2006..2100 -> 1..95.
    band_index = (year - 2006) + 1

    # Union geometries to avoid tiny gaps between multipart features.
    union_geom = gdf.geometry.unary_union
    geometries = [union_geom] if union_geom is not None else []
    if not geometries:
        raise ValueError("No valid geometries found in session shapefile.")

    base_shapes = [geom.__geo_interface__ for geom in geometries]
    candidate_paths = _rank_netcdf_dataset_paths(nc4_path)

    last_error: Exception | None = None
    for candidate in candidate_paths:
        try:
            with rasterio.open(candidate) as src:
                # Avoid lat/lon/time vector subdatasets.
                if src.width <= 1 or src.height <= 1:
                    continue
                # Verify we have enough bands for the requested year
                if src.count < band_index:
                    last_error = ValueError(
                        f"Dataset has only {src.count} bands; need band {band_index} for year {year}"
                    )
                    continue

                # Some NetCDF rasters come through without CRS metadata.
                # These population grids are typically lat/lon.
                src_crs = src.crs
                if src_crs is None:
                    src_crs = CRS.from_epsg(4326)

                shapes = base_shapes
                if gdf.crs is not None and src_crs is not None and gdf.crs != src_crs:
                    shapes_gdf = gdf.to_crs(src_crs)
                    shapes = [
                        geom.__geo_interface__
                        for geom in shapes_gdf.geometry
                        if geom is not None
                    ]
                else:
                    shapes_gdf = gdf

                # Check if shapefile area is very small relative to raster resolution.
                # For small areas, we need to resample to finer resolution first.
                px = abs(src.transform.a)
                py = abs(src.transform.e)
                pixel_size = max(px, py)
                
                nodata_value = src.nodata if src.nodata is not None else -9999
                
                shapefile_bounds = shapes_gdf.total_bounds  # [minx, miny, maxx, maxy]
                shapefile_width = shapefile_bounds[2] - shapefile_bounds[0]
                shapefile_height = shapefile_bounds[3] - shapefile_bounds[1]
                shapefile_diagonal = (shapefile_width**2 + shapefile_height**2) ** 0.5
                
                # IMPORTANT: For population data, do NOT resample.
                # Resampling population counts with bilinear interpolation creates artificial
                # population that doesn't exist. Population is a count/density measure that
                # should not be interpolated.
                # We use all_touched=True below to ensure small areas capture nearby pixels.
                needs_resampling = False
                
                if needs_resampling:
                    # FIRST: Clip to a coarse window around the shapefile to reduce data size
                    # Add generous padding (50 pixels) to ensure we don't lose data
                    padding_pixels = 50
                    minx, miny, maxx, maxy = shapefile_bounds
                    
                    # Convert shapefile bounds to pixel coordinates
                    inv_transform = ~src.transform
                    col_min, row_max = inv_transform * (minx, miny)
                    col_max, row_min = inv_transform * (maxx, maxy)
                    
                    # Add padding and clamp to raster bounds
                    col_min = max(0, int(col_min) - padding_pixels)
                    row_min = max(0, int(row_min) - padding_pixels)
                    col_max = min(src.width, int(col_max) + padding_pixels)
                    row_max = min(src.height, int(row_max) + padding_pixels)
                    
                    window_width = col_max - col_min
                    window_height = row_max - row_min
                    
                    # Read only the windowed region
                    window = rasterio.windows.Window(col_min, row_min, window_width, window_height)
                    windowed_data = src.read(band_index, window=window)
                    windowed_transform = src.window_transform(window)
                    
                    # NOW resample just this small window
                    scale_factor = 10
                    new_width = int(window_width * scale_factor)
                    new_height = int(window_height * scale_factor)
                    
                    # Calculate new transform for resampled window
                    windowed_bounds = rasterio.windows.bounds(window, src.transform)
                    # windowed_bounds is a tuple: (left, bottom, right, top)
                    new_transform = rasterio.transform.from_bounds(
                        windowed_bounds[0],  # left
                        windowed_bounds[1],  # bottom
                        windowed_bounds[2],  # right
                        windowed_bounds[3],  # top
                        new_width,
                        new_height,
                    )
                    
                    # Resample the windowed data
                    resampled_data = np.empty((new_height, new_width), dtype=src.dtypes[0])
                    reproject(
                        source=windowed_data,
                        destination=resampled_data,
                        src_transform=windowed_transform,
                        src_crs=src_crs,
                        dst_transform=new_transform,
                        dst_crs=src_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=src.nodata,
                        dst_nodata=nodata_value,
                    )
                    
                    # Buffer shapes by half pixel in resampled resolution
                    new_resolution = pixel_size / scale_factor
                    half_pixel = new_resolution * 0.5
                    try:
                        buffered = shapes_gdf.buffer(half_pixel)
                        buffered_union = buffered.unary_union
                        shapes = [buffered_union.__geo_interface__]
                    except Exception:
                        pass
                    
                    # Mask the resampled data
                    from rasterio.io import MemoryFile
                    with MemoryFile() as memfile:
                        with memfile.open(
                            driver='GTiff',
                            width=new_width,
                            height=new_height,
                            count=1,
                            dtype=resampled_data.dtype,
                            crs=src_crs,
                            transform=new_transform,
                            nodata=nodata_value,
                        ) as mem_src:
                            mem_src.write(resampled_data, 1)
                            
                            out_image, out_transform = rasterio.mask.mask(
                                mem_src,
                                shapes,
                                crop=True,
                                filled=True,
                                nodata=nodata_value,
                                all_touched=True,
                            )
                else:
                    # Normal flow: no resampling needed
                    # To avoid losing edge pixels due to pixel-center inclusion rules,
                    # we buffer by ~half a pixel in raster units.
                    half_pixel = 0.5 * pixel_size
                    try:
                        buffered = shapes_gdf.buffer(half_pixel)
                        buffered_union = buffered.unary_union
                        shapes = [buffered_union.__geo_interface__]
                    except Exception:
                        # Fall back to unbuffered shapes.
                        pass

                    nodata_value = src.nodata if src.nodata is not None else -9999
                    try:
                        test_read = src.read(band_index, window=rasterio.windows.Window(0, 0, 10, 10))
                    except Exception as read_err:
                        last_error = read_err
                        continue
                    
                    out_image, out_transform = rasterio.mask.mask(
                        src,
                        shapes,
                        crop=True,
                        indexes=band_index,
                        filled=True,
                        nodata=nodata_value,
                        all_touched=True,
                    )

                # mask() returns a 2D array when indexes is an int.
                if out_image.ndim == 2:
                    out_image = out_image[None, ...]

                # If the clip produces a single pixel, we likely opened the wrong subdataset.
                if out_image.shape[1] <= 1 or out_image.shape[2] <= 1:
                    last_error = ValueError(
                        f"Clipped output too small: {out_image.shape[1]}x{out_image.shape[2]} pixels"
                    )
                    continue
                if (out_image[0] == nodata_value).all():
                    last_error = ValueError(
                        "All pixels in clipped output are NoData (possible geometry/raster misalignment)"
                    )
                    continue

                out_meta = src.meta.copy()
                out_meta.update(
                    {
                        "driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "count": 1,
                        "nodata": nodata_value,
                    }
                )

                scenario_dir.mkdir(parents=True, exist_ok=True)
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(out_image[0], 1)

                return out_path
        except Exception as exc:  # pragma: no cover
            last_error = exc

    raise RuntimeError(
        "Unable to slice NetCDF population raster to the session geometry. "
        "No suitable 2D subdataset produced a non-trivial clipped raster."
        + (f" Last error: {last_error}" if last_error else "")
    )

    return out_path


def generate_baseline_csv_projection(
    *,
    session_dir: Path,
    schema: str,
    scenario_dir: Path,
) -> Path:
    """Placeholder projection for tabular schemas.

    Copies the session baseline CSV from `default/<schema>.csv` into the scenario
    directory.
    """

    schema_norm = schema.strip().lower()
    if schema_norm not in {"sanitation", "treatment", "population"}:
        raise ValueError("Unsupported schema for CSV projection.")

    baseline_path = session_dir / "default" / f"{schema_norm}.csv"
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
    shapefile_path: Path,
) -> None:
    """Update population column in human_emissions.csv with values from isoraster.tif.
    
    Args:
        human_emissions_path: Path to the human_emissions.csv file to update
        isoraster_path: Path to the isoraster.tif raster file
        shapefile_path: Path to the session geodata shapefile
    """
    
    # Load human_emissions.csv first to check what we're working with
    df = pd.read_csv(human_emissions_path)
    
    # Determine the GID column name in the CSV
    csv_gid_column = None
    for col in ["gid", "alpha3", "iso3"]:
        if col in df.columns:
            csv_gid_column = col
            break
    
    if csv_gid_column is None:
        raise ValueError("human_emissions.csv missing required identifier column (gid/alpha3/iso3)")
    
    # Update population column with calculated values
    if "population" not in df.columns:
        raise ValueError("human_emissions.csv missing 'population' column")
    
    # Load shapefile to check its structure
    gdf = gpd.read_file(shapefile_path)
    
    # Determine the correct GID column in the shapefile based on admin level in CSV
    # Count dots in the CSV gid column to determine admin level
    sample_gid = str(df[csv_gid_column].iloc[0])
    dot_count = sample_gid.count(".")
    shapefile_gid_column = f"GID_{dot_count}"
    
    # Verify the column exists in the shapefile
    if shapefile_gid_column not in gdf.columns:
        # Fall back to auto-detection if the expected column doesn't exist
        possible_columns = ["GID_4", "GID_3", "GID_2", "GID_1", "GID_0", "gid", "alpha3", "iso3"]
        shapefile_gid_column = None
        for col in possible_columns:
            if col in gdf.columns:
                shapefile_gid_column = col
                break
        if shapefile_gid_column is None:
            raise ValueError(f"Could not find appropriate GID column in shapefile. Available: {list(gdf.columns)}")
    
    # If there's only one feature in the shapefile, sum all raster values
    if len(gdf) == 1:
        with rasterio.open(isoraster_path) as src:
            raster_data = src.read(1)
            nodata = src.nodata
            
            # Filter out nodata values (handle floating point comparison)
            if nodata is not None:
                if abs(nodata) > 1e10:
                    # For very large nodata values, use threshold comparison
                    raster_data = raster_data[raster_data < 1e10]
                else:
                    # For normal nodata values, use close-enough comparison
                    raster_data = raster_data[~np.isclose(raster_data, nodata, rtol=1e-5)]
            
            # Also filter out any remaining extreme values or NaNs
            raster_data = raster_data[np.isfinite(raster_data)]
            
            # Round to nearest integer since population represents people
            total_pop = round(float(np.sum(raster_data)))
        
        # Assign this total to all rows in the CSV (since there's only one area)
        df["population"] = total_pop
    else:
        # Multiple features: need to match by identifier
        # Pass the correct shapefile column to the zonal calculation function
        zonal_populations = calculate_zonal_population(
            isoraster_path, 
            shapefile_path,
            gid_column=shapefile_gid_column
        )
        df["population"] = df[csv_gid_column].astype(str).map(zonal_populations)
        
        # Handle any missing mappings
        if df["population"].isna().any():
            df["population"] = df["population"].fillna(0)
    
    # Write updated CSV back
    df.to_csv(human_emissions_path, index=False)

