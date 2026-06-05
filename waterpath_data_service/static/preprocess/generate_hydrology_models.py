#!/usr/bin/env python3
"""
Generate GloWPa-compatible hydrology TIF files from NC source files.

Handles two input types:

1. Tab-separated NC text files (monthly runoff / baseflow / discharge):
   static/data/hydrology/hydrology_original/
     monthly_runoff_{scenario}.nc.txt
     monthly_baseflow_{scenario}.nc.txt
     monthly_discharge_{scenario}.nc.txt
   → computes runoff, discharge, river_depth, river_restime

2. ISIMIP3b daily NetCDF4 binary files for surface solar radiation (rsds):
   static/data/hydrology/hydrology_original/rsds_daily/
     rsds_day_{MODEL}_{SSP}_r*_*_{YYYYMMDD}-{YYYYMMDD}.nc
   → computes ssrd (kJ/m²/day)

Implements the same algorithms as WPhydrology Julia scripts:
  - scriptWP2.jl    (master loop, 12-month averaging)
  - hydraulics.jl   (depth & residence time, Leopold & Maddock 1953)
  - prepRads2.jl    (daily rsds → monthly ssrd climatology, W/m² × 86.4)
  - constants.jl    (physical constants)

Output:
  static/data/hydrology/models/{scenario}/
    runoff/runoff_m{NN}.tif             [mm/day]
    discharge/discharge_m{NN}.tif       [m³/s]
    river_depth/river_depth_m{NN}.tif   [m]
    river_restime/river_restime_m{NN}.tif [days]
    ssrd/ssrd_m{NN}.tif                 [kJ/m²/day]  (when rsds NC available)

Output format:
  - Shape: 360 × 720  (global 0.5° grid)
  - Transform: origin (-180, 90), pixel 0.5°
  - CRS: EPSG:4326
  - NoData: -inf, dtype float32
"""

import calendar
import re
import sys
from math import cos, radians, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

# ==========================================
# CONSTANTS (from Julia constants.jl)
# ==========================================
DEPTH_A = 0.34
DEPTH_B = 0.341
VEL_A = 0.19
VEL_B = 0.266
EARTH_RADIUS = 6371003.0  # meters

# ==========================================
# GLOBAL GRID  (matches existing GFDL TIFs)
# ==========================================
GRID_ROWS = 360
GRID_COLS = 720
GRID_RES = 0.5          # degrees per pixel
GRID_LEFT = -180.0      # left edge of grid
GRID_TOP = 90.0         # top edge of grid

# Cell-centre coordinates for col/row 0
LON_CENTER_0 = GRID_LEFT + GRID_RES / 2   # = -179.75
LAT_CENTER_0 = GRID_TOP - GRID_RES / 2    # =  89.75

NODATA = -np.inf

# ==========================================
# FLOW-DIRECTION MAP
# VIC/TauDEM convention – values 1-8, clockwise from East:
#   1=E, 2=SE, 3=S, 4=SW, 5=W, 6=NW, 7=N, 8=NE
# In raster array coordinates: row increases downward (southward).
# ==========================================
D8_ROW_OFFSET = {
    1: 0,   # E
    2: 1,   # SE
    3: 1,   # S
    4: 1,   # SW
    5: 0,   # W
    6: -1,  # NW
    7: -1,  # N
    8: -1,  # NE
}
D8_COL_OFFSET = {
    1: 1,   # E
    2: 1,   # SE
    3: 0,   # S
    4: -1,  # SW
    5: -1,  # W
    6: -1,  # NW
    7: 0,   # N
    8: 1,   # NE
}


# ==========================================
# HELPERS
# ==========================================

def _lon_to_col(lon: float) -> int:
    """Map longitude (cell-centre) to 0-based column index."""
    return int(round((lon - LON_CENTER_0) / GRID_RES))


def _lat_to_row(lat: float) -> int:
    """Map latitude (cell-centre) to 0-based row index."""
    return int(round((LAT_CENTER_0 - lat) / GRID_RES))


def _write_tif(data: np.ndarray, output_path: Path) -> None:
    """Write a float32 numpy array to a GeoTIFF with the global grid transform."""
    transform = from_origin(GRID_LEFT, GRID_TOP, GRID_RES, GRID_RES)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.where(np.isfinite(data), data, NODATA).astype(np.float32)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=GRID_ROWS,
        width=GRID_COLS,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=NODATA,
    ) as dst:
        dst.write(out, 1)


# ==========================================
# LOAD NC TEXT FILE → MONTHLY GRIDS
# ==========================================

def load_nc_to_monthly_grids(filepath: Path) -> dict[int, np.ndarray]:
    """
    Parse a tab-separated .nc.txt file and return a dict:
      month (1–12) → float32 numpy array of shape (GRID_ROWS, GRID_COLS)

    The CSV format is:
      Latitude | Longitude | 2021-01 | 2021-02 | … | 2030-12
    Each row is a 0.5° cell-centre point; value columns are YYYY-MM dates.
    """
    print(f"    Loading {filepath.name} …", flush=True)
    df = pd.read_csv(filepath, sep="\t", na_values=["nan", "NaN", ""])

    lats = df["Latitude"].values
    lons = df["Longitude"].values

    # Grid indices for every CSV row (clamp to valid range)
    rows = np.clip(
        np.round((LAT_CENTER_0 - lats) / GRID_RES).astype(int), 0, GRID_ROWS - 1
    )
    cols = np.clip(
        np.round((lons - LON_CENTER_0) / GRID_RES).astype(int), 0, GRID_COLS - 1
    )

    # Column names that look like YYYY-MM
    date_cols = [
        c for c in df.columns if re.match(r"^\d{4}-\d{2}$", str(c).strip())
    ]

    monthly_grids: dict[int, np.ndarray] = {}

    for m in range(1, 13):
        # All columns that belong to this calendar month (across all years)
        month_cols = [c for c in date_cols if int(c.split("-")[1]) == m]
        if not month_cols:
            monthly_grids[m] = np.full((GRID_ROWS, GRID_COLS), np.nan, dtype=np.float32)
            continue

        # Average across years; shape (n_points,)
        month_data = df[month_cols].to_numpy(dtype=np.float64)
        month_avg = np.nanmean(month_data, axis=1)

        # Build spatial grid
        grid = np.full((GRID_ROWS, GRID_COLS), np.nan, dtype=np.float32)
        valid = np.isfinite(month_avg) & (month_avg >= 0)
        grid[rows[valid], cols[valid]] = month_avg[valid].astype(np.float32)
        monthly_grids[m] = grid

    del df  # free the large dataframe
    print(f"    → extracted {GRID_ROWS}×{GRID_COLS} grids for 12 months.", flush=True)
    return monthly_grids


# ==========================================
# LOAD FLOW DIRECTION
# ==========================================

def load_flowdir(flowdir_path: Path) -> np.ndarray:
    """Return the flow-direction raster as a float32 array (ROWS, COLS)."""
    with rasterio.open(flowdir_path) as src:
        fd = src.read(1)
        # Resize to GRID_ROWS × GRID_COLS if necessary (the static routing
        # grid is 280 rows, covering 83.75 N → −56.25 S).  We embed it into
        # the full 360-row global grid so indices align with the data grids.
        if fd.shape != (GRID_ROWS, GRID_COLS):
            transform = src.transform
            # Find the row offset of the first row in the global grid
            top_lat = transform.f             # top edge of first pixel
            row_offset = int(round((GRID_TOP - top_lat) / GRID_RES))
            full_fd = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
            r_end = row_offset + fd.shape[0]
            c_end = int(round((transform.c - GRID_LEFT) / GRID_RES)) + fd.shape[1]
            c_start = int(round((transform.c - GRID_LEFT) / GRID_RES))
            full_fd[row_offset:r_end, c_start:c_end] = fd.astype(np.float32)
            return full_fd
    return fd.astype(np.float32)


# ==========================================
# HYDRAULICS (from Julia hydraulics.jl)
# ==========================================

def compute_runoff_rate(
    runoff_mm: np.ndarray, baseflow_mm: np.ndarray, days: int
) -> np.ndarray:
    """
    Runoff rate [mm/day] = (runoff + baseflow) / days_in_month.
    Returns NaN where either input is NaN.
    """
    out = np.full((GRID_ROWS, GRID_COLS), np.nan, dtype=np.float32)
    valid = np.isfinite(runoff_mm) & np.isfinite(baseflow_mm)
    total = runoff_mm[valid] + baseflow_mm[valid]
    total = np.maximum(total, 0.0)
    out[valid] = (total / days).astype(np.float32)
    return out


def compute_depth(discharge_m3s: np.ndarray) -> np.ndarray:
    """
    River depth [m] = DEPTH_A * Q ^ DEPTH_B   (Leopold & Maddock 1953)
    Only where discharge > 0.
    """
    out = np.full((GRID_ROWS, GRID_COLS), np.nan, dtype=np.float32)
    valid = np.isfinite(discharge_m3s) & (discharge_m3s > 0)
    out[valid] = (DEPTH_A * (discharge_m3s[valid] ** DEPTH_B)).astype(np.float32)
    return out


def compute_restime(
    discharge_m3s: np.ndarray, flowdir: np.ndarray
) -> np.ndarray:
    """
    Residence time [days] = haversine_dist / (velocity * 86400)

    velocity [m/s] = VEL_A * Q ^ VEL_B   (Leopold & Maddock 1953)
    Haversine distance is computed between the cell centre and the
    centre of its downstream neighbour.
    """
    restime = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

    # Pre-compute lat/lon radians for every cell
    lats_rad = np.radians(
        LAT_CENTER_0 - np.arange(GRID_ROWS) * GRID_RES
    )  # shape (GRID_ROWS,)
    lons_rad = np.radians(
        LON_CENTER_0 + np.arange(GRID_COLS) * GRID_RES
    )  # shape (GRID_COLS,)

    valid_q = np.isfinite(discharge_m3s) & (discharge_m3s > 0)

    for fd_val in range(1, 9):
        dr = D8_ROW_OFFSET[fd_val]
        dc = D8_COL_OFFSET[fd_val]
        mask = valid_q & (np.round(flowdir).astype(int) == fd_val)
        if not np.any(mask):
            continue

        r_arr, c_arr = np.where(mask)
        next_r = np.clip(r_arr + dr, 0, GRID_ROWS - 1)
        next_c = (c_arr + dc) % GRID_COLS

        lat1 = lats_rad[r_arr]
        lat2 = lats_rad[next_r]
        lon1 = lons_rad[c_arr]
        lon2 = lons_rad[next_c]

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * (dlon / 2) ** 2
        a = np.clip(a, 0.0, 1.0)
        dist = 2.0 * EARTH_RADIUS * np.sqrt(a)  # metres

        vel = VEL_A * (discharge_m3s[r_arr, c_arr] ** VEL_B)
        vel = np.maximum(vel, 0.01)  # m/s floor

        restime[r_arr, c_arr] = (dist / (vel * 86400.0)).astype(np.float32)

    # Mark cells with no valid flow direction as NaN
    no_fd = ~(np.isfinite(flowdir) & (flowdir >= 1) & (flowdir <= 8))
    restime[valid_q & no_fd] = np.nan

    return restime


# ==========================================
# SCENARIO PROCESSING
# ==========================================

def process_scenario(
    scenario_name: str,
    data_dir: Path,
    out_dir: Path,
    flowdir_path: Path,
    overwrite: bool = False,
) -> None:
    """
    Process one scenario (e.g. GFDL-ESM4_ssp126_2021_2030):
      1. Load monthly grids from runoff / baseflow / discharge NC text files.
      2. Compute hydraulics for each of 12 months.
      3. Write GeoTIFFs to out_dir / scenario_name / {var} / {var}_m{NN}.tif.
    """
    out_scenario = out_dir / scenario_name

    # Check if already complete
    if not overwrite and out_scenario.exists():
        complete = all(
            (out_scenario / var / f"{var}_m{m:02d}.tif").exists()
            for var in ["runoff", "discharge", "river_depth", "river_restime"]
            for m in range(1, 13)
        )
        if complete:
            print(f"  ✓ {scenario_name} already complete – skipping.")
            return

    r_file = data_dir / f"monthly_runoff_{scenario_name}.nc.txt"
    b_file = data_dir / f"monthly_baseflow_{scenario_name}.nc.txt"
    q_file = data_dir / f"monthly_discharge_{scenario_name}.nc.txt"

    for f in [r_file, b_file, q_file]:
        if not f.exists():
            print(f"  ✗ Missing {f.name} – skipping {scenario_name}.")
            return

    print(f"\n=== {scenario_name} ===", flush=True)

    runoff_grids = load_nc_to_monthly_grids(r_file)
    baseflow_grids = load_nc_to_monthly_grids(b_file)
    discharge_grids = load_nc_to_monthly_grids(q_file)

    print("  Loading flow-direction grid …", flush=True)
    flowdir = load_flowdir(flowdir_path)

    for m in range(1, 13):
        month_str = f"m{m:02d}"
        days = calendar.monthrange(2015, m)[1]  # days in this month (non-leap ref)

        ro = runoff_grids[m]        # mm/month
        bf = baseflow_grids[m]      # mm/month
        q = discharge_grids[m]      # m³/s

        ro_rate = compute_runoff_rate(ro, bf, days)
        depth = compute_depth(q)
        restime = compute_restime(q, flowdir)

        _write_tif(ro_rate,  out_scenario / "runoff"       / f"runoff_{month_str}.tif")
        _write_tif(q,        out_scenario / "discharge"    / f"discharge_{month_str}.tif")
        _write_tif(depth,    out_scenario / "river_depth"  / f"river_depth_{month_str}.tif")
        _write_tif(restime,  out_scenario / "river_restime"/ f"river_restime_{month_str}.tif")

        print(f"  month {m:02d} done.", flush=True)

    print(f"  ✅ Saved to {out_scenario}", flush=True)


# ==========================================
# MAIN
# ==========================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate GloWPa hydrology TIFs from NC text files."
    )
    parser.add_argument(
        "--static-dir",
        default=None,
        help=(
            "Absolute path to the static/ directory. "
            "Defaults to two levels above this script "
            "(i.e. …/static/ when the script is at …/static/preprocess/)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate scenarios that already have all 48 output TIFs.",
    )
    args = parser.parse_args()

    if args.static_dir:
        static_dir = Path(args.static_dir).resolve()
    else:
        # Script lives at:  …/static/preprocess/generate_hydrology_models.py
        # Static data root: …/static/
        static_dir = Path(__file__).resolve().parent.parent

    data_dir = static_dir / "data" / "hydrology" / "hydrology_original"
    models_dir = static_dir / "data" / "hydrology" / "models"
    flowdir_path = static_dir / "data" / "hydrology" / "routing" / "flowdir.tif"

    overwrite = args.overwrite

    print(f"Data dir  : {data_dir}")
    print(f"Models dir: {models_dir}")
    print(f"Flowdir   : {flowdir_path}")

    if not data_dir.exists():
        print(f"ERROR: hydrology_original not found at {data_dir}")
        sys.exit(1)
    if not flowdir_path.exists():
        print(f"ERROR: flowdir.tif not found at {flowdir_path}")
        sys.exit(1)

    runoff_files = sorted(data_dir.glob("monthly_runoff_*.nc.txt"))
    if not runoff_files:
        print("ERROR: no monthly_runoff_*.nc.txt files found.")
        sys.exit(1)

    print(f"\nFound {len(runoff_files)} scenarios to process.\n")

    for r_file in runoff_files:
        scenario_name = r_file.name.replace("monthly_runoff_", "").replace(".nc.txt", "")
        process_scenario(scenario_name, data_dir, models_dir, flowdir_path, overwrite=overwrite)

    print("\n🎉 All hydrology scenarios processed!")
    # ---------------------------------------------------------------
    # SSRD: process daily ISIMIP3b rsds NetCDF4 files if present
    # ---------------------------------------------------------------
    rsds_dir = data_dir / "rsds_daily"
    if rsds_dir.is_dir():
        rsds_files = sorted(rsds_dir.glob("*.nc"))
        if rsds_files:
            print(f"\nFound {len(rsds_files)} daily rsds NC file(s) — generating ssrd TIFs.\n")
            for nc_file in rsds_files:
                scenario_name = _parse_rsds_scenario(nc_file.name)
                if scenario_name is None:
                    print(f"  ⚠ Could not parse scenario from '{nc_file.name}'; skipping.", flush=True)
                    continue
                out_ssrd_dir = models_dir / scenario_name / "ssrd"
                if not overwrite and out_ssrd_dir.is_dir() and len(list(out_ssrd_dir.glob("ssrd_m??.tif"))) == 12:
                    print(f"  Skipping {scenario_name} ssrd (already complete).", flush=True)
                    continue
                print(f"=== SSRD: {scenario_name} ===", flush=True)
                try:
                    monthly_grids = load_daily_ssrd_nc_to_monthly_grids(nc_file)
                    out_ssrd_dir.mkdir(parents=True, exist_ok=True)
                    for m, grid in monthly_grids.items():
                        month_str = f"m{m:02d}"
                        _write_tif(grid, out_ssrd_dir / f"ssrd_{month_str}.tif")
                    print(f"  ✅ Saved to {out_ssrd_dir}", flush=True)
                except Exception as exc:
                    print(f"  ❌ Failed: {exc}", flush=True)
            print("\n🎉 SSRD processing complete!")
        else:
            print(f"\nNo *.nc files found in {rsds_dir} — skipping ssrd.")
    else:
        print(f"\nNo rsds_daily/ directory found at {rsds_dir} — skipping ssrd.")
        print("  To generate ssrd: place ISIMIP3b daily rsds NetCDF4 files in:")
        print(f"  {rsds_dir}")
        print("  Expected naming: rsds_day_{{MODEL}}_{{SSP}}_r*_*_{{YYYYMMDD}}-{{YYYYMMDD}}.nc")


# ==========================================
# SSRD HELPERS
# ==========================================

# Pattern for ISIMIP3b bias-adjusted daily rsds filenames as distributed by
# the ISIMIP data portal:
#   gfdl-esm4_r1i1p1f1_w5e5_ssp126_rsds_global_daily_2021_2030.nc
#   ukesm1-0-ll_r1i1p1f2_w5e5_ssp585_rsds_global_daily_2091_2100.nc
_RSDS_FILENAME_RE = re.compile(
    r"^(?P<model>[a-z0-9][a-z0-9\-]+)"    # lowercase model name (e.g. gfdl-esm4)
    r"_r\d+i\d+p\d+f\d+"                   # realization index (r1i1p1f1 etc.)
    r"_\w+"                                 # bias-correction tag (w5e5 etc.)
    r"_(?P<ssp>ssp\d+)"                     # ssp126 / ssp585
    r"_rsds_global_daily"                   # fixed variable/domain/temporal fields
    r"_(?P<start>\d{4})_(?P<end>\d{4})\.nc$"
)


def _parse_rsds_scenario(filename: str) -> str | None:
    """Extract scenario string from an ISIMIP3b daily rsds filename.

    Returns e.g. ``'GFDL-ESM4_ssp126_2021_2030'``, or ``None`` if the
    filename does not match the expected pattern.

    Model names in ISIMIP filenames are lowercase; they are uppercased here
    to match the canonical directory names (e.g. ``GFDL-ESM4``).
    """
    m = _RSDS_FILENAME_RE.match(filename)
    if not m:
        return None
    model_upper = m["model"].upper()
    return f"{model_upper}_{m['ssp']}_{m['start']}_{m['end']}"


def load_daily_ssrd_nc_to_monthly_grids(filepath: Path) -> dict[int, np.ndarray]:
    """Load an ISIMIP3b daily rsds NetCDF4 file and return a 12-month climatology.

    Algorithm (replicates WPhydrology/src/prepRads2.jl):
      1. Group all daily timesteps by calendar month across the full decade.
      2. Average each group to get 12 mean monthly values per grid cell.
      3. Convert W/m² → kJ/m²/day  (multiply by 86.4).

    Parameters
    ----------
    filepath:
        Path to a daily ISIMIP3b NetCDF4 file containing variable ``rsds``.

    Returns
    -------
    dict: month (1–12) → float32 array of shape (GRID_ROWS, GRID_COLS),
          oriented North-up (row 0 = 89.75°N).
    """
    try:
        import netCDF4 as nc4  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "netCDF4 is required for daily rsds processing. "
            "Install it with: pip install netCDF4"
        ) from exc

    print(f"    Loading {filepath.name} …", flush=True)

    with nc4.Dataset(filepath, "r") as ds:
        # --- Time axis ---
        time_var = ds.variables["time"]
        time_vals = nc4.chartostring(time_var[:]) if time_var.dtype == "S1" else time_var[:]
        time_units = time_var.units
        calendar = getattr(time_var, "calendar", "standard")
        dates = nc4.num2date(time_vals, units=time_units, calendar=calendar)
        months = np.array([int(d.month) for d in dates], dtype=np.int16)

        # --- rsds variable ---
        rsds_var_name = next(
            (v for v in ("rsds", "RSDS", "ssrd", "SSRD") if v in ds.variables),
            None,
        )
        if rsds_var_name is None:
            raise KeyError(
                f"Variable 'rsds' not found in {filepath}. "
                f"Available: {list(ds.variables)}"
            )
        rsds_var = ds.variables[rsds_var_name]

        # --- Spatial axes ---
        lat_name = next((v for v in ("lat", "latitude", "y") if v in ds.variables), "lat")
        lon_name = next((v for v in ("lon", "longitude", "x") if v in ds.variables), "lon")
        lat = ds.variables[lat_name][:].data  # (nlat,)  may be S→N or N→S
        lon = ds.variables[lon_name][:].data  # (nlon,)

        # Whether lat is ascending (S→N — needs flip to match our N→S convention)
        lat_ascending = bool(lat[0] < lat[-1])

        monthly_grids: dict[int, np.ndarray] = {}

        for m in range(1, 13):
            idx = np.where(months == m)[0]
            if idx.size == 0:
                monthly_grids[m] = np.full((GRID_ROWS, GRID_COLS), NODATA, dtype=np.float32)
                continue

            # Read only the required time slices.
            # netCDF4 masked arrays: use .data with the mask to avoid slow Python loops.
            chunk = rsds_var[idx, :, :]           # (n_days, nlat, nlon)
            if hasattr(chunk, "data"):             # MaskedArray
                mask = np.ma.getmaskarray(chunk)
                data = chunk.data.astype(np.float64)
                data[mask] = np.nan
            else:
                data = chunk.astype(np.float64)

            # Handle scalar _FillValue / missing_value
            for attr in ("_FillValue", "missing_value"):
                fill = getattr(rsds_var, attr, None)
                if fill is not None:
                    data[data == float(fill)] = np.nan

            # Monthly climatology mean
            with np.errstate(all="ignore"):
                grid = np.nanmean(data, axis=0).astype(np.float32)   # (nlat, nlon)

            # W/m² → kJ/m²/day
            grid = np.where(np.isfinite(grid), grid * 86.4, np.nan).astype(np.float32)

            # Flip to North-up if needed
            if lat_ascending:
                grid = np.flipud(grid)

            # Embed into full GRID_ROWS × GRID_COLS when the NC is not global
            if grid.shape != (GRID_ROWS, GRID_COLS):
                full = np.full((GRID_ROWS, GRID_COLS), np.nan, dtype=np.float32)
                # North-edge latitude after potential flip
                north_lat = lat[-1] if lat_ascending else lat[0]
                row_start = int(round((GRID_TOP - (north_lat + GRID_RES / 2)) / GRID_RES))
                west_lon = lon[0]
                col_start = int(round((west_lon - GRID_LEFT) / GRID_RES))
                row_start = max(0, row_start)
                col_start = max(0, col_start)
                row_end = min(GRID_ROWS, row_start + grid.shape[0])
                col_end = min(GRID_COLS, col_start + grid.shape[1])
                full[row_start:row_end, col_start:col_end] = grid[
                    : row_end - row_start, : col_end - col_start
                ]
                grid = full

            # Replace remaining NaN with NODATA sentinel
            monthly_grids[m] = np.where(np.isfinite(grid), grid, NODATA).astype(np.float32)

    print(f"    → extracted {GRID_ROWS}×{GRID_COLS} SSRD grids for 12 months.", flush=True)
    return monthly_grids

if __name__ == "__main__":
    main()
