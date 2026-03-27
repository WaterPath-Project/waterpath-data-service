# Utilities

This folder contains diagnostic scripts and tests used during development and validation of the waterpath-data-service.

## Prerequisites

Make sure you have poetry installed and dependencies set up:

```powershell
cd C:\Users\user\source\waterpath-data-service
poetry install
```

## Available Tools

### 1. Inspect NetCDF Structure (`inspect_netcdf.py`)

View the internal structure of the population NetCDF files, including subdatasets and band→year mapping.

**Usage:**
```powershell
# Inspect SSP1 NetCDF
poetry run python utils/inspect_netcdf.py --ssp SSP1

# Inspect SSP2 NetCDF
poetry run python utils/inspect_netcdf.py --ssp SSP2
```

**Output:**
- File path and size
- CRS information
- List of subdatasets with dimensions
- Band index → Year mapping (showing band 1 = 2006, band 25 = 2030, etc.)

---

### 2. Verify Clipping (`verify_clip.py`)

Check that the clipped `isoraster.tif` matches the session shapefile boundaries.

**Usage:**
```powershell
# Basic verification (text output only)
poetry run python utils/verify_clip.py <session_id>

# Verify specific scenario
poetry run python utils/verify_clip.py test_7 --scenario SSP1_2030

# With visualization (opens a plot window)
poetry run python utils/verify_clip.py kampala_dataset --visualize
```

**Output:**
- Shapefile CRS and bounds
- Clipped raster CRS, bounds, and shape
- Data statistics (min/max/mean, valid pixel count)
- CRS alignment check (✓ or ⚠)
- Bounds comparison to verify clipping didn't cut off data
- Optional: Visual overlay of raster and shapefile boundary

---

### 3. Extract Year Sample (`extract_year_sample.py`)

Pull a small data sample from a specific year in the NetCDF to verify the correct band is being used.

**Usage:**
```powershell
# Extract sample from SSP1, year 2030
poetry run python utils/extract_year_sample.py --ssp SSP1 --year 2030

# Extract larger sample (10x10 pixels instead of 5x5)
poetry run python utils/extract_year_sample.py --ssp SSP2 --year 2050 --sample-size 10
```

**Output:**
- Band index calculation (e.g., year 2030 → band 25)
- Which subdataset is being read
- Sample data values from the center of the grid
- Statistics (min/max/mean)

---

### 4. Debug Raster (`debug_raster.py`)

Quick diagnostic for any single-band GeoTIFF — prints shape, CRS, nodata value,
min/max/mean, non-zero pixel count, and a sample of unique values.  Accepts the
raster path as a command-line argument.

**Usage:**
```powershell
# Pass the path as an argument
poetry run python utils/debug_raster.py waterpath_data_service/data/<session>/scenarios/SSP1_2025/isoraster.tif

# Falls back to a hard-coded Kampala isoraster if no argument is given
poetry run python utils/debug_raster.py
```

---

### 5. Analyze Raster Values (`analyze_raster_values.py`)

More detailed statistics for a raster, including a bucketed value distribution
(< 10k, 10k–50k, 50k–100k, 100k–500k, > 500k) and sample raw pixel values.
The path is hard-coded to a Kampala isoraster; edit the script or adapt it for
other datasets.

**Usage:**
```powershell
poetry run python utils/analyze_raster_values.py
```

---

### 6. Check Baseline Population (`check_baseline_pop.py`)

Reads the baseline `human_emissions.csv` for a session and prints the total
population, per-area mean, area count, and the first ten rows.  The path is
hard-coded to the Kampala dataset; edit for other sessions.

**Usage:**
```powershell
poetry run python utils/check_baseline_pop.py
```

---

### 7. Test Zonal Calculation (`test_zonal_calc.py`)

Manual smoke-test for `calculate_zonal_population` from `projections.py`.
Reads the Kampala geodata shapefile and isoraster, runs the zonal calculation,
and reports per-area results, the total population sum, and any areas with zero
population.

**Usage:**
```powershell
poetry run python utils/test_zonal_calc.py
```

---

### 8. Projection Tests (`test_projections.py`)

Comprehensive pytest test suite that validates the projection pipeline end-to-end
for both country-level and sub-national data.  Covers:

- `read_schema_field_names` — verifies that all static schemas parse correctly
- Treatment projections — column set matches schema, fraction values ≤ 1,
  quaternary derivation when absent
- Sanitation projections — column set, numeric integrity checks
- Livestock schema filtering — tabular CSV is constrained to schema-declared
  fields; meat/dairy columns inherit cattle values when absent from the future CSV
- Country-level isodata projections (Balkan test session, mocked HTTP)
- Sub-national projections (Dhaka test session)

**Usage:**
```powershell
# Run all projection tests
poetry run pytest utils/test_projections.py -v

# Run a single test class
poetry run pytest utils/test_projections.py::TestTreatmentProjection -v
```

---

## Example Workflow

### Verify a projection was generated correctly:

1. **Check the NetCDF structure:**
   ```powershell
   poetry run python utils/inspect_netcdf.py --ssp SSP1
   ```
   Confirm that band 25 corresponds to year 2030, etc.

2. **Verify the year extraction:**
   ```powershell
   poetry run python utils/extract_year_sample.py --ssp SSP1 --year 2030
   ```
   Check that the sample data looks reasonable (non-zero population values).

3. **Verify the clipping:**
   ```powershell
   poetry run python utils/verify_clip.py test_7 --scenario SSP1_2030 --visualize
   ```
   Confirm:
   - CRS matches between shapefile and raster ✓
   - Raster bounds are within shapefile bounds ✓
   - Visual overlay shows the clipped region aligns with shapefile
   - Valid pixel percentage is reasonable (not 100% nodata)

4. **Run debt-free projection tests:**
   ```powershell
   poetry run pytest utils/test_projections.py -v
   ```

---

## Troubleshooting

### "No isoraster.tif files found"
- Make sure you've called the `/api/projections/generate` endpoint with `schema=population`
- Check that the scenario folder was created under `data/<session_id>/scenarios/<SSP>_<year>/`

### "CRS mismatch"
- This is usually OK if the code reprojects automatically
- The verify script will show both CRS values
- Use `--visualize` to confirm the overlay looks correct

### Visualization doesn't work
- You need matplotlib with a GUI backend (TkAgg, Qt5Agg)
- Try running without `--visualize` for text-only output
- On headless servers, skip visualization or export to image file

---

## Integration with API

These tools work with the data generated by:

```
POST /api/projections/generate?session_id=<id>&schema=population&ssp=SSP1&year=2030
```

Which produces:
```
data/<session_id>/scenarios/SSP1_2030/
  ├── isoraster.tif          ← Clipped population raster
  └── human_emissions.csv    ← Merged baseline emissions
```
