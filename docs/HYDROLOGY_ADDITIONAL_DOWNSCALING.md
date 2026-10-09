# Hydrology Inputs and Experimental Downscaling

## Default Behavior

Generated hydrology is clipped on its native 0.5-degree EPSG:4326 grid. It is not
resampled onto the emissions isoraster unless experimental downscaling is explicitly
enabled. The river-temperature fallback is aligned to the clipped native hydrology
grid. The isoraster, not hydrology, dictates the final model grid. Livestock and
continuous inputs are prepared against it without changing its resolution.
Hydrology is retained as a separate, internally consistent grid. Native D8 and
hydraulic rasters are never blindly resampled to the emissions isoraster.
Downloads allow mixed grids without resampling the stored rasters. The downstream
tool or user must provide any two-grid coupling and validate model compatibility;
download success is not a readiness check for GloWPa.
Administrative clipping also does not reconstruct missing upstream pathogen loads.

## Custom Hydrology Upload

Use `POST /api/data/input/upload` with multipart `file`, `session_id` and
`file_id=hydrology`. Only a `.zip` with valid ZIP contents is accepted.
Optional paired `ssp` and `year` target an existing `scenarios/{SSP}_{year}`;
otherwise the target is `baseline`. The session and study-area geography must exist.

The archive must have exactly this structure (each monthly range means 12 files):

```text
hydrology/
   runoff/runoff_m01.tif ... runoff_m12.tif
   discharge/discharge_m01.tif ... discharge_m12.tif
   river_depth/river_depth_m01.tif ... river_depth_m12.tif
   river_restime/river_restime_m01.tif ... river_restime_m12.tif
   ssrd/ssrd_m01.tif ... ssrd_m12.tif
   river_temperature/river_temperature_m01.tif ... river_temperature_m12.tif
   routing/flowdir.tif
   routing/flowacc.tif
   doc.tif
```

All 75 rasters are required; no static-data fallback or merging is performed.
The files must be readable single-band GeoTIFFs, north-up EPSG:4326 with square
pixels, sharing exactly the same transform and dimensions. Their extent must
cover the study area. Any internally consistent resolution is accepted, without
resampling. Variable-specific nodata masks are allowed; every raster needs valid
data. Valid-data coverage and scientific calibration remain the user's responsibility.

Required units:

| Variable | Units |
|---|---|
| runoff | mm/day |
| discharge | m3/s |
| river_depth | m |
| river_restime | days |
| ssrd | kJ/m2/day |
| river_temperature | degC |
| doc | mg/L |
| flowacc | upstream_cell_count |
| flowdir | ESRI_D8 |

D8 codes are 0, 1, 2, 4, 8, 16, 32, 64 and 128, with 0 denoting an outlet/sink.
Cycles are rejected. Accumulation must be nonnegative integer-valued counts;
it is not forced to equal locally recomputed counts because upstream contributing
cells may lie outside the supplied extent. Negative values are rejected except
for temperature. Unmasked infinities/NaNs are rejected.

Optional `hydrology/metadata.json` accepts only `source`, `period` and `notes`
strings plus an optional `units` object matching every entry in the table exactly.
Other files, including uploaded `source.json`, are rejected. The service writes
its own `source.json` with the archive digest, grid, uploader declarations and
warnings. Declarations are not evidence of scientific validation.

Uploads stage the destination package with the new hydrology and prepare its
emissions inputs against the unchanged isoraster, validate the hydrology grid and
routing, then replace the target. Hydrology remains unchanged on its supplied grid.
Alignment and validation failures leave the target untouched; failed promotion
restores the old directory. Conflicting session writes receive 409. Downloads are invalidated
after a successful replacement, and summaries read current provenance and rasters.
Direct scenario uploads take precedence over custom baseline data. Otherwise,
custom baseline data is copied unchanged when scenario hydrology is requested.
Uploading baseline hydrology also refreshes existing scenarios that inherit it and
prepares their emissions rasters against their existing isoraster. Direct scenario
uploads may use a different hydrology grid from the scenario isoraster.
Supplying climate-model or
experimental controls for custom hydrology returns 409.

Limits are configurable via the usual `WATERPATH_DATA_SERVICE_` settings prefix:

| Setting | Default |
|---|---|
| HYDROLOGY_UPLOAD_MAX_BYTES | 536870912 (512 MiB) |
| HYDROLOGY_UPLOAD_EXPANDED_BYTES | 4294967296 (4 GiB, also total decoded raster bytes) |
| HYDROLOGY_UPLOAD_MAX_PIXELS | 2000000 per raster |

At most 100 ZIP entries are accepted; metadata is limited to 64 KiB. ZIP members
must use stored or deflate compression. Unsafe paths, duplicates, symlinks,
encrypted members and unexpected files are rejected. Limits apply to processing
after multipart ingestion; configure an HTTP proxy/body limit as well for untrusted
internet uploads. A process crash may leave `.input-write-lock` in the session;
remove it only after confirming no input operation is still running.

Error responses: 404 unknown session/target, 415 wrong upload type, 422 invalid
parameters/archive/raster data, 413 resource limits, 409 conflicting operations or
source options. Validation errors identify the offending member where possible.

## Experimental Status

**HIGHLY EXPERIMENTAL: not validated for production modelling or flow consistency.**
This mode is not a calibrated replacement for the native hydrology dataset.

## Purpose

The ISIMIP3b hydrology inputs used by WaterPath have a native resolution of
0.5 degrees (about 55 km at the equator). Nearest-neighbour reprojection copies
each source value into many model cells, producing visible rectangular blocks
in discharge and therefore in concentration.

The optional method below explores finer-grid processing without creating new
hydrological observations. Partial smoothing alone does not repair discharge or
routing consistency.

## Method

1. Convert monthly coarse-cell runoff depth to runoff volume using the source
   cell area.
2. Interpolate runoff depth onto the target grid with bilinear interpolation.
   Interpolate only valid land cells; do not interpolate across nodata or ocean
   boundaries.
3. For every original 0.5-degree cell, multiply its target-cell runoff values
   by one correction factor so their area-weighted volume equals the original
   source-cell volume exactly:

   $$
   f_i = \frac{R_i A_i}{\sum_{j \in i} r_j a_j}, \qquad r'_j = f_i r_j
   $$

   Here $R_i$ is the original runoff depth and $A_i$ is the represented target
   area within the source cell, while $r_j$ and
   $a_j$ are the interpolated depth and area of target cell $j$ within source
   cell $i$.
4. Route the corrected target-grid runoff over a finer, hydrologically
   conditioned D8 network. MERIT Hydro or HydroSHEDS are appropriate sources.
5. Derive monthly discharge from the accumulated runoff volume. Check outlet
   discharge against the sum of runoff generated upstream.
6. Use this discharge together with matching target-grid flow direction and
   flow accumulation inputs in GloWPa.

Apply the procedure independently to each month. Preserve the original units,
calendar convention, CRS, extent, nodata mask, and monthly timestamps.

## API Controls

Both `/input/generate` and `/projections/generate` expose
`hydrology_downscaling` (default `false`). Custom datasets cannot be combined
with experimental controls.

- Leaving downscaling disabled retains the native 0.5-degree grid.
- `/input/generate` now exposes `hydro_raster` for a future generic
   `hydro_raster.tif` contract. Uploading it currently returns 422; it is not
   interpreted as D8. The old baseline `hydrology_flow_direction_tif` parameter
   is no longer part of this endpoint's contract.
- `/projections/generate` retains `hydrology_flow_direction_tif`, accepting a finer-than-0.5-degree
   HydroSHEDS/ESRI D8 GeoTIFF using codes `0, 1, 2, 4, 8, 16, 32, 64, 128`.
   It should already match the case-study grid. Raw higher-resolution direction
   codes must first be aggregated with the matching HydroSHEDS flow-accumulation
   layer by tracing the dominant outlet path from each target cell; nearest-
   neighbour resampling of direction codes can create cycles. An uploaded raster
   is retained with the case-study baseline and reused for later projections.

When downscaling is enabled without a fine flow-direction raster, the service
uses a limited fallback: runoff is bilinearly interpolated and corrected to
preserve each represented source cell's area-weighted volume, and continuous
environmental fields are smoothed. Discharge and routing retain their coarse
source information because they cannot be reconstructed defensibly without a
fine drainage network.

When a valid fine flow-direction raster is supplied, the service derives flow
accumulation in upstream-cell counts, routes corrected runoff to derive monthly
discharge, and recalculates river depth and residence time on the target grid.
The generated `assumptions.csv` records which mode was used.

## Other Hydrology Variables

Bilinear interpolation is generally reasonable for continuous fields such as
river temperature, solar radiation, dissolved organic carbon, and cautiously
for river depth or residence time. Mask and bound the results to physically
valid ranges.

Do not bilinearly interpolate flow direction, flow accumulation, categorical
river masks, pathogen loads, or concentration outputs. Flow direction and
accumulation must be derived from the target-grid drainage network. Discharge
should be generated by routing runoff rather than smoothed independently.

## Validation

- Coarse-cell runoff volume before and after downscaling agrees within numeric
  tolerance for every month.
- Basin outlet discharge approximately closes the upstream monthly water
  balance after any documented losses or abstractions.
- Flow directions remain downstream and do not cross catchment divides.
- Discharge is non-negative and generally increases downstream.
- Monthly and annual domain totals match the original forcing.
- Maps disclose the native forcing resolution and that spatial detail below
  0.5 degrees is downscaled, not observed.

This approach improves spatial continuity, but it cannot recover sub-grid
hydrological processes absent from the source data. A calibrated fine-resolution
hydrological model remains the preferred scientific solution.