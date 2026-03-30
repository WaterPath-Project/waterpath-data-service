# Raster Resolution Decision and Resampling Methodology

## Overview

`prepare_spatial_inputs` (in `waterpath_data_service/services/prepare_spatial.py`)
produces three output GeoTIFFs for every scenario:

| Output file      | Content                                        |
|------------------|------------------------------------------------|
| `isoraster.tif`  | Integer zone-index raster (one value per polygon) |
| `pop_urban.tif`  | Gridded urban population counts                |
| `pop_rural.tif`  | Gridded rural population counts                |

The spatial resolution of these outputs—the size of each pixel in decimal
degrees—must be chosen carefully.  Too coarse, and small dense areas (e.g.
cities) are averaged away to near-zero population.  Too fine, and the grid
invents spatial detail that does not exist in the source data.

---

## Auto-Resolution Algorithm

When the caller does not supply an explicit `res` parameter, the resolution is
chosen automatically in three steps.

### Step 1 — Diagonal-based target

A target pixel size is derived so that ~100 pixels span the bounding-box
diagonal of the study area:

```
diagonal = sqrt(extent_lon² + extent_lat²)   [decimal degrees]
target   = diagonal / 100
```

This keeps output rasters at a manageable size regardless of study-area scale:

| Study area example         | Typical diagonal | `target` |
|----------------------------|-----------------|---------|
| City district (Dhaka)      | ~0.4°           | ~0.004° |
| Sub-national region        | ~3°             | ~0.03°  |
| Single country (Uganda)    | ~10°            | ~0.1°   |
| Large country (Brazil)     | ~42°            | ~0.42°  |
| Global                     | ~402°           | clamped |

### Step 2 — Floor at source TIF native resolution

The raw target is **floored at the native pixel size of the source population
raster** (read directly from the TIF header via `rasterio`):

```
raw = max(src_native_res, min(0.5°, target))
```

This is the critical guard against **one cause** of the Dhaka zero-population
problem.  The bundled WorldPop / GHS-POP 1 km TIFs have a native resolution of
approximately **0.00833°** (~926 m at the equator).  Generating a 0.004° output
from a 0.00833° source would require inventing sub-pixel detail, yielding
near-zero population counts in small polygons because each tiny destination
pixel samples only a fraction of one source pixel.  By flooring at
`src_native_res`, the minimum output resolution is always at least as coarse
as the source.

### Step 3 — Snap to a standard value

`raw` is snapped to the nearest entry in a fixed list of acceptable resolutions:

```
[0.001°, 0.002°, 0.005°, 0.01°, 0.025°, 0.05°, 0.1°, 0.25°, 0.5°]
```

Snapping ensures output pixels align on clean decimal boundaries, which avoids
sub-pixel misalignment when multiple rasters are overlaid or exported.

### Resulting resolution by study-area scale

| Study area               | Source TIF   | `target` | `raw`   | Final (snapped) |
|--------------------------|-------------|---------|--------|----------------|
| City (Dhaka, ~0.4° diag) | 1 km/0.00833° | 0.004° | **0.00833°** (floored) | **0.01°** ≈ 1 km |
| District (~3° diag)      | 1 km/0.00833° | 0.030° | 0.030° | **0.025°** ≈ 2.8 km |
| Country (~10° diag)      | 1 km/0.00833° | 0.100° | 0.100° | **0.1°** ≈ 11 km |
| Large country (~40° diag)| 1 km/0.00833° | 0.400° | 0.400° | **0.25°** ≈ 28 km |
| Global                   | 1 km/0.00833° | >0.5°  | 0.500° (clamped) | **0.5°** ≈ 55 km |

To override auto-selection for a specific run, pass an explicit `res` value
(in degrees) to `prepare_spatial_inputs`.

---

## Resampling Methodology

Population cannot be resampled like elevation or temperature because it is
an **extensive quantity** (counts, not densities): doubling the pixel area
should double the value.  A direct bilinear or average resample of raw counts
would therefore misrepresent population when pixel sizes change.  The
following count-safe pipeline is used instead.

### Pipeline

```
Source raster (counts / source cell)
        │
        ▼  ① Convert to density
src_density = src_counts / src_cell_area_km²

        │
        ▼  ② Spatially resample density using Resampling.average
dst_density  (average of source densities within each destination pixel)

        │
        ▼  ③ Convert back to counts
dst_counts = dst_density × dst_cell_area_km²
```

### Why `Resampling.average` and not bilinear?

`Resampling.average` computes the mean of all *valid* (non-NaN) source pixels
that overlap each destination pixel.  `Resampling.bilinear` uses a weighted
average of the four nearest source pixel centres and propagates `NaN` from any
one of those four neighbours to the destination pixel.  Near coastlines and
polygon boundaries—where source rasters often have nodata or zero-padded
edges—bilinear resampling would blank out a ring of otherwise valid destination
pixels.  `average` avoids this edge effect.

### Cell area computation

Cell areas are calculated on the WGS-84 ellipsoid (not a sphere), accounting
for the cosine-latitude dependence of east–west cell widths:

```
cell_area_km² = cell_width_km × cell_height_km

where:
  cell_width_km  = N(φ) × cos(φ) × Δλ_rad
  cell_height_km = M(φ)           × Δφ_rad

  N(φ) = prime-vertical radius of curvature   (varies with latitude)
  M(φ) = meridional radius of curvature       (varies with latitude)
```

This mirrors `terra::cellSize(rast, unit="km")` from the original R workflow.

### Coarsening (aggregation, upsample factor > 1)

When the destination is coarser than the source (typical country/global runs):

- Each 55 km destination pixel covers ~3600 source 1 km pixels.
- `Resampling.average` averages their density values, then multiplies by the
  destination cell area to recover total counts.
- Population is conserved: `Σ(dst_counts) ≈ Σ(src_counts)` within the domain.

### Refinement is disabled by the floor constraint

Because `raw = max(src_native_res, ...)`, the destination pixel is always **≥**
the source pixel.  The pipeline is therefore never run in pure upsampling
(fine→coarser) mode.  Using `Resampling.average` for downsampling (coarse→
fine) would spread one source pixel's value uniformly across many destination
pixels, producing an artificially smooth and sometimes near-zero output.

---

## Polygon Rasterization (isoraster.tif)

### Burn order: largest polygons first

`rasterio.features.rasterize` renders shapes in list order and later shapes
**overwrite** earlier ones at contested pixels.  When all polygons are adjacent
(e.g. the ~54 Dhaka thanas), a small polygon that appears early in the
shapefile feature list can have every one of its raster cells claimed by a
later, larger neighbour.  The result is a zone with zero pixels → `nansum`
returns 0 → projected population is written as 0.

**Fix** ([prepare_spatial.py](waterpath_data_service/services/prepare_spatial.py)):
sort the shape list by bounding-box area **descending** before passing it to
`rasterize`.  Large polygons burn first; small polygons burn last and can
reclaim disputed border cells.

```python
shapes = [
    s for _, s in sorted(
        ((bbox_area, shape_tuple) for ...), key=..., reverse=True
    )
]
```

### Zero-population fallback to baseline

Even after sorting, some zones can still receive zero projected population:

- A polygon is **smaller than one pixel** at the chosen resolution – it claims
  a raster cell but the TIF source pixel for that cell is water/nodata
  (e.g. Kamrangir Char, a river island in Dhaka, whose WorldPop pixels are
  nodata by definition).
- A polygon shares its single pixel with a neighbour that wins the
  `all_touched=True` tiebreak.

In both cases, writing 0 to the CSV would be a data fabrication.  The
implementation ([projections.py](waterpath_data_service/services/projections.py))
instead retains the **baseline population** for any such zone and logs a
warning listing the affected area identifiers:

```
WARNING: Projected population is 0 for N area(s) (raster gap or water
coverage) – retaining baseline values: ['BGD.3.1.20_1', ...]
```

This means the projection for those areas is effectively a copy of the
baseline, which is the least-bad option given the source data constraints.

---

## Source TIF Conventions

The service expects population rasters in **WGS-84 (EPSG:4326)** with pixel
values representing **population counts per cell** (not density).

### Baseline raster

| Field        | Value                                              |
|--------------|----------------------------------------------------|
| Filename     | `global_pop_2025_CN_1km_R2025A_UA_v1.tif`         |
| Product      | GHS-POP R2025A (GHSL), 2025 epoch                  |
| Native res   | ~0.00833° ≈ 926 m at equator                       |
| Coverage     | Global                                             |

### Projection rasters (per SSP / year)

| Field        | Value                                              |
|--------------|----------------------------------------------------|
| Filename     | `FuturePop_<SSP>_<year>_1km_v0_2.tif`             |
| Native res   | ~0.00833° ≈ 926 m at equator                       |
| Coverage     | Global                                             |

---

## Livestock Raster Resampling

Livestock head-count rasters are generated independently of the population
pipeline, but use the same resolution-selection logic.  This section documents
the additional complications introduced by the GLW4 source data.

### Source data and units

| Source                       | Species             | Unit stored          |
|------------------------------|---------------------|----------------------|
| GLW4 2020 (`GLW4-2020.D-DA.*`) | Cattle, buffaloes, chickens, goats, pigs, sheep | **heads / km²** (density) |
| GLW4 2015 (`5_Dk_2015_Da.tif`) | Ducks               | **heads / pixel** (total) |
| Sheep + goat proxy (GLW4 2020) | Horses, asses, mules, camels, donkeys | derived from density |

The unit mismatch between the 2020 and 2015 rasters means density→counts
conversion must be applied selectively.

### Density → heads/pixel conversion

Immediately after clipping each GLW4 2020 raster to the zone grid, the array
is multiplied by a pixel-area matrix:

```
heads/pixel = (heads/km²) × pixel_area_km²
```

The pixel area is computed per row to account for the cosine-latitude dependence
of pixel width (identical formula to the population pipeline):

```
pixel_area_km² = (res_deg × 111.32)² × cos(lat)
```

This conversion is implemented in `_pixel_area_km2()` and applied in
`_generate_animal_heads_rasters()` in `livestock.py`.  The duck raster already
stores total heads/pixel, so no conversion is applied to it.  The sheep/goat
proxy arrays are converted before the proxy ratio is applied so that the proxy
species' rasters are also in heads/pixel.

### Resolution for livestock vs population rasters

The livestock zone template is built by `_build_livestock_zone_template()` using
a floored variant of the `diagonal / 100` rule:

```
raw = max(glw4_native_res, min(0.5°, diagonal / 100))
res = snap_to_nice_resolution(raw)
```

The floor at `glw4_native_res` (~0.083°) prevents the zone grid from being
finer than the GLW4 source data.  Upsampling GLW4 beyond its native resolution
would spread one source pixel's density uniformly over many smaller destination
pixels, adding no spatial information and producing an artificially smooth
distribution.

### Minimum study area requirement

The zone template also enforces a minimum extent: the study area must span at
least **4 GLW4 pixels** (~0.33°) in each spatial dimension.  Areas smaller than
this (e.g. city sub-districts at admin level 4 or finer) produce so few source
pixels that the resulting rasters carry no meaningful livestock distribution.
`_build_livestock_zone_template` raises a `ValueError` for such areas; the API
surfaces this as a 500 error with a descriptive message.

See [LIVESTOCK_ISSUES.md](LIVESTOCK_ISSUES.md) §5 for background.

### Projection: scaling heads rasters by SSP growth rates

Scenario projections scale the baseline heads rasters using per-country growth
rates derived from `livestock_future.csv`:

```
rate[alpha3] = future_heads[alpha3] / sum(baseline_pixels[alpha3])
```

Pixels in each country zone are multiplied by that country's rate.  Countries
absent from the future CSV default to rate = 1.0 (no change from baseline).
Total heads in the projected raster therefore match the future CSV values
(within rounding), regardless of the zone grid resolution.

---

## Adding Higher-Resolution Sources

If a finer source raster is available (e.g. WorldPop country tiles at 100 m ≈
0.00083°), place it in the static data directory and pass its path to
`prepare_spatial_inputs`.  The auto-resolution algorithm will automatically
lower the floor to 0.001° and the snapped output will then be at 0.001° for
city-scale study areas – no code changes required.
