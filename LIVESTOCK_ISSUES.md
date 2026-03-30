# LIVESTOCK_ISSUES.md

## Livestock Raster Generation: Issues and Fixes

### 1. Camels (and Proxy Species) in Europe

**Issue:**
- Camels, horses, donkeys, mules, and asses were appearing in regions (e.g. Europe) where they do not naturally occur.
- This happened because the spatial distribution for these species was proxied by the combined sheep+goat raster, which is present in Europe.

**Fix:**
- Proxy species are now masked to their plausible IPCC regions using a region mask (`zone_ipcc_arr`).
- Camels are restricted to Africa, Asia, and NENA (Northern Africa + Western Asia).

### 2. Duck Population Artefacts (Balkans)

**Issue:**
- The GLW4 2015 duck raster had nodata values (e.g. -9999) not encoded in the file metadata.
- Bilinear interpolation during reprojection treated these as valid, causing inflated values in regions like the Balkans.

**Fix:**
- Negative values and explicit nodata are clamped to `NaN` before and after reprojection.
- Any residual negative values from interpolation are also clamped to `NaN`.

### 3. General Projection and Masking Improvements

- All animal head rasters are now projected using a session-specific zone grid.
- Negative artefacts from interpolation are removed.
- Proxy species are only distributed in biologically plausible regions.

---

**Summary:**
- Camels and other proxy species are now geographically restricted.
- Duck and other animal rasters are free from nodata interpolation artefacts.
- The process is robust to missing nodata metadata and ensures realistic spatial distributions.

**Relevant code:**
- `waterpath_data_service/services/livestock.py`
- Functions: `_clip_raster_to_zone_grid`, `_generate_animal_heads_rasters`, `_generate_animal_isoraster`

For further details, see the comments in the code and this document.

---

### 4. GLW4 Rasters Store Density, Not Total Heads

**Issue:**
GLW4 2020 rasters (cattle, buffaloes, chickens, goats, pigs, sheep) store animal
*density* in heads/km², not total heads per pixel.  The GLW4 2015 duck raster
stores total heads per pixel.  Treating density rasters as counts inflated values
in large high-latitude pixels and produced biologically impossible results (e.g.
ducks greatly outnumbering chickens in the Balkans).

**Fix:**
A `_pixel_area_km2()` helper computes the geodetic area of every pixel in the
zone grid:

```
area_km² = (res_deg × 111.32)² × cos(lat)
```

GLW4 2020 density arrays are multiplied by this pixel-area matrix immediately
after clipping, converting them to total heads per pixel before any further
processing.  The duck raster is left unchanged.  The sheep/goat proxy arrays
used for horses, asses, mules, and camels are also converted before the proxy
ratio is applied.

---

### 5. Livestock Not Supported for Sub-District Study Areas

**Known Limitation:**
GLW4 has a native resolution of ~0.083° (~9 km per pixel).  For very small urban
study areas — such as Kampala level-4 sub-districts (extent ~0.16° × 0.19°) —
the entire study area is covered by fewer than 4 GLW4 pixels in each dimension.
This produces a near-empty raster with no meaningful spatial distribution of
animal heads, and the result cannot sensibly align with the fine-grained
human-emissions isoraster.

**Decision:**
Livestock inputs are **not supported** for study areas where the spatial extent
is less than 4 × the GLW4 native pixel size (~0.33°) in either direction.
`_build_livestock_zone_template` raises a `ValueError` for such areas, which
the API surfaces as a 500 error with a descriptive message.

The GLW4 native-resolution floor is retained in the resolution logic:
```python
raw = max(glw4_native_res, min(0.5, diagonal / 100.0))
```

**Affected case studies:** Any study area at sub-district scale (typically
admin level 4 or finer) where the bounding box is smaller than ~0.33° in
either longitude or latitude extent.