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