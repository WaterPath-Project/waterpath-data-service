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

---

### 6. Proxy Species Over-Estimated in Countries with Atypical Species Mixes (e.g. Uganda)

**Issue:**
Horses, donkeys, mules, asses, and camels have no dedicated GLW4 raster.
Previously they were distributed by scaling the combined sheep+goat spatial
proxy by a single *global* ratio:

```
species_heads_pixel = (sheep+goat)_pixel × (global FAOSTAT species total / global FAOSTAT sheep+goat total)
```

This assumes every country has the same camel-to-sheep-goat ratio as the world
average, which is badly wrong for many countries.  Uganda illustrates the
problem:

| Species | Uganda census (2016) | Old estimate | Error factor |
|---------|---------------------|--------------|-------------|
| Camels | 12,000 | ~440,000 | ~36× over |
| Donkeys | 63,000 | ~450,000 | ~7× over |

Uganda has ~21 million sheep+goats — one of the larger national totals in
Africa — so multiplying by the global camel/sheep-goat ratio (≈ 0.02) produced
a vastly inflated camel count.  The error is proportional to how much a country
deviates from the global species mix.

For ducks, the old code applied a single global 2015→2020 FAOSTAT growth factor,
ignoring national trends.

**Fix implemented:**
The new `_fao_country_total(fao, iso3, item_name, year)` helper looks up a
country's reported FAOSTAT animal count using its ISO3 country code.  It returns
`None` when the country, animal, or year is missing, so missing data is not
mistaken for a reported zero.

*Proxy species (horses, donkeys, asses, mules, camels):*
- For each country, the code reads the 2020 FAOSTAT totals for the proxy animal,
  sheep, and goats.
- It calculates that country's species mix:
  `FAOSTAT_species / (FAOSTAT_sheep + FAOSTAT_goats)`.
- That country ratio is applied to the local sheep+goat raster pattern inside
  the study area.
- If any required country value is missing, the code keeps the old global ratio
  and writes a warning to the log.

Using a ratio is important for subnational studies.  For example, Kampala must
receive its estimated share of Uganda's animals, not Uganda's entire national
animal population.

*Ducks:*
- For each country, the code calculates the FAOSTAT growth ratio:
  `FAOSTAT_ducks_2020 / FAOSTAT_ducks_2015`.
- The ratio is applied to the local GLW4 2015 duck pattern.
- If either country total is missing, the code keeps the global 2015→2020 ratio
  and writes a warning to the log.

The spatial *pattern* still comes from the proxy or GLW4 raster.  FAOSTAT changes
the country-specific ratio or growth rate, not the location of animals inside
the study area.

FAOSTAT does not contain every animal for every country.  For example, the
current source contains Uganda's 2020 asses count but no Uganda camel or duck
count.  Asses therefore use Uganda's country ratio, while camels and ducks keep
the global fallback.  Missing data is not treated as proof that an animal is
absent; the separate country gate in section 7 handles that decision.

Future scenarios keep the same complete set of animal rasters as the baseline.
Donkeys use the projected growth rate for the FAOSTAT `Asses` category.  A
species with no matching future projection keeps its country-scaled baseline
distribution instead of disappearing from the scenario folder.

**GLW4 direct species (cattle, chickens, goats, pigs, sheep, buffaloes):**
GLW4 2020 is already calibrated per country, so these rasters do not need the
FAOSTAT scaling proposed here.  They are still checked by the country-level
species gate described in section 7.  This prevents a small positive GLW4 value
from creating livestock that the case-study country does not support.

**Relevant code changes:**
- `waterpath_data_service/services/livestock.py`
  - New function: `_fao_country_total()`
  - Updated function: `_generate_animal_heads_rasters()` with `zone_idx` and
    `mapping`; per-country ratios for ducks and proxy species.
  - Updated function: `generate_livestock_projection_rasters()`; carries all
    baseline animal rasters into scenarios when no separate future projection
    is available.

---

### 7. Species Shown in a Raster but absent from the Country (e.g. Buffaloes in Uganda)

**Issue:**
GLW4 is a global raster dataset.  In a small study area it can contain a few
positive pixels for an animal that is not actually kept as livestock in that
country.  This can be caused by interpolation, model uncertainty, or nearby
wild populations.

Uganda is an example.  The GLW4 buffalo raster contained positive values around
Kampala, so the generated `production_systems.csv` said that buffaloes used
65% intensive and 35% extensive systems.  However, the Vermeulen manure
management table has no buffalo management fractions for Uganda.  GloWPa saw
the non-zero production fractions, tried to read the missing buffalo management
data, and failed.

**Fix implemented:**
The livestock generator now checks whether each animal is supported for the
case-study country before accepting values from a global raster.

The check uses the `Tot_<species>` fields in
`manure_management_systems.csv`.  These fields are checksums of the manure
management fractions; they are **not animal population counts**.  Normally the
fractions add up to about 1.0.  A special pattern of an explicit
`Tot_<species> = 0.0` together with empty management fractions is treated as a
signal that the species is not managed livestock in that country.

For Uganda, `Tot_buffaloes` is `0.0` and all buffalo management fractions are
empty.  Buffaloes are therefore gated off for Kampala.

When a species is gated off, the generator makes all related outputs agree:

- The animal-heads TIF contains `0.0` in every valid study-area grid cell.
- Cells outside the study area remain nodata (`NaN`).
- Production-system fractions are set to `0.0`.
- Manure fractions are set to `0.0`.
- Manure-management fractions are written as numeric `0.0`, not empty values.

Keeping the TIF file and writing zero values is intentional.  All animal rasters
still have the same grid, extent, and coordinate system, while zero heads means
that the animal contributes no manure or emissions.  Writing nodata everywhere
could instead look like a missing or broken input to downstream software.

The same check is available for cattle, buffaloes, poultry, pigs, sheep, goats,
horses, asses/donkeys, mules, and camels.  It is not hard-coded specifically for
Uganda or buffaloes.

**Relevant code:**
- `waterpath_data_service/services/livestock.py`
  - `_load_absent_livestock_species()`
  - `_generate_production_systems()`
  - `_generate_manure_fractions()`
  - `_generate_manure_management()`
  - `_generate_animal_heads_rasters()`
  - `generate_livestock_tabular_inputs()`