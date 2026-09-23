# Re-scaling Methodology (Shared Framework)

This document explains, in plain language, the **single re-scaling approach** the
WaterPath data service uses everywhere it moves data from a source raster grid
onto a study-area grid. Population, hydrology, QMRA, and livestock all follow the
same rules. New code should follow them too.

The goal of re-scaling is simple:

> Take a global data layer (population, runoff, animal density, …) and place it
> onto the study-area grid **without inventing detail that is not in the source
> and without losing or duplicating totals.**

---

## 1. The one idea you must remember

Every raster value is one of two types:

| Type | Meaning | Examples | Does the value change if the pixel size changes? |
|------|---------|----------|--------------------------------------------------|
| **Extensive** | A **total / count** inside the pixel | people per pixel, animal heads per pixel | **Yes** – a bigger pixel holds more |
| **Intensive** | A **density / rate / average** | people per km², animal heads per km², mm of runoff, °C, % prevalence | **No** – it describes a place, not a pixel |

**The rule:**

> **Always resample in the *intensive* (density) space. Never resample raw
> counts.** If the data is a count, first divide by pixel area to get a density,
> resample the density, then multiply by the *new* pixel area to get counts back.

Why: densities can be averaged or copied between grids safely, because they do
not depend on pixel size. Counts cannot – averaging counts across different
pixel sizes silently creates or destroys people/animals.

---

## 2. The pixel-area helper (used by every count↔density conversion)

Because a degree of longitude shrinks toward the poles, pixel area is computed
per row from latitude:

```
pixel_area_km²  =  (resolution_in_degrees × 111.32)²  ×  cos(latitude)
```

(The population pipeline uses a slightly more precise ellipsoid version with the
meridional and prime-vertical radii of curvature, matching R's
`terra::cellSize`; the cosine formula above is the same idea and is what the
livestock pipeline uses.)

This single helper is what makes "counts → density → counts" possible on any
grid, at any latitude.

---

## 3. Choosing the resampling operator

Once everything is a density/rate, we still have to move it from source pixels to
destination pixels. Which operator we use depends **only on whether the
destination grid is coarser or finer than the source**:

| Situation | Operator | Why |
|-----------|----------|-----|
| **Destination coarser than source** (aggregation, e.g. country/global runs) | `average` | Averages all source densities inside each big destination pixel. Conserves totals. |
| **Destination finer than source** (refinement / small study areas) | `nearest` | Copies the one source cell each fine pixel falls inside. Produces honest **uniform blocks** – no invented gradients, no nodata "bleed". |
| **Smooth continuous field only** (e.g. air/river temperature) | `bilinear` | Safe *only* when the field is genuinely smooth and has no sparse nodata. |

**Never use `bilinear` on sparse or nodata-heavy grids** (rivers, GDP, animal
rasters with `-9999`): it blends neighbouring cells and drags nodata into valid
pixels, fabricating values near edges.

The important consequence for **small study areas**: when the study area is
smaller than the source cells, the destination grid is finer than the source, so
we use **`nearest`**. Each destination pixel honestly repeats the value of the
single source cell that covers it. We then annotate this in `assumptions.csv` so
the output is self-documenting ("values are uniform coarse blocks, sub-cell
variation is not resolved").

---

## 4. How each pipeline applies the framework

| Pipeline | Source data | Type | Steps |
|----------|-------------|------|-------|
| **Population** (`prepare_spatial.py`) | GHS-POP counts/pixel | Extensive | counts → ÷ area → density → `average` → × new area → counts |
| **Hydrology / QMRA** (`hydrology.py`, `qmra.py`) | ISIMIP3b 0.5°, Kummu GDP | Intensive | reproject density/rate onto the isoraster grid with `nearest` |
| **Livestock – GLW4 2020** (`livestock.py`) | heads/km² (density) | Intensive → Extensive | `nearest`/`bilinear` onto zone grid → × pixel area → heads/pixel |
| **Livestock – GLW4 2015 ducks** | heads/pixel (already counts) | Extensive | clipped directly, no area conversion |
| **Livestock projections** | FAOSTAT national totals | Extensive | scale each country's baseline pixels by `future_total / baseline_pixel_sum` |

Notice every row is just the **same idea** – get into density space (or stay
there), pick `average`/`nearest` by direction, and only convert back to counts at
the very end using the destination pixel area.

### Livestock projection scaling (per-country total matching)

Projections keep the baseline **spatial pattern** but force each country's
**total** to match the future FAOSTAT value:

```
rate[country]   = future_heads[country] / sum(baseline_heads_pixels[country])
projected_pixel = baseline_pixel × rate[country]
```

Countries missing from the future file keep `rate = 1.0` (unchanged). This is the
extensive-quantity equivalent of the density trick: we rescale totals, not
individual pixels' meaning.

---

## 5. Worked example — Kampala level 4 (small / high-resolution study area)

This is the case the livestock pipeline currently **blocks** and that we are
enabling. All numbers below are the real values for the `kampala_level_4`
session.

**The study area**

| Property | Value |
|----------|-------|
| Extent | **0.157° × 0.189°** (~17 km × 21 km) |
| Human-emissions `isoraster.tif` grid | **19 × 22 pixels at 0.01°**, bounds `[32.49, 0.20, 32.68, 0.42]` |
| GLW4 livestock native cell | **~0.0833° (1/12°) ≈ 9.3 km ≈ 86 km²** |

The whole study area is only about **1.9 × 2.3 GLW4 cells** – fewer than 4 cells
in each direction, which is why the old code raised an error. One GLW4 cell is
about **8 isoraster pixels wide**, so a block of roughly 8×8 fine pixels shares a
single GLW4 value.

**Old behaviour:** `ValueError` – livestock not generated.

**New behaviour (same framework as hydrology):**

1. **Pick the grid.** Use the existing human-emissions `isoraster.tif` grid
   (19 × 22 @ 0.01°) as the livestock zone grid, so livestock pixels line up
   exactly with the human-emissions pixels.
2. **Resample in density space.** The GLW4 2020 cattle layer is already a density
   (heads/km²). The destination (0.01°) is *finer* than the source (0.0833°), so
   use **`nearest`**: every 0.01° pixel copies the density of the GLW4 cell it
   sits inside. The result is a few uniform blocks – honest, no invented detail.
3. **Convert density → heads using the destination pixel area.** Near the equator
   a 0.01° pixel is about:

   ```
   area = (0.01 × 111.32)² × cos(0.31°) ≈ 1.24 km²
   ```

   Suppose the GLW4 cell over central Kampala holds **15 cattle/km²**
   (illustrative). Then each isoraster pixel gets:

   ```
   15 heads/km² × 1.24 km² ≈ 18.6 cattle per pixel
   ```

4. **Totals stay correct.** Summing over the valid study-area pixels gives:

   ```
   15 heads/km² × (total study-area land = ~372 km²) ≈ 5,580 cattle
   ```

   That is exactly *15 × the study-area area* – i.e. precisely the slice of the
   surrounding GLW4 cells' cattle that lies inside Kampala level 4. Nothing is
   double-counted, nothing is fabricated. The **exact same arithmetic** works if
   the study area were even smaller and fell entirely inside **one** GLW4 cell:
   every pixel simply repeats that one cell's density, and the head total scales
   with the covered area.

5. **Document it.** A `livestock_resolution` row is written to
   `livestock_emissions/assumptions.csv`, mirroring the hydrology message:

   > Study area extent (0.157° × 0.189°) is smaller than 4 native GLW4 livestock
   > cells (~0.083° / ~9 km). Head rasters are reprojected to the isoraster grid
   > with nearest-neighbour resampling, so livestock values are uniform (coarse)
   > blocks copied from the overlapping source cell(s) and do not resolve
   > sub-cell spatial variation.

---

## 6. Do / Don't checklist

**Do**

- Convert counts → density before resampling; convert back only at the end.
- Use `average` when coarsening, `nearest` when refining.
- Compute pixel area with the latitude-aware formula for every count↔density step.
- Align small study areas to the `isoraster.tif` grid so all layers overlay.
- Record a resolution note in `assumptions.csv` whenever output is uniform blocks.

**Don't**

- Don't resample raw counts.
- Don't use `bilinear` on sparse / nodata rasters (rivers, GDP, animal heads).
- Don't invent sub-cell detail; label uniform-block output as an assumption.
- Don't block generation for small areas – produce honest coarse output instead.
