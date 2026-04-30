# Livestock High Emissions — Root Cause & Fix

## Summary

When the livestock module is enabled, GloWPa produces land emission values
several orders of magnitude higher than human emissions for all case studies.
Three compounding unit mismatches in the animal isodata files cause this.
Bugs 1 and 2 affect all non-poultry livestock; Bug 3 affects chickens and ducks only.

---

## Root Cause

### Bug 1 — Prevalence values are in % but the model expects a fraction (0–1)

**GloWPa documentation** (`README`, section *Livestock Animal Isodata*) states:

| Column | Unit | Range |
|---|---|---|
| `prev_young` | — | **0-1** |
| `prev_adult` | — | **0-1** |

**GloWPa model formula** (`R/animal.R`):
```r
animal_emission_excr_mass <- function(manure_production, prev, excr) {
  # factor 1000 is added to go from oocysts per gram to oocysts per kilogram
  animal_emission <- manure_production * prev * excr * 1000
  return(animal_emission)
}
```
`prev` is used **directly as a fraction** — the model performs no `/100` conversion.

**Actual values** produced by `prepare_livestock_isodata()` (from `vermeulen_2017/animals.csv`):

| Animal | `prev_young` | `prev_adult` |
|---|---|---|
| cattle | 30.3 | 13.4 |
| pigs | 25.4 | 19.8 |
| sheep | 25.1 | 13.2 |
| goats | 15.8 | 10.6 |
| buffaloes | 27.0 | 14.4 |

These are clearly **percentage values** (e.g. 30.3 %), not fractions (e.g. 0.303).

Note: The GloWPa README's own example table for cattle isodata also shows
`prev_young = 30.3`, which is inconsistent with the documented "0-1" range.
This contradiction exists in the upstream package. The model formula makes clear
that fraction-scale values are expected, so the source data is wrong.

**Effect: 100× overestimate** of pathogen emissions from livestock.

---

### Bug 2 — `manure_per_mass` unit is kg per 1000 kg LW/day, but the formula treats it as kg per kg LW/day

**GloWPa documentation** (`README`, section *Livestock Animal Isodata*):

| Column | Unit |
|---|---|
| `manure_per_mass` | **kg kg⁻¹⁰⁰⁰** (kg manure per 1000 kg live weight per day) |

**GloWPa model formula** (`R/animal.R`):
```r
animal_manure_production <- function(mass, manure_per_mass) {
  # unit kg/year
  manure_production <- mass * manure_per_mass * 365
  return(manure_production)
}
```
The formula multiplies `mass` (kg) by `manure_per_mass` directly. There is **no division by 1000** to account for the "per 1000 kg LW" denominator.

**Example — adult cattle, iso=3 (Europe)**:

| | formula | result |
|---|---|---|
| As coded (bug) | 550 kg × 72 × 365 | **14,454,000 kg/year per animal** |
| Correct | 550 kg × (72 / 1000) × 365 | **14,454 kg/year per animal** (~40 kg/day ✓) |

**Effect: 1000× overestimate** of manure production, which feeds directly into emission calculations.

---

### Bug 3 — Spurious ×1000 factor in the poultry emission formula

Chickens and ducks use a different emission pathway in `R/animal.R`: instead of
manure mass × excretion concentration, they use a pre-integrated daily excretion
value (`excr_day`).

**GloWPa documentation** (`README`, section *Livestock Animal Isodata*):

| Column | Unit | Applies to |
|---|---|---|
| `excr_day` | **particles day⁻¹** | chickens, ducks only |

`particles day⁻¹` is a **total** count of oocysts per animal per day — not a
concentration per gram of manure.

**GloWPa model formula** (`R/animal.R`):
```r
animal_emission_excr_day <- function(prev, excr_day) {
  # factor 1000 is added to go from oocysts per gram to oocysts per kilogram
  # factor 365 is added to compute the yearly emissions
  animal_emission <- prev * excr_day * 365 * 1000
  return(animal_emission)
}
```
The comment says "oocysts per gram → oocysts per kilogram", which is only
meaningful when the excretion value is a concentration (oocysts/g). Because
`excr_day` is a total daily count (oocysts/day), the ×1000 has no physical
basis — it is a copy-paste artefact from the non-poultry function.

**Back-of-envelope check — chickens (iso=3)**:

| | formula | result |
|---|---|---|
| As coded (bug) | 0.147 × 5,823,759 × 365 × 1000 | **3.1 × 10¹¹ oocysts/year/bird** |
| Correct | 0.147 × 5,823,759 × 365 | **3.1 × 10⁸ oocysts/year/bird** |

3.1 × 10⁸ is consistent with literature-reported Cryptosporidium loads from
chickens; 3.1 × 10¹¹ is three orders of magnitude too high.

**Effect: 1000× overestimate** of pathogen emissions from chickens and ducks.

---

### Combined effect

For non-poultry livestock, Bugs 1 and 2 multiply together:

$$\text{emission}_\text{actual} = \text{emission}_\text{correct} \times 100 \times 1000 = \text{emission}_\text{correct} \times 100{,}000$$

For poultry (chickens, ducks), only Bug 3 applies:

$$\text{emission}_\text{actual} = \text{emission}_\text{correct} \times 1000$$

This is consistent with the observed output: all livestock emissions are vastly
overestimated, and poultry emissions dominate because they do not benefit from
the partial cancellation that would occur if the non-poultry bugs were fixed alone.

---

## Proposed Fix — Data Preparation Phase

The fix should be applied at the point where animal isodata is generated,
before it is written to disk. Three corrections are needed:

1. Divide `prev_young` and `prev_adult` by **100** to convert from percentage
   scale to the fraction scale (0–1) that GloWPa's model formula expects.
   *(Applies to all animals except asses — see note below.)*
2. Divide `manure_per_mass` by **1000** to convert from the documented unit
   (kg per 1000 kg live weight per day) to the unit implicitly assumed by
   GloWPa's `animal_manure_production()` formula (kg per kg live weight per day).
   *(Applies to all non-poultry animals.)*
3. Divide `excr_day` by **1000** to compensate for the spurious ×1000 factor
   in `animal_emission_excr_day()`. *(Applies to chickens and ducks only.)*

Applying these corrections at data generation time ensures all subsequently
created case studies have the right values.

---

## Existing Case Study Data

Any case study that was already prepared (i.e. has `isodata_*.csv` files already
written to `input/baseline/livestock_emissions/animals/`) contains the wrong
values. Those files need to be corrected as well — either by:

1. **Re-running data preparation** for the affected case studies (cleanest option),
   or
2. **Applying the same factor corrections directly to the existing CSVs**:
   - divide `prev_young` and `prev_adult` by 100 (all animals except asses),
   - divide `manure_per_mass` by 1000 (all non-poultry animals),
   - divide `excr_day` by 1000 (chickens and ducks only).

Note: `asses` may already have near-correct prevalence values
(`prev_young = 0.9`, `prev_adult = 0.9`) because their true prevalence is
genuinely below 1 %. Check whether the correction pushes them to implausibly
small values (0.009) and exclude `asses` from the prevalence correction if
needed.

---

## Upstream Issue

All three discrepancies originate in the GloWPa R package:

- The `vermeulen_2017/animals.csv` source data stores prevalence in % scale.
- `animal_manure_production()` lacks the `/1000` divisor documented for
  `manure_per_mass`.
- `animal_emission_excr_day()` contains a spurious `× 1000` factor that is
  only appropriate for excretion values in oocysts/gram, not for the total
  oocysts/day values that `excr_day` represents.

It is worth reporting these to the GloWPa maintainers
(<https://git.wur.nl/glowpa/glowpa-r/-/issues>) so the package itself can be
corrected for downstream users.
