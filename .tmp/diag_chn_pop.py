"""Diagnose population sum discrepancy for the CHN baseline."""
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds, array_bounds
from rasterio.windows import from_bounds as window_from_bounds

BASE = Path("waterpath_data_service/data/chn/baseline/human_emissions")
iso_p   = BASE / "isoraster.tif"
urb_p   = BASE / "pop_urban.tif"
rur_p   = BASE / "pop_rural.tif"
csv_p   = BASE / "population.csv"

# Find a matching source pop raster (FuturePop_*_2025*).  We need the same
# raster the pipeline used.  Try to find a baseline year file.
src_dir = Path("waterpath_data_service/static/data/worldpop_2025")
candidates = sorted(src_dir.glob("FuturePop_*_2025_1km_v0_2.tif"))
print(f"candidate worldpop rasters for baseline (2025): {[c.name for c in candidates]}")
src_p = candidates[0] if candidates else None

# 1. CSV totals
df = pd.read_csv(csv_p)
csv_total = float(df["population"].sum())
print(f"\nCSV population total           : {csv_total:>20,.1f}")
print(f"CSV rows                       : {len(df)}")
print(f"CSV fraction_urban_pop stats   : min={df['fraction_urban_pop'].min():.4f} "
      f"mean={df['fraction_urban_pop'].mean():.4f} max={df['fraction_urban_pop'].max():.4f}")

# 2. Raster sums
def load(p):
    with rasterio.open(p) as src:
        a = src.read(1).astype("float64"); nd = src.nodata
        m = np.isfinite(a)
        if nd is not None:
            m &= ~np.isclose(a, nd)
        return a, m, src.transform, src.crs, src.bounds, src.shape, nd

ua, um, _, _, _, _, _ = load(urb_p)
ra, rm, _, _, _, _, _ = load(rur_p)
ia, im, itrans, icrs, ibounds, ishape, _ = load(iso_p)

urb_sum = ua[um].sum()
rur_sum = ra[rm].sum()
total_sum = urb_sum + rur_sum
print(f"\npop_urban sum                  : {urb_sum:>20,.1f}")
print(f"pop_rural sum                  : {rur_sum:>20,.1f}")
print(f"pop_urban + pop_rural          : {total_sum:>20,.1f}")
print(f"Diff (raster - csv)            : {total_sum - csv_total:>20,.1f}  "
      f"({100*(total_sum-csv_total)/csv_total:+.2f}%)")

# 3. Domain pixels (isoraster > 0)
dom_mask = ia > 0
print(f"\nisoraster shape                : {ia.shape}")
print(f"isoraster pixels with id>0     : {int(dom_mask.sum())}")
print(f"unique zone ids (>0)           : {int(np.unique(ia[dom_mask]).size)}")

# 4. Source raster: read same window the pipeline reads (boundless, fill 0)
if src_p is None:
    raise SystemExit("No baseline worldpop raster available for comparison.")
print(f"\nSource raster                  : {src_p.name}")
xmin, ymin, xmax, ymax = array_bounds(ishape[0], ishape[1], itrans)
with rasterio.open(src_p) as src:
    print(f"  src CRS={src.crs}  res=({abs(src.transform.a):.5f}, {abs(src.transform.e):.5f})  nodata={src.nodata}")
    win = window_from_bounds(xmin, ymin, xmax, ymax, src.transform)
    sa = src.read(1, window=win, boundless=True, fill_value=0).astype("float64")
    snd = src.nodata
    sm = np.isfinite(sa)
    if snd is not None:
        sm &= ~np.isclose(sa, snd)
    sa_valid = sa[sm]
    print(f"  window read shape          : {sa.shape}")
    print(f"  source sum (window, valid) : {sa_valid.sum():,.1f}")
    print(f"  source sum (window, raw)   : {sa.sum():,.1f}")

# 5. Reproduce the resampling the pipeline does: density → average → counts
def cell_area_km2(transform, height, width):
    a = 6378.137
    b = 6356.752314245
    e2 = 1.0 - (b / a) ** 2
    res_lon = abs(transform.a); res_lat = abs(transform.e)
    lat_centers = transform.f + transform.e * (np.arange(height) + 0.5)
    lat_rad = np.radians(lat_centers)
    N = a / np.sqrt(1.0 - e2 * np.sin(lat_rad) ** 2)
    M = a * (1.0 - e2) / (1.0 - e2 * np.sin(lat_rad) ** 2) ** 1.5
    cw = N * np.cos(lat_rad) * np.radians(res_lon)
    ch = M * np.radians(res_lat)
    return np.broadcast_to((cw * ch)[:, None], (height, width)).copy()

dst_h, dst_w = ishape
dst_area = cell_area_km2(itrans, dst_h, dst_w)
print(f"\nDest grid cell area km²: min={dst_area.min():.4f} mean={dst_area.mean():.4f} max={dst_area.max():.4f}")

with rasterio.open(src_p) as src:
    win = window_from_bounds(xmin, ymin, xmax, ymax, src.transform)
    sa = src.read(1, window=win, boundless=True, fill_value=0).astype("float32")
    src_trans = src.window_transform(win)
    src_nd = src.nodata
    if src_nd is not None:
        sa[np.isclose(sa, src_nd)] = np.nan
    sh, sw = sa.shape
    src_area = cell_area_km2(src_trans, sh, sw)
    src_density = sa / src_area  # pop / km²
    src_density[~np.isfinite(src_density)] = np.nan

    dst_density = np.full((dst_h, dst_w), np.nan, dtype="float32")
    reproject(
        source=src_density,
        destination=dst_density,
        src_transform=src_trans,
        src_crs=src.crs,
        dst_transform=itrans,
        dst_crs=icrs,
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
dst_pop = dst_density * dst_area
print(f"resampled dst_pop sum (raw)    : {np.nansum(dst_pop):,.1f}")
print(f"resampled dst_pop sum (mask>0) : {np.nansum(dst_pop[dom_mask]):,.1f}")

# 6. Effect of NaN→0 fill inside the domain (the suspicious line)
in_domain_nan = dom_mask & ~np.isfinite(dst_pop)
print(f"\nDomain pixels                  : {int(dom_mask.sum())}")
print(f"  …with NaN in resampled pop   : {int(in_domain_nan.sum())} "
      f"({100*in_domain_nan.sum()/max(dom_mask.sum(),1):.2f}%)")
print(f"  …positive resampled pop pix  : {int((dom_mask & (dst_pop > 0)).sum())}")

# 7. Per-zone comparison: raster (urban+rural) vs csv per gid
# Build zone-id → gid mapping the same way the pipeline does: row order = 1..N
gid_to_pop = {}
for idx, row in df.iterrows():
    gid_to_pop[idx + 1] = float(row["population"])

zone_ids = np.unique(ia[dom_mask])
print(f"\nzone ids covered by isoraster  : {len(zone_ids)} (csv has {len(df)} rows)")

# total per-zone from urban+rural raster
combined = np.where(um, ua, 0.0) + np.where(rm, ra, 0.0)
sums = {}
for z in zone_ids:
    sums[int(z)] = float(combined[ia == z].sum())

# Compare a few biggest discrepancies
rows = []
for z, rs in sums.items():
    cs = gid_to_pop.get(z, np.nan)
    rows.append((z, cs, rs, rs - cs, (rs - cs) / cs * 100 if cs else np.nan))
cmp = pd.DataFrame(rows, columns=["zone_id", "csv_pop", "raster_pop", "diff", "pct"])
cmp = cmp.sort_values("diff", key=abs, ascending=False)
print("\nTop 10 zones by absolute discrepancy (raster - csv):")
print(cmp.head(10).to_string(index=False))

print("\nOverall: ")
print(f"  sum(csv_pop over zones)   : {cmp.csv_pop.sum():,.1f}")
print(f"  sum(raster_pop over zones): {cmp.raster_pop.sum():,.1f}")
