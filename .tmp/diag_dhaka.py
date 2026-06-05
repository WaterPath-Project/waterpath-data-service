"""Diagnose test_dhaka rasterization — small polygons vs coarse dest grid."""
import sys, math
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds as window_from_bounds
from pathlib import Path
sys.path.insert(0, "/work")
from waterpath_data_service.services.prepare_spatial import _cell_area_km2, _native_tif_resolution, _round_to_nice_res

BASE = Path("waterpath_data_service/data/test_dhaka/baseline")
shp = BASE / "geodata/geodata.shp"
csv = BASE / "human_emissions/population.csv"
iso_p = BASE / "human_emissions/isoraster.tif"
src_p = Path("waterpath_data_service/static/data/worldpop_2025/FuturePop_SSP1_2025_1km_v0_2.tif")

import pandas as pd
df = pd.read_csv(csv)
print(f"CSV: rows={len(df)} total_pop={df.population.sum():,.0f}")

g = gpd.read_file(shp)
print(f"SHP: rows={len(g)} crs={g.crs}")
print(f"  bounds: {g.total_bounds.tolist()}")

# Per-feature area
geoms = g.geometry
if g.crs and g.crs.is_geographic:
    g_proj = g.to_crs(3857)
else:
    g_proj = g
areas_km2 = g_proj.geometry.area / 1e6
print(f"  per-polygon area km²: min={areas_km2.min():.2f} median={areas_km2.median():.2f} max={areas_km2.max():.2f} mean={areas_km2.mean():.2f}")

# Resolution selection that prepare_spatial_inputs would pick
xmin, ymin, xmax, ymax = g.total_bounds.tolist()
extent_x = xmax - xmin; extent_y = ymax - ymin
diagonal = math.hypot(extent_x, extent_y)
src_native = _native_tif_resolution(str(src_p))
target = diagonal / 100.0
raw = max(src_native, min(0.5, target))
res = _round_to_nice_res(raw)
print(f"\nAuto resolution: {res}°  (diagonal={diagonal:.4f}, src_native={src_native:.5f}, target={target:.5f})")
print(f"  cell size at lat {(ymin+ymax)/2:.2f}: ≈ {res*111.32:.2f} km on N-S, "
      f"{res*111.32*math.cos(math.radians((ymin+ymax)/2)):.2f} km on E-W")

# Build dest grid like the pipeline
xmin_p = max(math.floor(xmin/res)*res - res, -180.0)
ymin_p = max(math.floor(ymin/res)*res - res,  -90.0)
xmax_p = min(math.ceil(xmax/res)*res + res, 180.0)
ymax_p = min(math.ceil(ymax/res)*res + res,  90.0)
W = round((xmax_p - xmin_p)/res); H = round((ymax_p - ymin_p)/res)
from rasterio.transform import from_bounds
dst_trans = from_bounds(xmin_p, ymin_p, xmax_p, ymax_p, W, H)
print(f"  dest grid: {H} × {W} cells")

# Read existing isoraster
with rasterio.open(iso_p) as r:
    iso_arr = r.read(1)
print(f"\nexisting isoraster: shape={iso_arr.shape}")
unique, counts = np.unique(iso_arr[iso_arr > 0], return_counts=True)
print(f"  zones in raster: {len(unique)}  (CSV has {len(df)})")
print(f"  pixels per zone: min={counts.min()} median={int(np.median(counts))} max={counts.max()}")
missing_in_iso = set(range(1, len(df) + 1)) - set(int(u) for u in unique)
print(f"  zones MISSING from isoraster: {len(missing_in_iso)}  e.g. {list(sorted(missing_in_iso))[:10]}")
print(f"  total domain pixels: {int((iso_arr > 0).sum())}")

# Native (1km) per-feature population — ground truth
geom_union = g.geometry.union_all()
gxmin, gymin, gxmax, gymax = geom_union.bounds
with rasterio.open(src_p) as src:
    win = window_from_bounds(gxmin - 0.05, gymin - 0.05, gxmax + 0.05, gymax + 0.05, src.transform)
    sa = src.read(1, window=win, boundless=True, fill_value=0).astype("float64")
    s_trans = src.window_transform(win)
    snd = src.nodata
    if snd is not None:
        sa[np.isclose(sa, snd)] = 0.0
    sa[~np.isfinite(sa)] = 0.0
    sh, sw = sa.shape

per_feat_native_pop = []
for idx, geom in enumerate(g.geometry):
    m = rasterize([(geom.__geo_interface__, 1)], out_shape=(sh, sw),
                  transform=s_trans, fill=0, dtype=np.uint8,
                  all_touched=False).astype(bool)
    per_feat_native_pop.append((idx + 1, sa[m].sum() if m.any() else 0.0))
nat_df = pd.DataFrame(per_feat_native_pop, columns=["zone_id", "native_pop_1km"])
nat_total = nat_df.native_pop_1km.sum()
print(f"\nNative 1km source pop summed over all 46 polygons (non-overlapping all_touched=False): {nat_total:,.1f}")
print(f"CSV reports total: {df.population.sum():,.0f}")

# Per-zone, what does the regenerated raster give? (sum of urban+rural per zone)
with rasterio.open(BASE / "human_emissions/pop_urban.tif") as r:
    pu = r.read(1).astype("float64")
with rasterio.open(BASE / "human_emissions/pop_rural.tif") as r:
    pr = r.read(1).astype("float64")
pop = np.where(np.isfinite(pu) & (pu != -9999), pu, 0.0) + np.where(np.isfinite(pr) & (pr != -9999), pr, 0.0)
zone_sums = []
for z in unique:
    zone_sums.append((int(z), float(pop[iso_arr == z].sum())))
zs_df = pd.DataFrame(zone_sums, columns=["zone_id", "raster_pop"])
merged = nat_df.merge(zs_df, on="zone_id", how="left").fillna(0.0)
merged["csv_pop"] = df["population"].values[:len(merged)]
merged["diff"] = merged.raster_pop - merged.csv_pop
merged["pct"] = 100 * merged["diff"] / merged.csv_pop.replace(0, np.nan)
print("\nTop 10 by absolute discrepancy (raster - csv):")
print(merged.reindex(merged["diff"].abs().sort_values(ascending=False).index).head(10).to_string(index=False))
print(f"\nTotals: csv={merged.csv_pop.sum():,.0f}  raster={merged.raster_pop.sum():,.0f}  diff={merged.raster_pop.sum() - merged.csv_pop.sum():+,.0f}")
print(f"Native truth (sum over disjoint polygons): {nat_total:,.0f}")
print(f"  → CSV vs native: {df.population.sum() - nat_total:+,.0f} ({100*(df.population.sum() - nat_total)/nat_total:+.2f}%)")
print(f"  → raster vs native: {merged.raster_pop.sum() - nat_total:+,.0f} ({100*(merged.raster_pop.sum() - nat_total)/nat_total:+.2f}%)")
