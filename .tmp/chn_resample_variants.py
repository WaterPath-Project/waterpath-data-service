"""Test whether the overcount is caused by NaN/0 asymmetry in Resampling.average:
fill_value=0 vs leaving as NaN."""
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds as window_from_bounds
from pathlib import Path
import sys; sys.path.insert(0, "waterpath_data_service/services")
from prepare_spatial import _cell_area_km2

iso_p = Path("waterpath_data_service/data/chn/baseline/human_emissions/isoraster.tif")
src_p = Path("waterpath_data_service/static/data/worldpop_2025/FuturePop_SSP1_2025_1km_v0_2.tif")

with rasterio.open(iso_p) as iso:
    iso_arr = iso.read(1)
    itrans, icrs, ishape = iso.transform, iso.crs, iso.shape
xmin, ymin, xmax, ymax = array_bounds(ishape[0], ishape[1], itrans)
iso_mask = iso_arr > 0

dst_area = _cell_area_km2(itrans, ishape[0], ishape[1])

with rasterio.open(src_p) as src:
    win = window_from_bounds(xmin, ymin, xmax, ymax, src.transform)
    sa_raw = src.read(1, window=win, boundless=True, fill_value=0).astype("float64")
    src_trans = src.window_transform(win)
    src_nd = src.nodata

sh, sw = sa_raw.shape
src_area = _cell_area_km2(src_trans, sh, sw)

print(f"Source window: {sh}x{sw}, dst: {ishape[0]}x{ishape[1]}")
print(f"source pixel count: {sh*sw:,}")
nd_mask = np.isclose(sa_raw, src_nd) if src_nd is not None else np.zeros_like(sa_raw, bool)
print(f"  pixels at source nodata sentinel: {int(nd_mask.sum())} ({100*nd_mask.mean():.2f}%)")
print(f"  pixels == 0 exactly:               {int((sa_raw == 0).sum())} ({100*(sa_raw==0).mean():.2f}%)")
print(f"  pixels > 0:                        {int((sa_raw > 0).sum())} ({100*(sa_raw>0).mean():.2f}%)")
print(f"  pixel value stats: min={sa_raw.min()} max={sa_raw.max():.1f} sum={sa_raw[~nd_mask].sum():,.1f}")

# --- Variant A: current pipeline (NaN for nodata, then density average) ---
sa_a = sa_raw.copy().astype("float32")
sa_a[nd_mask] = np.nan
density_a = sa_a / src_area
density_a[~np.isfinite(density_a)] = np.nan
dst_dens_a = np.full(ishape, np.nan, dtype="float32")
reproject(source=density_a, destination=dst_dens_a,
          src_transform=src_trans, src_crs="EPSG:4326",
          dst_transform=itrans, dst_crs=icrs,
          resampling=Resampling.average,
          src_nodata=np.nan, dst_nodata=np.nan)
pop_a = dst_dens_a * dst_area
print(f"\n[Variant A: current — NaN nodata, density average]")
print(f"  total grid sum: {np.nansum(pop_a):,.1f}")
print(f"  iso_mask sum:   {np.nansum(pop_a[iso_mask]):,.1f}")

# --- Variant B: treat nodata as 0, density average ---
sa_b = sa_raw.copy().astype("float32")
sa_b[nd_mask] = 0.0
density_b = sa_b / src_area
dst_dens_b = np.full(ishape, np.nan, dtype="float32")
reproject(source=density_b, destination=dst_dens_b,
          src_transform=src_trans, src_crs="EPSG:4326",
          dst_transform=itrans, dst_crs=icrs,
          resampling=Resampling.average,
          src_nodata=None, dst_nodata=np.nan)
pop_b = dst_dens_b * dst_area
print(f"\n[Variant B: 0-fill nodata, density average]")
print(f"  total grid sum: {np.nansum(pop_b):,.1f}")
print(f"  iso_mask sum:   {np.nansum(pop_b[iso_mask]):,.1f}")

# --- Variant C: Resampling.sum on counts directly ---
sa_c = sa_raw.copy()
sa_c[nd_mask] = 0.0
dst_c = np.full(ishape, np.nan, dtype="float64")
reproject(source=sa_c, destination=dst_c,
          src_transform=src_trans, src_crs="EPSG:4326",
          dst_transform=itrans, dst_crs=icrs,
          resampling=Resampling.sum,
          src_nodata=None, dst_nodata=np.nan)
print(f"\n[Variant C: Resampling.sum on counts]")
print(f"  total grid sum: {np.nansum(dst_c):,.1f}")
print(f"  iso_mask sum:   {np.nansum(dst_c[iso_mask]):,.1f}")

# native truth
print(f"\nNative-resolution source sum in bbox: {sa_c.sum():,.1f}")
