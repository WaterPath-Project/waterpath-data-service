"""Quantify contribution of (a) all_touched mask expansion and
(b) cross-border density smearing to the chn resampling overcount."""
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
from rasterio.transform import array_bounds
from rasterio.windows import from_bounds as window_from_bounds
from pathlib import Path

iso_p = Path("waterpath_data_service/data/chn/baseline/human_emissions/isoraster.tif")
shp = Path("waterpath_data_service/data/chn/baseline/geodata/geodata.shp")
src_p = Path("waterpath_data_service/static/data/worldpop_2025/FuturePop_SSP1_2025_1km_v0_2.tif")

with rasterio.open(iso_p) as iso:
    iso_arr = iso.read(1)
    itrans, icrs, ishape = iso.transform, iso.crs, iso.shape
xmin, ymin, xmax, ymax = array_bounds(ishape[0], ishape[1], itrans)
print(f"dest grid: {ishape[0]} × {ishape[1]} ({xmin:.4f},{ymin:.4f}..{xmax:.4f},{ymax:.4f})")

# Cell-area helper
def cell_area_km2(transform, h, w):
    a = 6378.137; b = 6356.752314245; e2 = 1.0 - (b/a)**2
    rlon = abs(transform.a); rlat = abs(transform.e)
    lat = transform.f + transform.e * (np.arange(h) + 0.5)
    lat_rad = np.radians(lat)
    N = a / np.sqrt(1 - e2*np.sin(lat_rad)**2)
    M = a*(1-e2) / (1 - e2*np.sin(lat_rad)**2)**1.5
    cw = N*np.cos(lat_rad)*np.radians(rlon)
    ch = M*np.radians(rlat)
    return np.broadcast_to((cw*ch)[:,None], (h, w)).copy()

dst_area = cell_area_km2(itrans, ishape[0], ishape[1])

# === Step 1: read source within dest bbox ===
with rasterio.open(src_p) as src:
    win = window_from_bounds(xmin, ymin, xmax, ymax, src.transform)
    sa = src.read(1, window=win, boundless=True, fill_value=0).astype("float32")
    src_trans = src.window_transform(win)
    src_nd = src.nodata
    if src_nd is not None:
        sa[np.isclose(sa, src_nd)] = np.nan
    sh, sw = sa.shape
    src_area = cell_area_km2(src_trans, sh, sw)
    src_density = sa / src_area
    src_density[~np.isfinite(src_density)] = np.nan

# === Step 2: resample density to dest ===
dst_density = np.full(ishape, np.nan, dtype="float32")
reproject(
    source=src_density, destination=dst_density,
    src_transform=src_trans, src_crs="EPSG:4326",
    dst_transform=itrans, dst_crs=icrs,
    resampling=Resampling.average,
    src_nodata=np.nan, dst_nodata=np.nan,
)
dst_pop = dst_density * dst_area
print(f"\nresampled dst_pop total (entire grid): {np.nansum(dst_pop):,.1f}")

# === Step 3: rasterise polygon to dest grid with both all_touched modes ===
gdf = gpd.read_file(shp)
geom = gdf.geometry.union_all()

mask_at_true = rasterize([(geom.__geo_interface__, 1)], out_shape=ishape,
                         transform=itrans, fill=0, dtype=np.uint8,
                         all_touched=True).astype(bool)
mask_at_false = rasterize([(geom.__geo_interface__, 1)], out_shape=ishape,
                          transform=itrans, fill=0, dtype=np.uint8,
                          all_touched=False).astype(bool)

iso_mask = iso_arr > 0
print(f"\nMask comparison (cells):")
print(f"  isoraster.tif (all_touched=True burn order): {int(iso_mask.sum())}")
print(f"  rasterize(all_touched=True):                 {int(mask_at_true.sum())}")
print(f"  rasterize(all_touched=False):                {int(mask_at_false.sum())}")
print(f"  dest cell area: min={dst_area.min():.1f}, mean={dst_area.mean():.1f}, max={dst_area.max():.1f} km²")
print(f"  total domain area (iso_mask):     {dst_area[iso_mask].sum():,.0f} km²")
print(f"  total domain area (at=False):     {dst_area[mask_at_false].sum():,.0f} km²")
print(f"  China actual area: ~9,597,000 km²")

print(f"\nResampled pop sums by mask:")
print(f"  iso_mask (pipeline mask):       {np.nansum(dst_pop[iso_mask]):,.1f}")
print(f"  rasterize(all_touched=True):    {np.nansum(dst_pop[mask_at_true]):,.1f}")
print(f"  rasterize(all_touched=False):   {np.nansum(dst_pop[mask_at_false]):,.1f}")

# === Step 4: source-side comparison — sum at native resolution ===
src_pop = np.where(np.isfinite(sa), sa, 0.0)
src_in_poly_true = rasterize([(geom.__geo_interface__, 1)], out_shape=(sh, sw),
                             transform=src_trans, fill=0, dtype=np.uint8,
                             all_touched=True).astype(bool)
src_in_poly_false = rasterize([(geom.__geo_interface__, 1)], out_shape=(sh, sw),
                              transform=src_trans, fill=0, dtype=np.uint8,
                              all_touched=False).astype(bool)
print(f"\nNATIVE source pop sums (1 km, no resample):")
print(f"  inside polygon (all_touched=True):  {src_pop[src_in_poly_true].sum():,.1f}")
print(f"  inside polygon (all_touched=False): {src_pop[src_in_poly_false].sum():,.1f}")
print(f"  total in bbox:                       {src_pop.sum():,.1f}")

# === Step 5: correct way — aggregate counts by destination cell ===
# Project source COUNTS using Resampling.sum so totals are preserved.
dst_counts = np.full(ishape, np.nan, dtype="float64")
reproject(
    source=src_pop.astype("float64"), destination=dst_counts,
    src_transform=src_trans, src_crs="EPSG:4326",
    dst_transform=itrans, dst_crs=icrs,
    resampling=Resampling.sum,
    src_nodata=None, dst_nodata=np.nan,
)
print(f"\nCount-preserving resample (Resampling.sum) totals:")
print(f"  whole grid:                       {np.nansum(dst_counts):,.1f}")
print(f"  iso_mask:                          {np.nansum(dst_counts[iso_mask]):,.1f}")
print(f"  rasterize(all_touched=False):      {np.nansum(dst_counts[mask_at_false]):,.1f}")
