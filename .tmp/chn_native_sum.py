"""Ground-truth China population by masking the source 1km raster
with the China polygon — no resampling."""
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds as window_from_bounds
from pathlib import Path

shp = Path("waterpath_data_service/data/chn/baseline/geodata/geodata.shp")
src_p = Path("waterpath_data_service/static/data/worldpop_2025/FuturePop_SSP1_2025_1km_v0_2.tif")

gdf = gpd.read_file(shp)
print("shapefile rows:", len(gdf), "crs:", gdf.crs)
print(gdf[[c for c in gdf.columns if c != 'geometry']].head().to_string())
geom_union = gdf.geometry.union_all()
xmin, ymin, xmax, ymax = geom_union.bounds
print(f"polygon bounds: ({xmin:.4f}, {ymin:.4f}, {xmax:.4f}, {ymax:.4f})")

with rasterio.open(src_p) as src:
    print(f"src CRS={src.crs} res={abs(src.transform.a):.5f} nodata={src.nodata}")
    win = window_from_bounds(xmin, ymin, xmax, ymax, src.transform)
    data = src.read(1, window=win, boundless=True, fill_value=0).astype("float64")
    trans = src.window_transform(win)
    nd = src.nodata
    # treat nodata as 0 for in-domain summation purposes
    if nd is not None:
        data = np.where(np.isclose(data, nd), 0.0, data)
    data = np.where(np.isfinite(data), data, 0.0)

    h, w = data.shape
    print(f"window: {h} x {w}")
    print(f"source sum over bbox window (all): {data.sum():,.1f}")

    # Rasterize the polygon at source resolution (1km) to mask exactly inside polygon
    mask = rasterize(
        [(geom_union.__geo_interface__, 1)],
        out_shape=(h, w),
        transform=trans,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    in_poly = mask.astype(bool)
    print(f"in-polygon pixels: {int(in_poly.sum())} / {h*w}")
    print(f"source POP sum inside China polygon (1km mask, all_touched=False): "
          f"{data[in_poly].sum():,.1f}")

    mask_at = rasterize(
        [(geom_union.__geo_interface__, 1)],
        out_shape=(h, w),
        transform=trans,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    in_poly_at = mask_at.astype(bool)
    print(f"source POP sum inside China polygon (1km mask, all_touched=True):  "
          f"{data[in_poly_at].sum():,.1f}")

    # Sample some statistics about the source pixel values
    vals = data[in_poly]
    print(f"per-pixel inside polygon: min={vals.min()} max={vals.max():.1f} "
          f"mean={vals.mean():.4f} median={np.median(vals):.4f}")
    print(f"  nonzero pixels: {int((vals > 0).sum())} ({100*(vals>0).mean():.2f}%)")
