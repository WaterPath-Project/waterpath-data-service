import rasterio
import numpy as np
from pathlib import Path
import sys

sys.stdout = open(".tmp/pop_tif_stats.txt", "w")

files = [
    "waterpath_data_service/data/test/baseline/human_emissions/pop_urban.tif",
    "waterpath_data_service/data/test/baseline/human_emissions/pop_rural.tif",
]

for fname in files:
    f = Path(fname)
    with rasterio.open(f) as src:
        arr = src.read(1)
        nd = src.nodata
        valid = arr[arr != nd] if nd is not None else arr
        print(f"File: {f.name}")
        print(f"  CRS: {src.crs}")
        print(f"  Shape: {arr.shape}")
        print(f"  Bounds: {src.bounds}")
        print(f"  Nodata: {nd}")
        print(f"  Valid pixels: {valid.size}")
        print(f"  Min (valid): {valid.min():.4f}")
        print(f"  Max (valid): {valid.max():.4f}")
        print(f"  Sum (valid): {valid.sum():.2f}")
        print(f"  Mean (valid): {valid.mean():.4f}")
        print(f"  Nonzero valid: {np.count_nonzero(valid)}")
        print(f"  First 10 unique vals: {np.unique(valid)[:10]}")
        print()

# Also check if the two arrays are identical
with rasterio.open(files[0]) as s1:
    a1 = s1.read(1)
with rasterio.open(files[1]) as s2:
    a2 = s2.read(1)

print(f"Arrays identical: {np.array_equal(a1, a2)}")
print(f"Max absolute difference: {np.abs(a1.astype(float) - a2.astype(float)).max():.6f}")
print(f"Sum urban: {a1.sum():.2f}, Sum rural: {a2.sum():.2f}")
print(f"Sum urban + rural: {a1.sum() + a2.sum():.2f}")
sys.stdout.close()
