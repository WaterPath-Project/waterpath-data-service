"""Verify pop_urban + pop_rural totals vs CSV for newly regenerated sessions."""
from pathlib import Path
import pandas as pd
import rasterio
import numpy as np

SESSIONS = ["kampala_level_4", "uganda"]
ROOT = Path("/work/waterpath_data_service/data")
NODATA = -9999.0

for s in SESSIONS:
    base = ROOT / s / "baseline"
    csv = base / "human_emissions" / "population.csv"
    df = pd.read_csv(csv)
    csv_total = float(df["population"].sum())
    sums = {}
    for name in ("pop_urban", "pop_rural"):
        with rasterio.open(base / "human_emissions" / f"{name}.tif") as ds:
            a = ds.read(1).astype("float64")
            a = np.where((a == NODATA) | ~np.isfinite(a), 0.0, a)
            sums[name] = float(a.sum())
    total = sums["pop_urban"] + sums["pop_rural"]
    diff = total - csv_total
    pct = 100.0 * diff / csv_total if csv_total else float("nan")
    print(f"=== {s} ===")
    print(f"  CSV pop total : {csv_total:>20,.1f}")
    print(f"  pop_urban     : {sums['pop_urban']:>20,.1f}")
    print(f"  pop_rural     : {sums['pop_rural']:>20,.1f}")
    print(f"  urban+rural   : {total:>20,.1f}")
    print(f"  diff vs CSV   : {diff:>20,.1f}  ({pct:+.2f}%)")
    print()
