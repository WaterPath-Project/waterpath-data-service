"""Verify regenerated CHN and test_dhaka population rasters."""
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

for session in ["chn", "test_dhaka"]:
    base = Path(f"waterpath_data_service/data/{session}/baseline/human_emissions")
    csv_total = float(pd.read_csv(base / "population.csv")["population"].sum())
    sums = {}
    for name in ["pop_urban.tif", "pop_rural.tif"]:
        with rasterio.open(base / name) as src:
            a = src.read(1).astype("float64"); nd = src.nodata
            m = np.isfinite(a)
            if nd is not None:
                m &= ~np.isclose(a, nd)
            sums[name] = float(a[m].sum())
    tot = sums["pop_urban.tif"] + sums["pop_rural.tif"]
    print(f"=== {session} ===")
    print(f"  CSV pop total : {csv_total:>20,.1f}")
    print(f"  pop_urban     : {sums['pop_urban.tif']:>20,.1f}")
    print(f"  pop_rural     : {sums['pop_rural.tif']:>20,.1f}")
    print(f"  urban+rural   : {tot:>20,.1f}")
    print(f"  diff vs CSV   : {tot-csv_total:>20,.1f}  ({100*(tot-csv_total)/csv_total:+.2f}%)")
    print()
