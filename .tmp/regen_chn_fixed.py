"""Regenerate CHN baseline rasters via prepare_spatial_inputs and verify totals."""
import sys
from pathlib import Path
import shutil
import numpy as np
import rasterio

sys.path.insert(0, "/work")
from waterpath_data_service.services.prepare_spatial import prepare_spatial_inputs

SHP = "waterpath_data_service/data/chn/baseline/geodata/geodata.shp"
CSV = "waterpath_data_service/data/chn/baseline/human_emissions/population.csv"
SRC = "waterpath_data_service/static/data/worldpop_2025/FuturePop_SSP1_2025_1km_v0_2.tif"
OUT = Path("/tmp/chn_out_fixed")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

paths = prepare_spatial_inputs(
    geodata_path=SHP,
    isodata_path=CSV,
    pop_raster_path=SRC,
    out_dir=str(OUT),
)
print("Wrote:", paths)

def s(p):
    with rasterio.open(p) as r:
        a = r.read(1).astype("float64"); nd = r.nodata
        m = np.isfinite(a)
        if nd is not None:
            m &= ~np.isclose(a, nd)
        return float(a[m].sum()), int(m.sum()), float(a[m].min() if m.any() else 0), float(a[m].max() if m.any() else 0)

iso = paths["isoraster"]; urb = paths["pop_urban"]; rur = paths["pop_rural"]
us, un, umn, umx = s(urb); rs, rn, rmn, rmx = s(rur)
import pandas as pd
csv_total = float(pd.read_csv(CSV)["population"].sum())
print(f"\nCSV total population : {csv_total:>20,.1f}")
print(f"pop_urban  sum       : {us:>20,.1f}   (n_valid={un}, min={umn}, max={umx:.1f})")
print(f"pop_rural  sum       : {rs:>20,.1f}   (n_valid={rn}, min={rmn}, max={rmx:.1f})")
print(f"urban + rural        : {us+rs:>20,.1f}")
print(f"Diff vs CSV          : {us+rs - csv_total:>20,.1f}  "
      f"({100*(us+rs-csv_total)/csv_total:+.2f}%)")

with rasterio.open(iso) as r:
    iso_arr = r.read(1)
print(f"\nisoraster: shape={iso_arr.shape}, domain pixels={int((iso_arr>0).sum())}")
