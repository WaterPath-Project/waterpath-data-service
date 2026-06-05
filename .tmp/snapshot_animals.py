"""Compare duck head totals across sessions, snapshotting the current state."""
from pathlib import Path
import sys
import json
import numpy as np
import rasterio

SESSIONS = sys.argv[1:] or ["chn", "uganda"]
ROOT = Path("/work/waterpath_data_service/data")
out = {}
for s in SESSIONS:
    info = {}
    animals_dir = ROOT / s / "baseline" / "livestock_emissions" / "animals"
    if not animals_dir.is_dir():
        info["error"] = f"no animals dir: {animals_dir}"
        out[s] = info
        continue
    for tif in sorted(animals_dir.glob("*_heads.tif")):
        with rasterio.open(tif) as ds:
            a = ds.read(1).astype("float64")
            total = float(np.nansum(a))
        info[tif.stem] = total
    out[s] = info
print(json.dumps(out, indent=2))
