"""Inspect NetCDF population files to verify structure, years, and subdatasets.

Usage:
    poetry run python utils/inspect_netcdf.py [--ssp SSP1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rasterio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect NetCDF population files"
    )
    parser.add_argument(
        "--ssp",
        type=str,
        default="SSP1",
        help="SSP scenario (SSP1..SSP5)",
    )
    args = parser.parse_args()

    ssp_lower = args.ssp.lower()
    project_root = Path(__file__).parent.parent
    nc_path = (
        project_root
        / "waterpath_data_service"
        / "static"
        / "data"
        / f"population_{ssp_lower}soc_2.5min_annual_2006-2100.nc4"
    )

    if not nc_path.is_file():
        print(f"ERROR: NetCDF file not found: {nc_path}")
        return

    print(f"NetCDF File: {nc_path}")
    print(f"Size: {nc_path.stat().st_size / 1024 / 1024:.1f} MB\n")

    # Open the container and list subdatasets
    with rasterio.open(nc_path) as container:
        print(f"Container CRS: {container.crs}")
        print(f"Container dimensions: {container.width} x {container.height}")
        print(f"Container band count: {container.count}")

        subdatasets = list(getattr(container, "subdatasets", []) or [])
        
        if not subdatasets:
            print("\nNo subdatasets found (single-layer NetCDF)")
            print(f"Bounds: {container.bounds}")
            print(f"Transform: {container.transform}")
            return

        print(f"\nFound {len(subdatasets)} subdatasets:\n")
        
        for i, sub_path in enumerate(subdatasets, 1):
            print(f"[{i}] {sub_path}")
            
            try:
                with rasterio.open(sub_path) as sub:
                    print(f"    CRS: {sub.crs}")
                    print(f"    Dimensions: {sub.width} x {sub.height}")
                    print(f"    Bands: {sub.count}")
                    print(f"    Bounds: {sub.bounds}")
                    print(f"    NoData: {sub.nodata}")
                    
                    # If this looks like the population grid (multi-band), show year mapping
                    if sub.count > 1:
                        print(f"\n    Band → Year mapping (assuming 2006..2100):")
                        # Show first 5, middle, and last 5
                        for band_idx in [1, 2, 3, 4, 5]:
                            year = 2006 + (band_idx - 1)
                            print(f"        Band {band_idx:3d} → Year {year}")
                        
                        if sub.count > 10:
                            print(f"        ...")
                            mid_band = sub.count // 2
                            mid_year = 2006 + (mid_band - 1)
                            print(f"        Band {mid_band:3d} → Year {mid_year}")
                            print(f"        ...")
                        
                        for band_idx in range(max(sub.count - 4, 6), sub.count + 1):
                            year = 2006 + (band_idx - 1)
                            print(f"        Band {band_idx:3d} → Year {year}")
                    
                    print()
            except Exception as e:
                print(f"    ERROR opening subdataset: {e}\n")


if __name__ == "__main__":
    main()
