"""Extract a sample from a specific year in the NetCDF to verify year selection.

This reads a small lat/lon window from the NetCDF for a given year and prints
the values to verify the band→year mapping is correct.

Usage:
    poetry run python utils/extract_year_sample.py --ssp SSP1 --year 2030
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rasterio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a data sample from NetCDF for a specific year"
    )
    parser.add_argument(
        "--ssp",
        type=str,
        default="SSP1",
        help="SSP scenario (SSP1..SSP5)",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to extract (2006..2100)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Size of the sample window (pixels)",
    )
    args = parser.parse_args()

    if args.year < 2006 or args.year > 2100:
        print("ERROR: Year must be between 2006 and 2100")
        return

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

    # Calculate band index
    band_index = (args.year - 2006) + 1

    print(f"NetCDF: {nc_path.name}")
    print(f"SSP: {args.ssp}")
    print(f"Year: {args.year}")
    print(f"Band index: {band_index}")
    print()

    # Try to find the population subdataset
    with rasterio.open(nc_path) as container:
        subdatasets = list(getattr(container, "subdatasets", []) or [])
        
        if not subdatasets:
            print("No subdatasets; opening as single raster")
            subdatasets = [str(nc_path)]
        else:
            print(f"Found {len(subdatasets)} subdatasets:\n")
            for i, sub_path in enumerate(subdatasets, 1):
                print(f"  [{i}] {sub_path}")
            print()

        # Find the most likely population grid (largest multi-band dataset)
        best_candidate = None
        max_bands = 0
        candidates_info = []

        for sub_path in subdatasets:
            try:
                with rasterio.open(sub_path) as sub:
                    info = {
                        'path': sub_path,
                        'bands': sub.count,
                        'width': sub.width,
                        'height': sub.height,
                        'name': str(sub_path).split(':')[-1] if ':' in str(sub_path) else 'main'
                    }
                    candidates_info.append(info)
                    
                    # Prefer subdatasets with "pop" in the name and multiple bands
                    name_lower = str(sub_path).lower()
                    is_pop = 'pop' in name_lower and 'lat' not in name_lower and 'lon' not in name_lower and 'time' not in name_lower
                    
                    if sub.count > max_bands and sub.width > 1 and sub.height > 1 and is_pop:
                        max_bands = sub.count
                        best_candidate = sub_path
            except Exception as e:
                print(f"  Warning: Could not open {sub_path}: {e}")

        # Fallback: if no "pop" subdataset, just pick the one with most bands
        if not best_candidate:
            for info in candidates_info:
                if info['bands'] > max_bands and info['width'] > 1 and info['height'] > 1:
                    max_bands = info['bands']
                    best_candidate = info['path']

        if not best_candidate:
            print("ERROR: Could not find a suitable population subdataset")
            return

        print(f"Selected subdataset: {best_candidate}")
        print(f"  (has {max_bands} bands)")
        print()

        with rasterio.open(best_candidate) as src:
            if src.count < band_index:
                print(f"ERROR: Dataset only has {src.count} bands, cannot read band {band_index}")
                return

            print(f"Dataset dimensions: {src.width} x {src.height}")
            print(f"Dataset bands: {src.count}")
            print(f"CRS: {src.crs}")
            print(f"NoData value: {src.nodata}")
            print(f"Transform: {src.transform}")
            print()

            # Read multiple samples from different locations
            locations = [
                ("center", src.width // 2, src.height // 2),
                ("upper-left", src.width // 4, src.height // 4),
                ("upper-right", 3 * src.width // 4, src.height // 4),
                ("lower-left", src.width // 4, 3 * src.height // 4),
            ]

            for loc_name, center_x, center_y in locations:
                half_size = args.sample_size // 2

                window = rasterio.windows.Window(
                    col_off=max(0, center_x - half_size),
                    row_off=max(0, center_y - half_size),
                    width=min(args.sample_size, src.width),
                    height=min(args.sample_size, src.height),
                )

                data = src.read(band_index, window=window)
                
                # Get geographic coordinates of the center
                transform = src.window_transform(window)
                center_geo_x, center_geo_y = transform * (window.width / 2, window.height / 2)

                print(f"Sample from {loc_name} (band {band_index}, year {args.year}):")
                print(f"  Window: row {window.row_off}, col {window.col_off}, size {window.width}x{window.height}")
                print(f"  Approx center coords: ({center_geo_x:.2f}, {center_geo_y:.2f})")
                print(f"  Data shape: {data.shape}")
                
                # Filter out nodata
                if src.nodata is not None:
                    valid_data = data[data != src.nodata]
                else:
                    valid_data = data.flatten()
                
                if len(valid_data) > 0:
                    print(f"  Statistics (valid pixels only):")
                    print(f"    Min: {valid_data.min():.2f}")
                    print(f"    Max: {valid_data.max():.2f}")
                    print(f"    Mean: {valid_data.mean():.2f}")
                    print(f"    Valid pixels: {len(valid_data)} / {data.size}")
                else:
                    print(f"  ⚠ All pixels are NoData!")
                
                print(f"  Sample values (first 3 rows):")
                for i in range(min(3, data.shape[0])):
                    print(f"    {data[i]}")
                print()


if __name__ == "__main__":
    main()
