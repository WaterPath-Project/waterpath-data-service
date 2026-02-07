"""Verify that raster clipping was performed correctly.

Compares the shapefile boundary with the clipped GeoTIFF and optionally
visualizes them together.

Usage:
    poetry run python utils/verify_clip.py <session_id> [--visualize]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.plot import show


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify raster clipping against shapefile"
    )
    parser.add_argument(
        "session_id",
        type=str,
        help="Session ID to verify",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visual plot (requires matplotlib with GUI backend)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario folder name (e.g., SSP1_2030). If not provided, searches all scenarios.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    session_dir = project_root / "waterpath_data_service" / "data" / args.session_id

    if not session_dir.is_dir():
        print(f"ERROR: Session directory not found: {session_dir}")
        return

    # Find shapefile
    shp_candidates = [
        session_dir / "default" / "geodata" / "geodata.shp",
        session_dir / "geodata" / "geodata.shp",
    ]
    shp_path = None
    for candidate in shp_candidates:
        if candidate.is_file():
            shp_path = candidate
            break

    if not shp_path:
        print(f"ERROR: Shapefile not found in session {args.session_id}")
        return

    print(f"Shapefile: {shp_path}")

    # Load shapefile
    gdf = gpd.read_file(shp_path)
    print(f"Shapefile CRS: {gdf.crs}")
    print(f"Shapefile bounds: {gdf.total_bounds}")
    print(f"Shapefile features: {len(gdf)}")
    print()

    # Find isoraster.tif
    tif_paths = []
    if args.scenario:
        scenario_dir = session_dir / "scenarios" / args.scenario
        tif_candidate = scenario_dir / "isoraster.tif"
        if tif_candidate.is_file():
            tif_paths.append(tif_candidate)
    else:
        # Search all scenario folders
        scenarios_dir = session_dir / "scenarios"
        if scenarios_dir.is_dir():
            for scenario_folder in scenarios_dir.iterdir():
                if scenario_folder.is_dir():
                    tif_candidate = scenario_folder / "isoraster.tif"
                    if tif_candidate.is_file():
                        tif_paths.append(tif_candidate)

    if not tif_paths:
        print(f"ERROR: No isoraster.tif files found for session {args.session_id}")
        return

    for tif_path in tif_paths:
        print(f"\n{'=' * 80}")
        print(f"Clipped Raster: {tif_path.relative_to(project_root)}")
        print('=' * 80)

        with rasterio.open(tif_path) as src:
            print(f"Raster CRS: {src.crs}")
            print(f"Raster bounds: {src.bounds}")
            print(f"Raster shape: {src.shape} (height x width)")
            print(f"Raster resolution: {abs(src.transform.a):.6f} x {abs(src.transform.e):.6f}")
            print(f"NoData value: {src.nodata}")

            data = src.read(1)
            valid_data = data[data != src.nodata] if src.nodata is not None else data
            
            if len(valid_data) > 0:
                print(f"\nData statistics:")
                print(f"  Min: {valid_data.min():.2f}")
                print(f"  Max: {valid_data.max():.2f}")
                print(f"  Mean: {valid_data.mean():.2f}")
                print(f"  Valid pixels: {len(valid_data)} / {data.size} ({100 * len(valid_data) / data.size:.1f}%)")
            else:
                print("\nWARNING: No valid data pixels found!")

            # Check CRS alignment
            print(f"\nCRS Comparison:")
            if gdf.crs == src.crs:
                print(f"  ✓ Shapefile and raster have matching CRS")
            else:
                print(f"  ⚠ CRS mismatch!")
                print(f"    Shapefile: {gdf.crs}")
                print(f"    Raster: {src.crs}")

            # Check bounds overlap
            gdf_reproj = gdf.to_crs(src.crs) if gdf.crs != src.crs else gdf
            shp_bounds = gdf_reproj.total_bounds  # [minx, miny, maxx, maxy]
            raster_bounds = src.bounds  # BoundingBox(left, bottom, right, top)

            print(f"\nBounds Comparison (in raster CRS):")
            print(f"  Shapefile: [{shp_bounds[0]:.4f}, {shp_bounds[1]:.4f}, {shp_bounds[2]:.4f}, {shp_bounds[3]:.4f}]")
            print(f"  Raster:    [{raster_bounds.left:.4f}, {raster_bounds.bottom:.4f}, {raster_bounds.right:.4f}, {raster_bounds.top:.4f}]")

            # Check if raster bounds are contained within shapefile (with some tolerance)
            tolerance = max(abs(src.transform.a), abs(src.transform.e)) * 2  # 2 pixels
            if (raster_bounds.left >= shp_bounds[0] - tolerance and
                raster_bounds.bottom >= shp_bounds[1] - tolerance and
                raster_bounds.right <= shp_bounds[2] + tolerance and
                raster_bounds.top <= shp_bounds[3] + tolerance):
                print(f"  ✓ Raster bounds are within shapefile bounds (±{tolerance:.6f})")
            else:
                print(f"  ⚠ Raster bounds may extend outside shapefile!")

            if args.visualize:
                try:
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Show raster
                    show(src, ax=ax, alpha=0.7, cmap='viridis', title=f"Clipped Raster + Shapefile\n{tif_path.name}")
                    
                    # Overlay shapefile boundary
                    gdf_reproj.boundary.plot(ax=ax, color='red', linewidth=2, label='Shapefile boundary')
                    
                    ax.legend()
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"\nERROR visualizing: {e}")
                    print("(Try running without --visualize or ensure matplotlib has a display backend)")


if __name__ == "__main__":
    main()
