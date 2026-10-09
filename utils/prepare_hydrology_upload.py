"""Build a complete hydrology upload ZIP from an existing GloWPa dataset."""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.windows


VARIABLES = {
    "runoff": "runoff",
    "discharge": "discharge",
    "rdepth": "river_depth",
    "restime": "river_restime",
    "ssrd": "ssrd",
    "triver": "river_temperature",
}
UNITS = {
    "runoff": "mm/day",
    "discharge": "m3/s",
    "river_depth": "m",
    "river_restime": "days",
    "ssrd": "kJ/m2/day",
    "river_temperature": "degC",
    "doc": "mg/L",
    "flowacc": "upstream_cell_count",
    "flowdir": "ESRI_D8",
}


def _common_window(source: rasterio.DatasetReader, bounds: tuple[float, ...]):
    fractional = rasterio.windows.from_bounds(*bounds, transform=source.transform)
    column_start = max(0, int(np.floor(fractional.col_off)))
    row_start = max(0, int(np.floor(fractional.row_off)))
    column_stop = min(source.width, int(np.ceil(fractional.col_off + fractional.width)))
    row_stop = min(source.height, int(np.ceil(fractional.row_off + fractional.height)))
    if column_start >= column_stop or row_start >= row_stop:
        raise ValueError("Study area does not overlap the hydrology grid.")
    return rasterio.windows.Window(
        column_start,
        row_start,
        column_stop - column_start,
        row_stop - row_start,
    )


def build_upload(source_dir: Path, shapefile: Path, output_zip: Path) -> dict:
    source_dir = source_dir.resolve()
    geography = gpd.read_file(shapefile).to_crs(4326)
    if geography.empty:
        raise ValueError("Study-area shapefile is empty.")

    template_path = source_dir / "runoff" / "runoff_m01.tif"
    with rasterio.open(template_path) as template:
        if template.crs is None or template.crs.to_epsg() != 4326:
            raise ValueError("Source hydrology must use EPSG:4326.")
        window = _common_window(template, tuple(geography.total_bounds))
        source_transform = template.transform
        source_shape = template.shape
        transform = template.window_transform(window)
        height, width = int(window.height), int(window.width)

    shapes = [geometry.__geo_interface__ for geometry in geography.geometry]
    inside = rasterio.features.geometry_mask(
        shapes,
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=True,
    )
    source_files = []
    for source_name, target_name in VARIABLES.items():
        source_files.extend(
            (source_dir / source_name / f"{source_name}_m{month:02d}.tif",
             Path(target_name) / f"{target_name}_m{month:02d}.tif")
            for month in range(1, 13)
        )
    source_files.extend([
        (source_dir / "routing" / "flowdir.tif", Path("routing/flowdir.tif")),
        (source_dir / "routing" / "flowacc.tif", Path("routing/flowacc.tif")),
        (source_dir / "doc.tif", Path("doc.tif")),
    ])

    with tempfile.TemporaryDirectory(prefix="hydrology-upload-") as temporary:
        root = Path(temporary) / "hydrology"
        for source_path, relative in source_files:
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing source hydrology raster: {source_path}")
            with rasterio.open(source_path) as source:
                if (
                    source.crs is None
                    or source.crs.to_epsg() != 4326
                    or source.transform != source_transform
                    or source.shape != source_shape
                ):
                    raise ValueError(f"Source grid mismatch: {source_path}")
                values = source.read(1, window=window)
                nodata = source.nodata
                if nodata is None:
                    if np.dtype(source.dtypes[0]).kind == "f":
                        nodata = np.nan
                    else:
                        raise ValueError(f"Integer raster requires nodata: {source_path}")
                values[~inside] = nodata
                profile = source.profile.copy()
                profile.update(
                    width=width,
                    height=height,
                    transform=transform,
                    count=1,
                    compress="deflate",
                )
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(destination, "w", **profile) as output:
                output.write(values, 1)

        metadata = {
            "source": source_dir.name,
            "period": "user-supplied Uganda-wide hydrology",
            "notes": "Clipped to the case-study geometry without resampling.",
            "units": UNITS,
        }
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, Path("hydrology") / path.relative_to(root))

    return {
        "output": str(output_zip),
        "raster_count": len(source_files),
        "resolution": [abs(transform.a), abs(transform.e)],
        "shape": [height, width],
        "bounds": list(rasterio.windows.bounds(window, transform=source_transform)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("shapefile", type=Path)
    parser.add_argument("output_zip", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(build_upload(arguments.source_dir, arguments.shapefile, arguments.output_zip), indent=2))


if __name__ == "__main__":
    main()