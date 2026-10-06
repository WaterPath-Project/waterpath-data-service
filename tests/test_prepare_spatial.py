from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from waterpath_data_service.services.prepare_spatial import (
    _auto_population_resolution,
    _resample_pop_raster,
)


def test_population_resolution_preserves_regional_spatial_detail() -> None:
    resolution = _auto_population_resolution(
        extent_x=13.75,
        extent_y=17.5,
        source_native_resolution=1 / 120,
    )

    assert resolution == 0.1


def test_population_resolution_retains_native_floor_and_global_cap() -> None:
    assert _auto_population_resolution(0.2, 0.2, 1 / 120) == 0.01
    assert _auto_population_resolution(360, 144, 1 / 120) == 0.5


def test_population_resampling_does_not_inflate_partial_nodata_cells(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "population.tif"
    nodata = -99999.0
    source = np.array(
        [
            [100.0, nodata],
            [100.0, nodata],
        ],
        dtype=np.float32,
    )
    source_transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 2.0)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=source_transform,
        nodata=nodata,
    ) as dst:
        dst.write(source, 1)

    result = _resample_pop_raster(
        str(source_path),
        Affine(2.0, 0.0, 0.0, 0.0, -2.0, 2.0),
        dst_height=1,
        dst_width=1,
        crs="EPSG:4326",
    )

    assert result[0, 0] == pytest.approx(200.0, rel=2e-4)
