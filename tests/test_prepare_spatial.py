from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from shapely.geometry import box

from waterpath_data_service.services import prepare_spatial
from waterpath_data_service.services.prepare_spatial import (
    _auto_population_resolution,
    _resample_pop_raster,
    prepare_spatial_inputs,
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


def test_spatial_inputs_preserve_template_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_path = tmp_path / "template.tif"
    template_transform = Affine(0.25, 0.0, 31.125, 0.0, -0.25, 0.9)
    with rasterio.open(
        template_path,
        "w",
        driver="GTiff",
        width=5,
        height=4,
        count=1,
        dtype="int32",
        crs="EPSG:4326",
        transform=template_transform,
        nodata=0,
    ) as template:
        template.write(np.ones((4, 5), dtype=np.int32), 1)

    population_path = tmp_path / "population.tif"
    with rasterio.open(
        population_path,
        "w",
        driver="GTiff",
        width=5,
        height=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=template_transform,
        nodata=-99999.0,
    ) as population:
        population.write(np.full((4, 5), 10.0, dtype=np.float32), 1)

    isodata_path = tmp_path / "isodata.csv"
    pd.DataFrame(
        {
            "gid": ["TST"],
            "iso": [1],
            "population": [200],
            "fraction_urban_pop": [0.5],
        }
    ).to_csv(isodata_path, index=False)
    features = pd.DataFrame({"GID_0": ["TST"]})
    features["geometry"] = None
    features.at[0, "geometry"] = box(31.125, -0.1, 32.0, 0.9)
    features.__dict__["total_bounds"] = np.array([31.125, -0.1, 32.0, 0.9])
    monkeypatch.setattr(prepare_spatial.pyogrio, "read_dataframe", lambda _: features)

    paths = prepare_spatial_inputs(
        geodata_path=str(tmp_path / "geodata.shp"),
        isodata_path=str(isodata_path),
        pop_raster_path=str(population_path),
        out_dir=str(tmp_path / "output"),
        template_raster_path=str(template_path),
    )

    for output_path in paths.values():
        with rasterio.open(output_path) as output:
            assert output.shape == (4, 5)
            assert output.transform == template_transform
            assert output.crs == rasterio.crs.CRS.from_epsg(4326)


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
