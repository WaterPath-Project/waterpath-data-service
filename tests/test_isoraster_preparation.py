from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from waterpath_data_service.services.model_grid import align_model_grid, validate_model_grid


def write_raster(path, values, resolution):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", count=1, dtype="float32",
                       width=values.shape[1], height=values.shape[0], crs="EPSG:4326",
                       transform=from_origin(0, 1, resolution, resolution), nodata=np.nan) as target:
        target.write(values, 1)


def test_isoraster_unchanged_when_livestock_needs_resampling(tmp_path):
    iso = tmp_path / "human_emissions/isoraster.tif"
    heads = tmp_path / "livestock_emissions/animals/cattle_heads.tif"
    write_raster(iso, np.ones((4, 4)), 0.25)
    write_raster(heads, np.full((2, 2), 10), 0.5)
    original = iso.read_bytes()
    align_model_grid(tmp_path)
    assert iso.read_bytes() == original
    validate_model_grid(tmp_path)
    with rasterio.open(heads) as source:
        assert source.shape == (4, 4)
        assert np.nansum(source.read(1)) == pytest.approx(40)


def test_incompatible_routing_is_preserved_without_coarsening_isoraster(tmp_path):
    iso = tmp_path / "human_emissions/isoraster.tif"
    routing = tmp_path / "hydrology/routing/flowdir.tif"
    write_raster(iso, np.ones((4, 4)), 0.25)
    write_raster(routing, np.zeros((2, 2)), 0.5)
    original = {path: path.read_bytes() for path in (iso, routing)}
    align_model_grid(tmp_path)
    validate_model_grid(tmp_path)
    assert all(path.read_bytes() == contents for path, contents in original.items())


def test_livestock_crop_preserves_native_hydrology(tmp_path):
    iso = tmp_path / "human_emissions/isoraster.tif"
    routing = tmp_path / "hydrology/routing/flowdir.tif"
    heads = tmp_path / "livestock_emissions/animals/cattle_heads.tif"
    write_raster(iso, np.ones((2, 2)), 0.25)
    write_raster(routing, np.zeros((4, 4)), 0.25)
    write_raster(heads, np.full((4, 4), 10), 0.25)
    original_routing = routing.read_bytes()
    original_iso = iso.read_bytes()
    align_model_grid(tmp_path)
    assert iso.read_bytes() == original_iso
    with rasterio.open(heads) as source:
        assert np.nansum(source.read(1)) == pytest.approx(40)
    assert routing.read_bytes() == original_routing
    assert not (tmp_path / "hydrology/native_source.zip").exists()
    validate_model_grid(tmp_path)


def test_routing_cycle_rejected_before_any_resampling_is_promoted(tmp_path):
    iso = tmp_path / "human_emissions/isoraster.tif"
    routing = tmp_path / "hydrology/routing/flowdir.tif"
    heads = tmp_path / "livestock_emissions/animals/cattle_heads.tif"
    write_raster(iso, np.ones((4, 4)), 0.25)
    values = np.zeros((4, 4))
    values[0, :2] = [1, 16]
    write_raster(routing, values, 0.25)
    write_raster(heads, np.ones((2, 2)), 0.5)
    original = heads.read_bytes()
    with pytest.raises(ValueError, match="cycle"):
        align_model_grid(tmp_path)
    assert heads.read_bytes() == original