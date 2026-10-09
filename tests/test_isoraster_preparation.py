from pathlib import Path
import json

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


def write_raster_at(path, values, resolution, left, top):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", count=1, dtype="float32",
                       width=values.shape[1], height=values.shape[0], crs="EPSG:4326",
                       transform=from_origin(left, top, resolution, resolution), nodata=np.nan) as target:
        target.write(values, 1)


def read(path):
    with rasterio.open(path) as source:
        return source.read(1)


def coastal_case(tmp_path):
    """Two zones on a 0.25-degree grid; the top-right 0.5-degree hydrology cell is sea."""
    zones = np.ones((4, 4))
    zones[:, 2:] = 2
    for name in ("human_emissions/isoraster.tif", "livestock_emissions/animal_isoraster.tif"):
        write_raster(tmp_path / name, zones, 0.25)
    write_raster(tmp_path / "human_emissions/pop_urban.tif", np.full((4, 4), 10), 0.25)
    write_raster(tmp_path / "livestock_emissions/animals/ducks_heads.tif", np.full((4, 4), 3), 0.25)
    routing = np.zeros((2, 2))
    routing[0, 1] = np.nan
    write_raster(tmp_path / "hydrology/routing/flowdir.tif", routing, 0.5)
    write_raster(tmp_path / "hydrology/routing/flowacc.tif", np.where(np.isnan(routing), np.nan, 1), 0.5)
    (tmp_path / "hydrology/source.json").write_text(json.dumps({"source": "native_generated"}))


def test_custom_hydrology_does_not_modify_emission_cells(tmp_path):
    coastal_case(tmp_path)
    (tmp_path / "hydrology/source.json").write_text(json.dumps({"source": "custom_uploaded"}))
    iso = tmp_path / "human_emissions/isoraster.tif"
    original = iso.read_bytes()
    align_model_grid(tmp_path)
    validate_model_grid(tmp_path)
    assert iso.read_bytes() == original


def test_emission_cells_in_unrouted_hydrology_cells_are_rejected(tmp_path):
    coastal_case(tmp_path)
    with pytest.raises(ValueError, match="without routing"):
        validate_model_grid(tmp_path)


def test_unrouted_coastal_cells_are_excluded_and_zone_totals_conserved(tmp_path):
    coastal_case(tmp_path)
    align_model_grid(tmp_path)
    validate_model_grid(tmp_path)
    for name in ("human_emissions/isoraster.tif", "livestock_emissions/animal_isoraster.tif"):
        zones = read(tmp_path / name)
        assert np.isnan(zones[:2, 2:]).all()
        assert (zones[2:, 2:] == 2).all() and (zones[:, :2] == 1).all()
    population = read(tmp_path / "human_emissions/pop_urban.tif")
    ducks = read(tmp_path / "livestock_emissions/animals/ducks_heads.tif")
    assert np.isnan(population[:2, 2:]).all() and np.isnan(ducks[:2, 2:]).all()
    assert np.nansum(population[:, 2:]) == pytest.approx(80)
    assert np.nansum(population[:, :2]) == pytest.approx(80)
    assert np.nansum(ducks) == pytest.approx(48)


def test_emission_cells_outside_hydrology_extent_are_excluded(tmp_path):
    write_raster(tmp_path / "human_emissions/isoraster.tif", np.ones((4, 4)), 0.25)
    write_raster(tmp_path / "human_emissions/pop_urban.tif", np.full((4, 4), 10), 0.25)
    # Hydrology starts at x=0.5: the western emission cells only touch or miss it.
    write_raster_at(tmp_path / "hydrology/routing/flowdir.tif", np.zeros((2, 2)), 0.5, 0.5, 1)
    align_model_grid(tmp_path)
    validate_model_grid(tmp_path)
    assert np.isnan(read(tmp_path / "human_emissions/isoraster.tif")[:, :2]).all()
    assert np.nansum(read(tmp_path / "human_emissions/pop_urban.tif")) == pytest.approx(160)


def test_zone_entirely_in_unrouted_cells_fails_without_changing_inputs(tmp_path):
    coastal_case(tmp_path)
    zones = np.ones((4, 4))
    zones[:2, 2:] = 2
    iso = tmp_path / "human_emissions/isoraster.tif"
    write_raster(iso, zones, 0.25)
    write_raster(tmp_path / "livestock_emissions/animal_isoraster.tif", zones, 0.25)
    original = iso.read_bytes()
    with pytest.raises(ValueError, match="Emission zone\\(s\\) 2 lie entirely"):
        align_model_grid(tmp_path)
    assert iso.read_bytes() == original