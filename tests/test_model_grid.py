from pathlib import Path
import json
from unittest.mock import AsyncMock

import io
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from waterpath_data_service.services.model_grid import align_model_grid, validate_model_grid
from waterpath_data_service.web.api.data import views


def write_raster(path: Path, values, resolution):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.array(values, dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
                       count=1, dtype="float32", crs="EPSG:4326", nodata=np.nan,
                       transform=from_origin(0, 1, resolution, resolution)) as source:
        source.write(values, 1)


@pytest.mark.parametrize("resolution,size", [(0.5, 2), (0.125, 8)])
def test_alignment_preserves_isoraster_categories_and_counts(tmp_path, resolution, size):
    hydrology = tmp_path / "hydrology/routing/flowdir.tif"
    write_raster(hydrology, np.zeros((4, 4)), 0.25)
    iso = tmp_path / "human_emissions/isoraster.tif"
    population = tmp_path / "human_emissions/pop_urban.tif"
    write_raster(iso, np.full((4, 4), 7), 0.25)
    original = iso.read_bytes()
    write_raster(population, np.full((size, size), 120 / size ** 2), resolution)
    with pytest.raises(ValueError, match="do not match"):
        validate_model_grid(tmp_path)

    align_model_grid(tmp_path)

    validate_model_grid(tmp_path)
    assert iso.read_bytes() == original
    with rasterio.open(iso) as source:
        assert source.shape == (4, 4)
        assert set(np.unique(source.read(1))) == {7}
    with rasterio.open(population) as source:
        assert np.nansum(source.read(1)) == pytest.approx(120)


@pytest.mark.anyio
@pytest.mark.parametrize("custom", [False, True])
async def test_generation_returns_one_grid_for_all_modules(client, tmp_path, monkeypatch, custom):
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    session = tmp_path / "case"
    baseline = session / "baseline"
    geography = baseline / "geodata/geodata.shp"
    geography.parent.mkdir(parents=True)
    geography.touch()
    hydrology = baseline / "hydrology/routing/flowdir.tif"
    if custom:
        write_raster(hydrology, np.zeros((4, 4)), 0.25)
        (hydrology.parent.parent / "source.json").write_text(json.dumps({"source": "custom_uploaded"}))
    monkeypatch.setattr(views, "generateData", AsyncMock(return_value={
        "population": "iso,gid,population,fraction_urban_pop\n1,UGA,160,0.5\n",
        "sanitation": "gid\nUGA\n",
        "treatment": "gid\nUGA\n",
    }))
    monkeypatch.setattr(views, "fetch_assumptions", AsyncMock(return_value=[]))

    def spatial(**kwargs):
        output = Path(kwargs["out_dir"])
        write_raster(output / "isoraster.tif", np.ones((4, 4)), 0.25)
        write_raster(output / "pop_urban.tif", np.full((4, 4), 5), 0.25)
        write_raster(output / "pop_rural.tif", np.full((4, 4), 5), 0.25)

    def hydro(**kwargs):
        write_raster(hydrology, np.zeros((4, 4)), 0.25)

    def livestock(**kwargs):
        write_raster(baseline / "livestock_emissions/animal_isoraster.tif", np.ones((4, 4)), 0.25)
        write_raster(baseline / "livestock_emissions/animals/cattle_heads.tif", np.ones((4, 4)), 0.25)
        return {"status": "written"}

    def qmra(**kwargs):
        write_raster(Path(kwargs["out_dir"]) / "treatment.tif", np.full((4, 4), 2), 0.25)

    monkeypatch.setattr(views, "prepare_spatial_inputs", spatial)
    monkeypatch.setattr(views, "generate_hydrology_inputs", hydro)
    monkeypatch.setattr(views, "generate_livestock_tabular_inputs", livestock)
    monkeypatch.setattr(views, "generate_qmra_inputs", qmra)
    monkeypatch.setattr(views, "_generate_baseline_livestock_temperature", lambda **kwargs: None)
    response = await client.post("/api/data/input/generate", params={
        "session_id": "case", "gids": "UGA", "include_livestock": True,
        "include_hydrology": True, "include_qmra": True,
    })

    assert response.status_code == 200, response.text
    validate_model_grid(baseline)
    with rasterio.open(baseline / "human_emissions/isoraster.tif") as source:
        assert source.shape == (4, 4)
    assert (baseline / "qmra/treatment.tif").is_file()


def test_small_area_not_at_coarse_cell_centre_remains_represented(tmp_path):
    write_raster(tmp_path / "hydrology/routing/flowdir.tif", np.zeros((4, 4)), 0.25)
    values = np.full((4, 4), np.nan)
    values[0, 0] = 7
    path = tmp_path / "human_emissions/isoraster.tif"
    write_raster(path, values, 0.25)
    align_model_grid(tmp_path)
    with rasterio.open(path) as source:
        assert source.read(1)[0, 0] == 7


@pytest.mark.anyio
@pytest.mark.parametrize("archive_type", ["full", "cached", "scenario"])
async def test_download_exports_mixed_grids_unchanged(client, tmp_path, monkeypatch, archive_type):
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(views, "ensure_human_emissions_csv", lambda session_id: None)
    session = tmp_path / "case"
    baseline = session / "baseline"
    scenario = session / "scenarios/SSP1_2050"
    rasters = [
        baseline / "hydrology/routing/flowdir.tif",
        baseline / "human_emissions/isoraster.tif",
        scenario / "hydrology/routing/flowdir.tif",
        scenario / "isoraster.tif",
    ]
    for path in rasters:
        if path.name == "flowdir.tif":
            write_raster(path, np.zeros((2, 2)), 0.5)
        else:
            write_raster(path, np.ones((4, 4)), 0.25)
    (baseline / "human_emissions/treatment.csv").write_text("iso,value\n1,0.5\n")
    originals = {path: path.read_bytes() for path in rasters}
    if archive_type == "cached":
        cache = views._input_download_cache_path("case")
        cache.parent.mkdir()
        with zipfile.ZipFile(cache, "w") as archive:
            for path in rasters:
                archive.write(path, path.relative_to(session).as_posix())
    params = {"session_id": "case"}
    if archive_type == "scenario":
        params.update(scenario="SSP1", year=2050)
    response = await client.get("/api/data/input/download", params=params)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    if archive_type == "cached":
        assert response.content == cache.read_bytes()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for path, contents in originals.items():
            assert path.read_bytes() == contents
            if archive_type != "scenario" or path.is_relative_to(scenario):
                assert archive.read(path.relative_to(session).as_posix()) == contents