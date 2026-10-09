import io
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from fastapi import HTTPException
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box

from waterpath_data_service.services import hydrology, hydrology_upload
from waterpath_data_service.web.api.data import views


def _tiff(value=2, resolution=0.25, cycle=False):
    values = np.full((4, 4), value, dtype="float32")
    if cycle:
        values[0, :2] = [1, 16]
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", height=4, width=4, count=1,
                         dtype="float32", crs="EPSG:4326", nodata=-9999,
                         transform=from_origin(0, 1, resolution, resolution)) as raster:
            raster.write(values, 1)
        return memory.read()


def _archive(value=2, omit=None, extra=None, bad_grid=False, cycle=False):
    payload = io.BytesIO()
    data = _tiff(value)
    direction = _tiff(0, cycle=cycle)
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(hydrology_upload.REQUIRED_FILES):
            if relative == omit:
                continue
            content = direction if relative == "routing/flowdir.tif" else data
            if bad_grid and relative == "runoff/runoff_m02.tif":
                content = _tiff(value, resolution=0.5)
            archive.writestr("hydrology/" + relative, content)
        if extra:
            archive.writestr(extra, b"unexpected")
    return payload.getvalue()


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    session = tmp_path / "case"
    geography = session / "baseline/geodata/geodata.shp"
    geography.parent.mkdir(parents=True)
    geography.touch()
    (session / "scenarios/SSP3_2050").mkdir(parents=True)
    monkeypatch.setattr(
        hydrology_upload.gpd, "read_file",
        lambda _: gpd.GeoDataFrame(geometry=[box(0.01, 0.01, 0.99, 0.99)], crs=4326),
    )
    return session


async def _upload(client, payload, **params):
    return await client.post(
        "/api/data/input/upload",
        params={"session_id": "case", "file_id": "hydrology", **params},
        files={"file": ("hydrology.zip", payload, "application/zip")},
    )


@pytest.mark.anyio
@pytest.mark.parametrize("scenario", [False, True])
async def test_complete_upload_preserves_rasters_and_replaces_only_hydrology(client, session, scenario):
    target = session / ("scenarios/SSP3_2050" if scenario else "baseline")
    (target / "hydrology").mkdir()
    (target / "hydrology/stale.tif").write_bytes(b"old")
    (target / "unrelated.csv").write_text("untouched")
    cache = views._input_download_cache_path("case")
    cache.parent.mkdir()
    cache.write_bytes(b"stale archive")
    payload = _archive()
    params = {"ssp": "ssp3", "year": 2050} if scenario else {}

    response = await _upload(client, payload, **params)

    assert response.status_code == 200, response.text
    assert response.json()["source"] == "custom_uploaded"
    assert response.json()["raster_count"] == 75
    assert not cache.exists()
    assert not (target / "hydrology/stale.tif").exists()
    assert (target / "unrelated.csv").read_text() == "untouched"
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name in archive.namelist():
            assert (target / name).read_bytes() == archive.read(name)
    assert hydrology_upload.source_info(target / "hydrology")["grid"]["resolution"] == [0.25, 0.25]
    if scenario:
        download = await client.get("/api/data/input/download", params={
            "session_id": "case", "scenario": "SSP3", "year": 2050,
        })
        assert download.status_code == 200
        with zipfile.ZipFile(io.BytesIO(download.content)) as archive:
            assert archive.read("scenarios/SSP3_2050/hydrology/runoff/runoff_m01.tif") == _tiff(2)


@pytest.mark.anyio
@pytest.mark.parametrize("options", [
    {"omit": "runoff/runoff_m12.tif"}, {"extra": "../escape.tif"},
    {"extra": "hydrology/source.json"}, {"extra": "hydrology/RUNOFF/runoff_m01.tif"},
    {"bad_grid": True}, {"cycle": True}, {"value": -2},
])
async def test_invalid_upload_preserves_previous_dataset(client, session, options):
    old = session / "baseline/hydrology/previous.txt"
    old.parent.mkdir()
    old.write_bytes(b"last good")
    response = await _upload(client, _archive(**options))
    assert response.status_code == 422, response.text
    assert old.read_bytes() == b"last good"
    assert not list((session / "baseline").glob(".hydrology-*"))


@pytest.mark.anyio
async def test_zip_type_and_corruption_errors(client, session):
    response = await client.post(
        "/api/data/input/upload", params={"session_id": "case", "file_id": "hydrology"},
        files={"file": ("runoff.tif", _tiff(), "image/tiff")},
    )
    assert response.status_code == 415
    response = await _upload(client, b"not a zip")
    assert response.status_code == 422


@pytest.mark.anyio
@pytest.mark.parametrize("params,status", [
    ({"ssp": "SSP3"}, 422), ({"year": 2050}, 422),
    ({"ssp": "SSP0", "year": 2050}, 422), ({"ssp": "SSP3", "year": 2051}, 422),
    ({"ssp": "SSP5", "year": 2050}, 404), ({"session_id": "missing"}, 404),
    ({"session_id": "../case"}, 422),
])
async def test_target_errors(client, session, params, status):
    response = await _upload(client, b"unused", **params)
    assert response.status_code == status, response.text


@pytest.mark.anyio
@pytest.mark.parametrize("setting,value", [
    ("hydrology_upload_max_bytes", 10), ("hydrology_upload_expanded_bytes", 10),
    ("hydrology_upload_max_pixels", 1),
])
async def test_upload_limits(client, session, monkeypatch, setting, value):
    monkeypatch.setattr(hydrology_upload.settings, setting, value)
    response = await _upload(client, _archive())
    assert response.status_code == 413, response.text
    assert not (session / "baseline/hydrology").exists()


@pytest.mark.anyio
async def test_custom_generation_preserves_and_inherits_without_static_sources(client, session):
    assert (await _upload(client, _archive(2))).status_code == 200
    destination = session / "scenarios/SSP3_2050"
    arguments = {"shapefile_path": session / "absent.shp", "static_data_dir": session / "absent-static"}
    baseline = hydrology.generate_hydrology_inputs(**arguments, out_dir=session / "baseline")
    assert baseline["source"]["source"] == "custom_uploaded"
    result = hydrology.generate_hydrology_inputs(**arguments, out_dir=destination, ssp="SSP3", year=2050)
    assert result["source"]["source"] == "custom_baseline_reused"
    assert "not climate-projected" in result["source"]["assumption"]
    assert (await _upload(client, _archive(3))).status_code == 200
    hydrology.generate_hydrology_inputs(**arguments, out_dir=destination, ssp="SSP3", year=2050)
    assert (destination / "hydrology/runoff/runoff_m01.tif").read_bytes() == _tiff(3)
    assert (await _upload(client, _archive(4), ssp="SSP3", year=2050)).status_code == 200
    hydrology.generate_hydrology_inputs(**arguments, out_dir=destination, ssp="SSP3", year=2050)
    assert (destination / "hydrology/runoff/runoff_m01.tif").read_bytes() == _tiff(4)
    with pytest.raises(HTTPException) as error:
        hydrology.generate_hydrology_inputs(**arguments, out_dir=destination, ssp="SSP3", hydrology_downscaling=True)
    assert error.value.status_code == 409


@pytest.mark.anyio
async def test_custom_hydrology_without_emissions_retains_original_grid(client, session):
    assert (await _upload(client, _archive())).status_code == 200
    with rasterio.open(session / "baseline/hydrology/routing/flowdir.tif") as source:
        assert source.res == (0.25, 0.25)


@pytest.mark.anyio
@pytest.mark.parametrize("scenario", [False, True])
async def test_upload_preserves_incompatible_hydrology_and_emissions_grids(client, session, scenario):
    target = session / ("scenarios/SSP3_2050" if scenario else "baseline")
    human = target if scenario else target / "human_emissions"
    human.mkdir(exist_ok=True)
    (human / "isoraster.tif").write_bytes(_tiff(7, resolution=0.5))
    (human / "pop_urban.tif").write_bytes(_tiff(10, resolution=0.5))
    response = await _upload(client, _archive(), **({"ssp": "SSP3", "year": 2050} if scenario else {}))
    assert response.status_code == 200, response.text
    with rasterio.open(human / "pop_urban.tif") as source:
        assert source.res == (0.5, 0.5)
        assert np.nansum(source.read(1)) == pytest.approx(160)
    with rasterio.open(target / "hydrology/routing/flowdir.tif") as source:
        assert source.res == (0.25, 0.25)


@pytest.mark.anyio
async def test_upload_alignment_failure_keeps_previous_package(client, session, monkeypatch):
    previous = session / "baseline/hydrology/previous.txt"
    previous.parent.mkdir()
    previous.write_text("previous hydrology")

    def fail_alignment(package):
        raise HTTPException(422, "simulated alignment failure")

    monkeypatch.setattr(hydrology_upload, "align_model_grid", fail_alignment)
    response = await _upload(client, _archive())
    assert response.status_code == 422
    assert previous.read_text() == "previous hydrology"



@pytest.mark.anyio
async def test_preview_uses_custom_grid_and_average(client, session, monkeypatch):
    from waterpath_data_service.web.api.geodata import views as preview_views
    monkeypatch.setattr(preview_views, "_DATA_DIR", session.parent)
    assert (await _upload(client, _archive(7))).status_code == 200
    response = await client.post("/api/geodata/preview", params={"session_id": "case", "file": "hydrology-runoff"})
    assert response.status_code == 200
    with MemoryFile(response.content) as memory, memory.open() as raster:
        assert raster.res == (0.25, 0.25)
        np.testing.assert_array_equal(raster.read(1), np.full((4, 4), 7))


@pytest.mark.anyio
async def test_scenario_csv_preserves_other_schema_columns(client, session):
    destination = session / "scenarios/SSP3_2050/isodata.csv"
    destination.write_text("gid,iso,population,fraction_urban_pop,flushSewer_urb\nUGA,1,100,0.4,0.2\n")
    response = await client.post(
        "/api/data/input/upload",
        params={"session_id": "case", "file_id": "population", "ssp": "SSP3", "year": 2050},
        files={"file": ("population.csv", b"gid,population\nUGA,200\n", "text/csv")},
    )
    assert response.status_code == 200, response.text
    frame = views.pd.read_csv(destination)
    assert frame.iloc[0]["population"] == 200
    assert frame.iloc[0]["flushSewer_urb"] == 0.2


def test_failed_promotion_restores_previous_folder(tmp_path, monkeypatch):
    destination = tmp_path / "hydrology"
    destination.mkdir()
    (destination / "old").write_text("last good")
    staging = tmp_path / "staging"
    staging.mkdir()
    staged = staging / "hydrology"
    staged.mkdir()
    original = Path.rename

    def fail_promotion(path, target):
        if path == staged:
            raise OSError("simulated promotion failure")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", fail_promotion)
    with pytest.raises(OSError):
        hydrology_upload.replace_folder(staged, destination)
    assert (destination / "old").read_text() == "last good"


def test_promotion_retries_transient_permission_error(tmp_path, monkeypatch):
    staged = tmp_path / "staged"
    staged.mkdir()
    destination = tmp_path / "hydrology"
    original = Path.rename
    attempts = 0

    def transient(path, target):
        nonlocal attempts
        if path == staged and attempts < 2:
            attempts += 1
            raise PermissionError("simulated bind-mount contention")
        return original(path, target)

    monkeypatch.setattr(Path, "rename", transient)
    hydrology_upload.replace_folder(staged, destination)
    assert destination.is_dir()
    assert attempts == 2


@pytest.mark.anyio
async def test_session_conflict(client, session):
    with hydrology_upload.session_lock(session):
        response = await _upload(client, b"unused")
    assert response.status_code == 409


@pytest.mark.anyio
async def test_generic_raster_is_not_silently_interpreted_as_d8(client, session):
    response = await client.post(
        "/api/data/input/generate", params={"session_id": "case", "gids": "UGA"},
        files={"hydro_raster": ("hydro_raster.tif", _tiff(), "image/tiff")},
    )
    assert response.status_code == 422
    assert "not yet defined" in response.json()["detail"]


@pytest.mark.anyio
async def test_custom_conflicts_are_rejected_before_generation(client, session):
    assert (await _upload(client, _archive())).status_code == 200
    for route, params in [
        ("/input/generate", {"gids": "UGA", "include_hydrology": True}),
        ("/projections/generate", {"schema": "hydrology", "ssp": "SSP3", "year": 2050}),
    ]:
        response = await client.post("/api/data" + route, params={
            "session_id": "case", "hydrology_downscaling": True, **params,
        })
        assert response.status_code == 409, response.text


def test_baseline_openapi_marks_experimental_controls():
    from waterpath_data_service.web.application import get_app
    schema = get_app().openapi()
    operation = schema["paths"]["/api/data/input/generate"]["post"]
    parameter = next(item for item in operation["parameters"] if item["name"] == "hydrology_downscaling")
    assert parameter["schema"]["default"] is False
    assert "HIGHLY EXPERIMENTAL" in parameter["description"]
    body_ref = operation["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    properties = schema["components"]["schemas"][body_ref.rsplit("/", 1)[-1]]["properties"]
    assert "hydro_raster" in properties
    assert "hydrology_flow_direction_tif" not in properties


@pytest.mark.anyio
async def test_facility_treatment_upload_does_not_require_gid(client, session):
    schema = session / "schemas/treatment.json"
    schema.parent.mkdir()
    schema.write_bytes((views._STATIC_DIR / "schemas/treatment_high_resolution.json").read_bytes())
    contents = b"lon,lat,capacity,treatment_type\n32.1,0.5,100,Primary\n32.2,0.6,200,Secondary\n"
    response = await client.post(
        "/api/data/input/upload", params={"session_id": "case", "file_id": "treatment"},
        files={"file": ("treatment.csv", contents, "text/csv")},
    )
    assert response.status_code == 200, response.text
    assert (session / "baseline/human_emissions/treatment.csv").read_bytes() == contents