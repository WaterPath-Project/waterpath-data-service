import io
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from httpx import AsyncClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Point

from waterpath_data_service.web.api.geodata import views


def _write_tif(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=1,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 1, 1, 1),
        nodata=-9999.0,
    ) as destination:
        destination.write(np.array([[value, -9999.0]], dtype=np.float32), 1)


def _read_response_raster(content: bytes) -> np.ndarray:
    with MemoryFile(content) as memory_file:
        with memory_file.open() as source:
            return source.read(1)


@pytest.mark.anyio
async def test_preview_without_file_returns_session_geojson(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    shapefile = tmp_path / "case" / "baseline" / "geodata" / "geodata.shp"
    shapefile.parent.mkdir(parents=True)
    shapefile.touch()
    geodata = gpd.GeoDataFrame(
        {"gid": ["UGA"]},
        geometry=[Point(32.0, 1.0)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(views.gpd, "read_file", lambda _: geodata)

    response = await client.post(
        "/api/geodata/preview",
        params={"session_id": "case"},
    )

    assert response.status_code == 200
    assert response.json()["type"] == "FeatureCollection"
    assert response.json()["features"][0]["properties"]["gid"] == "UGA"


@pytest.mark.anyio
async def test_preview_resolves_scenario_population_and_livestock(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    scenario = tmp_path / "case" / "scenarios" / "SSP3_2050"
    _write_tif(scenario / "pop_urban.tif", 10)
    _write_tif(scenario / "livestock_emissions" / "animals" / "cattle_heads.tif", 20)

    common = {"session_id": "case", "SSP": "SSP3", "year": 2050}
    population = await client.post(
        "/api/geodata/preview",
        params={**common, "file": "population-distribution"},
    )
    livestock = await client.post(
        "/api/geodata/preview",
        params={
            **common,
            "file": "livestock-distribution",
            "dimension": "cattle",
        },
    )

    assert population.status_code == 200
    assert population.headers["content-type"] == "image/tiff"
    assert _read_response_raster(population.content)[0, 0] == 10
    assert livestock.status_code == 200
    assert _read_response_raster(livestock.content)[0, 0] == 20


@pytest.mark.anyio
async def test_preview_selects_month_or_averages_hydrology(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    runoff_dir = tmp_path / "case" / "baseline" / "hydrology" / "runoff"
    for month in range(1, 13):
        _write_tif(runoff_dir / f"runoff_m{month:02d}.tif", month)

    common = {
        "session_id": "case",
        "file": "hydrology-runoff",
    }
    monthly = await client.post(
        "/api/geodata/preview",
        params={**common, "dimension": "mar"},
    )
    annual = await client.post("/api/geodata/preview", params=common)

    assert monthly.status_code == 200
    assert _read_response_raster(monthly.content)[0, 0] == 3
    assert annual.status_code == 200
    assert _read_response_raster(annual.content)[0, 0] == 6.5
    assert "runoff_average.tif" in annual.headers["content-disposition"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("file_name", "raster_path"),
    [
        ("hydrology-flow", "hydrology/discharge/discharge_m01.tif"),
        (
            "hydrology-river_temperature",
            "hydrology/river_temperature/river_temperature_m01.tif",
        ),
        ("hydrology-ssrd", "hydrology/ssrd/ssrd_m01.tif"),
        ("risk-treatment", "qmra/treatment.tif"),
    ],
)
async def test_preview_resolves_remaining_raster_types(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
    raster_path: str,
) -> None:
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    _write_tif(tmp_path / "case" / "baseline" / raster_path, 42)
    params = {"session_id": "case", "file": file_name}
    if file_name.startswith("hydrology-"):
        params["dimension"] = "jan"

    response = await client.post("/api/geodata/preview", params=params)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/tiff"
    assert _read_response_raster(response.content)[0, 0] == 42


@pytest.mark.anyio
async def test_preview_validates_dependent_parameters(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    (tmp_path / "case").mkdir()

    missing_file = await client.post(
        "/api/geodata/preview",
        params={"session_id": "case", "SSP": "SSP3", "year": 2050},
    )
    missing_year = await client.post(
        "/api/geodata/preview",
        params={
            "session_id": "case",
            "SSP": "SSP3",
            "file": "population-distribution",
        },
    )
    missing_animal = await client.post(
        "/api/geodata/preview",
        params={"session_id": "case", "file": "livestock-distribution"},
    )
    invalid_month = await client.post(
        "/api/geodata/preview",
        params={
            "session_id": "case",
            "file": "hydrology-flow",
            "dimension": "march",
        },
    )

    assert missing_file.status_code == 422
    assert missing_year.status_code == 422
    assert missing_animal.status_code == 422
    assert invalid_month.status_code == 422