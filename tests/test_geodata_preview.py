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


class _GeometryResult:
    def __init__(self, admin: list[str], level: int) -> None:
        self._admin = admin
        self._level = level
        self.geometry = _GeometrySeries()

    def to_geo_dict(self) -> dict:
        return {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"GID": area, "level": self._level},
                    "geometry": {
                        "type": "Point",
                        "coordinates": [32.123456789, 1.987654321],
                    },
                }
                for area in self._admin
            ],
        }


class _GeometrySeries:
    def __init__(self) -> None:
        self.simplify_calls = []

    def simplify(
        self,
        tolerance: float,
        preserve_topology: bool,
    ) -> "_GeometrySeries":
        self.simplify_calls.append((tolerance, preserve_topology))
        return self


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
async def test_get_geometries_returns_geojson_for_gadm_ids(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    results = []
    views._get_geometry_features.cache_clear()

    def items(admin: list[str], content_level: int) -> _GeometryResult:
        calls.append((admin, content_level))
        result = _GeometryResult(admin, content_level)
        results.append(result)
        return result

    monkeypatch.setattr(views.pygadm, "Items", items)

    response = await client.post(
        "/api/geodata/get-geometries",
        json=[" UGA.1 ", "UGA.1.2", "UGA.3"],
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/geo+json"
    assert response.json()["type"] == "FeatureCollection"
    assert response.json()["features"][0]["geometry"]["coordinates"] == [
        32.12346,
        1.98765,
    ]
    assert [feature["properties"]["GID"] for feature in response.json()["features"]] == [
        "UGA.1",
        "UGA.3",
        "UGA.1.2",
    ]
    assert calls == [
        (["UGA.1", "UGA.3"], 1),
        (["UGA.1.2"], 2),
    ]
    assert all(
        result.geometry.simplify_calls == [(0.005, True)]
        for result in results
    )


@pytest.mark.anyio
async def test_get_geometries_can_return_unsimplified_geometry_and_caches_results(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    results = []
    views._get_geometry_features.cache_clear()

    def items(admin: list[str], content_level: int) -> _GeometryResult:
        calls.append((admin, content_level))
        result = _GeometryResult(admin, content_level)
        results.append(result)
        return result

    monkeypatch.setattr(views.pygadm, "Items", items)

    for _ in range(2):
        response = await client.post(
            "/api/geodata/get-geometries",
            params={"simplify_tolerance": 0},
            json=["UGA.1"],
        )
        assert response.status_code == 200

    assert calls == [(["UGA.1"], 1)]
    assert results[0].geometry.simplify_calls == []


@pytest.mark.anyio
async def test_get_geometries_fetches_shared_parent_once(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gadm_ids = [
        "UGA.1.1.1.1_1",
        "UGA.1.1.1.2_1",
        "UGA.1.1.2.1_1",
    ]
    calls = []
    views._get_geometry_features.cache_clear()

    def items(admin: list[str], content_level: int) -> gpd.GeoDataFrame:
        calls.append((admin, content_level))
        return gpd.GeoDataFrame(
            {"GID_4": list(reversed(gadm_ids))},
            geometry=[Point(32.0, 1.0), Point(32.1, 1.1), Point(32.2, 1.2)],
            crs="EPSG:4326",
        )

    monkeypatch.setattr(views.pygadm, "Items", items)

    response = await client.post(
        "/api/geodata/get-geometries",
        json=gadm_ids,
    )

    assert response.status_code == 200
    assert calls == [(["UGA.1.1_1"], 4)]
    assert [
        feature["properties"]["GID_4"]
        for feature in response.json()["features"]
    ] == gadm_ids


@pytest.mark.anyio
@pytest.mark.parametrize("gadm_ids", [[], [""], ["   "]])
async def test_get_geometries_rejects_empty_gadm_ids(
    client: AsyncClient,
    gadm_ids: list[str],
) -> None:
    response = await client.post("/api/geodata/get-geometries", json=gadm_ids)

    assert response.status_code == 422


@pytest.mark.anyio
async def test_get_geometries_reports_invalid_gadm_id(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_items(admin: list[str], content_level: int) -> None:
        raise ValueError(f"Unknown GADM ID: {admin[0]}")

    monkeypatch.setattr(views.pygadm, "Items", invalid_items)

    response = await client.post(
        "/api/geodata/get-geometries",
        json=["INVALID.1"],
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Unknown GADM ID: INVALID.1"


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