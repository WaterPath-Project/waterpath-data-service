import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from httpx import AsyncClient
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from waterpath_data_service.services import projections
from waterpath_data_service.web.api.data import views


@pytest.mark.anyio
async def test_livestock_only_download_uses_private_population_grid(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_tif = tmp_path / "population.tif"
    source_tif.touch()
    captured_reference: dict[str, Path] = {}
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="int32",
            crs="EPSG:4326",
            transform=from_origin(31.0, 1.0, 0.25, 0.25),
            nodata=0,
        ) as output:
            output.write(np.ones((2, 2), dtype=np.int32), 1)
        baseline_grid = memory.read()

    def fake_shapefile(_gids: list[str], output_dir: str) -> str:
        path = Path(output_dir) / "geodata" / "geodata.shp"
        path.parent.mkdir()
        path.touch()
        return str(path)

    def fake_prepare_spatial_inputs(**kwargs: str) -> dict[str, str]:
        output_dir = Path(kwargs["out_dir"])
        output_dir.mkdir()
        template_path = Path(kwargs["template_raster_path"])
        with rasterio.open(template_path) as template:
            assert template.shape == (2, 2)
            assert template.res == (0.25, 0.25)
        isoraster = output_dir / "isoraster.tif"
        isoraster.touch()
        return {
            "isoraster": str(isoraster),
            "pop_urban": str(output_dir / "pop_urban.tif"),
            "pop_rural": str(output_dir / "pop_rural.tif"),
        }

    def fake_build_template(
        _session_dir: Path,
        _static_data_dir: Path,
        _mapping: pd.DataFrame,
        reference_isoraster_path: Path | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        assert reference_isoraster_path is not None
        captured_reference["path"] = reference_isoraster_path
        zone_idx = np.ones((1, 1), dtype=np.int32)
        return zone_idx, zone_idx > 0, {}

    async def fake_fetch_livestock(
        _alpha3: list[str],
        _ssp: str,
        _year: int,
    ) -> pd.DataFrame:
        return pd.DataFrame({"alpha3": ["UGA"]})

    monkeypatch.setattr(views, "shapefile", fake_shapefile)
    monkeypatch.setattr(views, "prepare_spatial_inputs", fake_prepare_spatial_inputs)
    monkeypatch.setattr(projections, "_population_tif_path", lambda *_: source_tif)
    monkeypatch.setattr(
        views,
        "_load_iso_gid_mapping",
        lambda _: pd.DataFrame({"gid": ["UGA"], "iso": [1]}),
    )
    monkeypatch.setattr(views, "generate_livestock_tabular_inputs", lambda *_: {})
    monkeypatch.setattr(views, "_build_livestock_zone_template", fake_build_template)
    monkeypatch.setattr(views, "fetch_livestock_future_csv", fake_fetch_livestock)
    monkeypatch.setattr(
        views,
        "generate_livestock_projection_rasters",
        lambda **_: {"output_dir": str(tmp_path / "livestock"), "animal_heads": {}},
    )

    response = await client.post(
        "/api/data/projections/download",
        params={"schema": "livestock_emissions", "year": 2050, "ssp": "SSP2"},
        files={
            "file": (
                "baseline.csv",
                "gid,population,fraction_urban_pop\nUGA,100,0.2\n",
                "text/csv",
            ),
            "isoraster": ("isoraster.tif", baseline_grid, "image/tiff"),
        },
    )

    assert response.status_code == 200
    assert captured_reference["path"].parent.name == "projection_grid"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "isoraster.tif" not in archive.namelist()


def test_generate_population_isoraster_preserves_baseline_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "case"
    human_dir = session_dir / "baseline" / "human_emissions"
    human_dir.mkdir(parents=True)
    (human_dir / "population.csv").touch()
    template = human_dir / "isoraster.tif"
    with rasterio.open(
        template,
        "w",
        driver="GTiff",
        width=3,
        height=2,
        count=1,
        dtype="int32",
        crs="EPSG:4326",
        transform=from_origin(31.0, 1.0, 0.25, 0.25),
        nodata=0,
    ) as output:
        output.write(np.ones((2, 3), dtype=np.int32), 1)

    source_population = tmp_path / "source.tif"
    source_population.touch()
    shapefile = tmp_path / "geodata.shp"
    shapefile.touch()
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        projections,
        "_population_tif_path",
        lambda *_: source_population,
    )
    monkeypatch.setattr(projections, "_session_shapefile_path", lambda _: shapefile)

    def fake_prepare_spatial_inputs(**kwargs: str) -> dict[str, str]:
        captured.update(kwargs)
        output_path = Path(kwargs["out_dir"]) / "isoraster.tif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return {"isoraster": str(output_path)}

    monkeypatch.setattr(
        projections,
        "prepare_spatial_inputs",
        fake_prepare_spatial_inputs,
    )

    projections.generate_population_isoraster(
        session_dir=session_dir,
        scenario_dir=session_dir / "scenarios" / "SSP2_2050",
        static_data_dir=tmp_path,
        ssp="SSP2",
        year=2050,
    )

    assert captured["template_raster_path"] == str(template)


def test_projection_grid_validation_reports_mismatched_raster(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.tif"
    scenario_dir = tmp_path / "scenario"
    animal_isoraster = scenario_dir / "livestock_emissions" / "animal_isoraster.tif"
    animal_isoraster.parent.mkdir(parents=True)

    for path, resolution in ((reference, 0.25), (animal_isoraster, 0.1)):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=from_origin(31.0, 1.0, resolution, resolution),
        ) as output:
            output.write(np.ones((2, 2), dtype=np.float32), 1)

    with pytest.raises(ValueError, match="animal_isoraster.tif"):
        views._validate_generated_projection_grids(reference, scenario_dir)
