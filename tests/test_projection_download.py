import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from httpx import AsyncClient

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

    def fake_shapefile(_gids: list[str], output_dir: str) -> str:
        path = Path(output_dir) / "geodata" / "geodata.shp"
        path.parent.mkdir()
        path.touch()
        return str(path)

    def fake_prepare_spatial_inputs(**kwargs: str) -> dict[str, str]:
        output_dir = Path(kwargs["out_dir"])
        output_dir.mkdir()
        template_path = Path(kwargs["template_raster_path"])
        assert template_path.read_bytes() == b"baseline-grid"
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
            "isoraster": ("isoraster.tif", b"baseline-grid", "image/tiff"),
        },
    )

    assert response.status_code == 200
    assert captured_reference["path"].parent.name == "projection_grid"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "isoraster.tif" not in archive.namelist()
