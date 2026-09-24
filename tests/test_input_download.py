import io
import zipfile
from pathlib import Path

import pytest
from httpx import AsyncClient

from waterpath_data_service.web.api.data import views


def _write_csv(path: Path, content: str = "iso,value\n1,0.5\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.mark.anyio
async def test_download_previews_livestock_csvs(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    livestock_dir = tmp_path / "case" / "baseline" / "livestock_emissions"
    previews = {
        "livestock_manure_fractions": ("meat_fgi", "0.5"),
        "livestock_manure_management": ("PP_cattle", "0.6"),
        "livestock_production_systems": ("meat_i", "0.7"),
    }
    for file_id, (field, value) in previews.items():
        csv_name = file_id.removeprefix("livestock_") + ".csv"
        _write_csv(
            livestock_dir / csv_name,
            f"iso,{field},extra\n1,{value},ignored\n",
        )
        response = await client.get(
            "/api/data/input/download",
            params={"session_id": "case", "file_id": file_id},
        )

        assert response.status_code == 200
        assert response.text == f"iso,{field}\n1,{value}\n"
        assert f'filename="{file_id}.csv"' in response.headers["content-disposition"]


@pytest.mark.anyio
async def test_download_combines_livestock_isodata_previews(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    animals_dir = (
        tmp_path / "case" / "baseline" / "livestock_emissions" / "animals"
    )
    _write_csv(animals_dir / "isodata_cattle.csv", "iso,frac_young,extra\n1,0.5,x\n")
    _write_csv(animals_dir / "isodata_goats.csv", "iso,frac_young,extra\n2,0.7,y\n")

    response = await client.get(
        "/api/data/input/download",
        params={"session_id": "case", "file_id": "livestock_isodata"},
    )

    assert response.status_code == 200
    assert response.text == "iso,frac_young\n1,0.5\n2,0.7\n"


@pytest.mark.anyio
async def test_download_requires_scenario_and_year_together(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    (tmp_path / "case").mkdir()

    response = await client.get(
        "/api/data/input/download",
        params={"session_id": "case", "scenario": "SSP3"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "scenario and year must be provided together."


@pytest.mark.anyio
async def test_download_selects_scenario_preview_and_archive(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    session_dir = tmp_path / "case"
    scenario_dir = session_dir / "scenarios" / "SSP3_2050"
    _write_csv(session_dir / "baseline" / "human_emissions" / "treatment.csv", "source\nbaseline\n")
    _write_csv(
        scenario_dir / "treatment.csv",
        "gid,FractionPrimarytreatment,extra\nUGA,0.3,ignored\n",
    )
    _write_csv(scenario_dir / "livestock_emissions" / "production_systems.csv")

    params = {"session_id": "case", "scenario": "ssp3", "year": 2050}
    preview = await client.get(
        "/api/data/input/download",
        params={**params, "file_id": "treatment"},
    )
    archive = await client.get("/api/data/input/download", params=params)

    assert preview.status_code == 200
    assert preview.text == "gid,FractionPrimarytreatment\nUGA,0.3\n"
    assert archive.status_code == 200
    with zipfile.ZipFile(io.BytesIO(archive.content)) as zipf:
        assert sorted(zipf.namelist()) == [
            "scenarios/SSP3_2050/livestock_emissions/production_systems.csv",
            "scenarios/SSP3_2050/treatment.csv",
        ]


@pytest.mark.anyio
async def test_scenario_preview_filters_merged_isodata_by_schema(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    scenario_dir = tmp_path / "case" / "scenarios" / "SSP3_2050"
    _write_csv(
        scenario_dir / "isodata.csv",
        "gid,iso,population,fraction_urban_pop,flushSewer_urb,extra\n"
        "UGA,1,100,0.4,0.2,ignored\n",
    )

    params = {
        "session_id": "case",
        "scenario": "SSP3",
        "year": 2050,
    }
    population = await client.get(
        "/api/data/input/download",
        params={**params, "file_id": "population"},
    )
    sanitation = await client.get(
        "/api/data/input/download",
        params={**params, "file_id": "sanitation"},
    )

    assert population.status_code == 200
    assert population.text == "iso,gid,population,fraction_urban_pop\n1,UGA,100,0.4\n"
    assert population.headers["cache-control"] == "no-store"
    assert sanitation.status_code == 200
    assert sanitation.text == "gid,flushSewer_urb\nUGA,0.2\n"