from pathlib import Path

import pytest
from httpx import AsyncClient

from waterpath_data_service.web.api.data import views


def _write_isodata(path: Path, population: int, urban_fraction: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "gid,population,fraction_urban_pop,fraction_pop_under5,hdi,"
        "flushSewer_urb,flushSewer_rur,openDefecation_urb,openDefecation_rur,"
        "sewageTreated_urb,sewageTreated_rur,fecalSludgeTreated_urb,"
        "fecalSludgeTreated_rur\n"
        f"UGA,{population},{urban_fraction},0.15,0.55,0.6,0.2,0.1,0.3,"
        "0.4,0.2,0.3,0.1\n",
        encoding="utf-8",
    )


@pytest.mark.anyio
async def test_summarize_returns_baseline_and_sorted_scenario_values(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)
    session_dir = tmp_path / "case"
    _write_isodata(
        session_dir / "baseline" / "human_emissions" / "isodata.csv",
        population=100,
        urban_fraction=0.25,
    )
    _write_isodata(
        session_dir / "scenarios" / "SSP3_2050" / "isodata.csv",
        population=300,
        urban_fraction=0.75,
    )
    _write_isodata(
        session_dir / "scenarios" / "SSP1_2030" / "isodata.csv",
        population=200,
        urban_fraction=0.5,
    )

    response = await client.get(
        "/api/data/input/summarize",
        params={"session_id": "case"},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert payload["baseline_scenario_id"] == "baseline"
    assert [scenario["id"] for scenario in payload["scenarios"]] == [
        "baseline",
        "SSP1_2030",
        "SSP3_2050",
    ]
    assert payload["scenarios"][0]["metrics"]["population_total"] == 100
    assert payload["scenarios"][1]["metrics"]["population_total"] == 200
    assert payload["scenarios"][2]["metrics"]["population_urban_mean_pct"] == 75
    assert len(payload["metrics"]) == 30
    assert payload["metrics"][0]["delta_mode"] == "relative_pct"
    assert "delta" not in payload["scenarios"][1]


@pytest.mark.anyio
async def test_summarize_rejects_unknown_session(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(views, "_DATA_DIR", tmp_path)

    response = await client.get(
        "/api/data/input/summarize",
        params={"session_id": "missing"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Session ID not found."