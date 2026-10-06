from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from starlette import status

from waterpath_data_service.web.api.data import views


@pytest.mark.anyio
async def test_health(client: AsyncClient, fastapi_app: FastAPI) -> None:
    """
    Checks the health endpoint.

    :param client: client for the app.
    :param fastapi_app: current FastAPI application.
    """
    url = fastapi_app.url_path_for("health_check")
    response = await client.get(url)
    assert response.status_code == status.HTTP_200_OK


def test_baseline_temperature_is_not_generated_without_livestock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generate_temperature = Mock()
    monkeypatch.setattr(views, "generate_temperature_tif", generate_temperature)

    views._generate_baseline_livestock_temperature(
        include_livestock=False,
        geodata_shp=tmp_path / "geodata.shp",
        human_emissions_output_path=tmp_path / "human_emissions",
        session_dir=tmp_path / "case",
    )

    generate_temperature.assert_not_called()
