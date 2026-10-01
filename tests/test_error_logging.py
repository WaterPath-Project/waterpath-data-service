import warnings
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, HTTPException
from httpx import AsyncClient
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine

from waterpath_data_service.services import prepare_spatial
from waterpath_data_service.web import application


def test_raster_warning_logs_path_and_gids(monkeypatch: pytest.MonkeyPatch) -> None:
    source = Mock()
    source.crs = None
    source.transform = Affine.identity()

    def open_without_georeferencing(_path: str) -> Mock:
        warnings.warn(
            "Dataset has no geotransform, gcps, or rpcs.",
            NotGeoreferencedWarning,
        )
        return source

    warning_log = Mock()
    monkeypatch.setattr(prepare_spatial.rasterio, "open", open_without_georeferencing)
    monkeypatch.setattr(prepare_spatial.logger, "warning", warning_log)

    resolution = prepare_spatial._native_tif_resolution(
        "population.tif", ["UGA.1_1", "UGA.2_1"]
    )

    assert resolution == 1
    warning_log.assert_called_once()
    message, raster_path, gids, *_ = warning_log.call_args.args
    assert "no georeferencing metadata" in message
    assert raster_path == "population.tif"
    assert gids == ["UGA.1_1", "UGA.2_1"]
    source.close.assert_called_once()


@pytest.mark.anyio
async def test_http_500_logs_request_context(
    client: AsyncClient,
    fastapi_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_log = Mock()
    monkeypatch.setattr(application.logger, "error", error_log)

    @fastapi_app.get("/_test/internal-error")
    async def internal_error() -> None:
        try:
            raise ValueError("source raster is invalid")
        except ValueError as exc:
            raise HTTPException(
                status_code=500,
                detail="projection failed",
            ) from exc

    response = await client.get(
        "/_test/internal-error",
        params={"session_id": "case", "gids": "UGA.1_1,UGA.2_1"},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "projection failed"}
    error_log.assert_called_once()
    _, status_code, context, detail = error_log.call_args.args
    assert status_code == 500
    assert context["method"] == "GET"
    assert context["path"] == "/_test/internal-error"
    assert context["query"]["gids"] == "UGA.1_1,UGA.2_1"
    assert detail == "projection failed"
    exception_type, exception, traceback = error_log.call_args.kwargs["exc_info"]
    assert exception_type is ValueError
    assert str(exception) == "source raster is invalid"
    assert traceback is not None