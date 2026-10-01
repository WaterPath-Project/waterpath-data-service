import logging
from importlib import metadata
from pathlib import Path

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, UJSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from waterpath_data_service.log import configure_logging
from waterpath_data_service.web.api.router import api_router
from waterpath_data_service.web.lifespan import lifespan_setup

APP_ROOT = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


def _request_log_context(request: Request) -> dict[str, object]:
    client = request.client
    return {
        "method": request.method,
        "path": request.url.path,
        "query": dict(request.query_params),
        "client": f"{client.host}:{client.port}" if client else None,
    }


def get_app() -> FastAPI:
    """
    Get FastAPI application.

    This is the main constructor of an application.

    :return: application.
    """
    configure_logging()
    app = FastAPI(
        title="WaterPath_Data_Service",
        version=metadata.version("WaterPath_Data_Service"),
        lifespan=lifespan_setup,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
        default_response_class=UJSONResponse,
    )

    @app.middleware("http")
    async def log_unhandled_errors(request: Request, call_next):
        try:
            return await call_next(request)
        except Exception:
            logger.exception(
                "Unhandled request error: context=%s",
                _request_log_context(request),
            )
            raise

    @app.exception_handler(StarletteHTTPException)
    async def log_http_errors(request: Request, exc: StarletteHTTPException):
        if exc.status_code >= 500:
            underlying_error = exc.__cause__ or exc.__context__
            exception_info = (
                (
                    type(underlying_error),
                    underlying_error,
                    underlying_error.__traceback__,
                )
                if underlying_error is not None
                else None
            )
            logger.error(
                "HTTP %d response: context=%s detail=%r",
                exc.status_code,
                _request_log_context(request),
                exc.detail,
                exc_info=exception_info,
            )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
            headers=exc.headers,
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Main router for the API.
    app.include_router(router=api_router, prefix="/api")
    # Adds static directory.
    # This directory is used to access swagger files.
    app.mount("/static", StaticFiles(directory=APP_ROOT / "static"), name="static")

    return app
