import uvicorn

from WaterPath_Data_Service.settings import settings


def main() -> None:
    """Entrypoint of the application."""
    uvicorn.run(
        "WaterPath_Data_Service.web.application:get_app",
        workers=settings.workers_count,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.value.lower(),
        factory=True,
    )


if __name__ == "__main__":
    main()
