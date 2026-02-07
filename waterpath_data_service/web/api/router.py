from fastapi.routing import APIRouter

from waterpath_data_service.web.api import data, docs, geodata, monitoring, session

api_router = APIRouter()
api_router.include_router(monitoring.router)
api_router.include_router(session.router, prefix="/session", tags=["session"])
api_router.include_router(geodata.router, prefix="/geodata", tags=["geodata"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(docs.router)
# api_router.include_router(echo.router, prefix="/echo", tags=["echo"])
