import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from waterpath_data_service.settings import settings

router = APIRouter()

_DATA_DIR: Path = settings.data_dir


@router.post("/create/")
async def create_session(session_id: str) -> str:
    session_dir = _DATA_DIR / session_id

    if not os.path.isdir(session_dir):
        os.makedirs(session_dir)
    else:
        raise HTTPException(
            status_code=500,
            detail="Session folder could not be created. Perhaps it already exists or the provided name is invalid.",
        )
    return session_id


@router.get("/")
async def get_session_items(session_id: str) -> list:
    try:
        session_dir = _DATA_DIR / session_id
        if os.path.isdir(session_dir):
            return [x.name for x in session_dir.glob("**/*")]
        else:
            raise HTTPException(status_code=404, detail="Session ID not found.")
    except:
        raise HTTPException(
            status_code=404,
            detail="Session data could not be retrieved.",
        )
