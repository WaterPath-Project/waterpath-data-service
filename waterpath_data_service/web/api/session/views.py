import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/create/")
async def create_session(session_id: str) -> str:
    path = "data/" + session_id
    project_folder = Path(__file__).parent.parent.parent.parent

    if not os.path.isdir(project_folder / path):
        os.makedirs(project_folder / path)
    else:
        raise HTTPException(
            status_code=500,
            detail="Session folder could not be created. Perhaps it already exists or the provided name is invalid.",
        )
    return session_id


@router.get("/")
async def get_session_items(session_id: str) -> list:
    project_folder = Path(__file__).parent.parent.parent.parent
    try:
        if os.path.isdir(project_folder / "data" / session_id):
            session_folder = project_folder / "data" / session_id
            return [x.name for x in session_folder.glob("**/*")]
        else:
            raise HTTPException(status_code=404, detail="Session ID not found.")
    except:
        raise HTTPException(
            status_code=404,
            detail="Session data could not be retrieved.",
        )
