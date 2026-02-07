import base64
import io
import os
from pathlib import Path

import pygadm
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.post("/polygons")
async def polygons(admin: str, level: int, session_id: str) -> list:

    project_folder = Path(__file__).parent.parent.parent.parent

    if os.path.isdir(project_folder / "data" / session_id):
        session_folder = project_folder / "data" / session_id
        print(session_folder)
    else:
        raise HTTPException(status_code=404, detail="Session ID not found.")
    # print(session_folder)
    areas = [x.strip(" ") for x in admin.split(",")]

    gdf = pygadm.Items(admin=list(areas), content_level=level)
    print(gdf.head())
    plot = gdf.plot(legend=True, cmap="Set1", column="GID_2", kind="geo")

    # gdf.plot()
    my_stringIObytes = io.BytesIO()
    # plot = gdf.plot()
    buffer = io.BytesIO()
    plot.to_image.save(buffer, "png")
    b64 = base64.b64encode(buffer.getvalue())
    print(b64)
    # my_stringIObytes.seek(0)
    # my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
    return []
    try:
        print("k")
        # gdf = pygadm.AdmItems(name, content_level)
        # return gdf
    except:
        raise HTTPException(
            status_code=500,
            detail="Could not load specified geography.",
        )
    # return level
