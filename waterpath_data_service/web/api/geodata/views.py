import base64, io, os, pandas as pd, json, base64
from pathlib import Path
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel
import pygadm
from fastapi import APIRouter, HTTPException
import matplotlib.pyplot as plt

router = APIRouter()


@router.post("/geometries")
async def polygons(admin: str, level: int) -> JSONResponse:

    # project_folder = Path(__file__).parent.parent.parent.parent

    # if os.path.isdir(project_folder / "data" / session_id):
    #     session_folder = project_folder / "data" / session_id
    # else:
    #     raise HTTPException(status_code=404, detail="Session ID not found.")

    areas = [x.strip(" ") for x in admin.split(",")]

    gdf = pygadm.Items(admin=list(areas), content_level=level)
    geometries = gdf.to_geo_dict()
    return JSONResponse(content=geometries)

@router.post("/names")
def geonames(admin: str) -> JSONResponse:

    areas = [x.strip(" ") for x in admin.split(",")]
    names = []
    for area in areas:
        level = area.count('.')
        try:
            name = pygadm.Names(admin=area, content_level=level)
            names.extend(name["NAME_"+str(level)].tolist())
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
        except:
            raise HTTPException(status_code=500, detail="Error processing provided areas.")
        
    names_df = pd.DataFrame(columns=["gid", "name"])
    names_df["gid"] = areas
    names_df["name"] = names
    return JSONResponse(content = names)


@router.post("/preview")
def preview(session_id: str) -> PlainTextResponse:
    project_folder = Path(__file__).parent.parent.parent.parent
    encoded_string = ""
    if os.path.isdir(project_folder / "data" / session_id):
        session_folder = project_folder / "data" / session_id
        if os.path.isfile(project_folder / "data" / session_id / "default" / "population.csv"):
            df = pd.read_csv(project_folder / "data" / session_id / "default" / "population.csv")
            areas = df['gid'].tolist()
            gdf = pygadm.Items(admin=list(areas), content_level=areas[0].count('.'))
            ax = gdf.plot(color='#8DD0A4', edgecolor='#0B4159')
            ax.set_axis_off()
            pic_IObytes = io.BytesIO()
            ax.figure.savefig(pic_IObytes,  format='png')
            pic_IObytes.seek(0)
            pic_hash = base64.b64encode(pic_IObytes.read())
            encoded_string = str(pic_hash)
            
    else:
        raise HTTPException(status_code=404, detail="Session ID not found.")
    return PlainTextResponse(content=encoded_string)
    
    
@router.get("/get-areas")
def get_areas(country_code: str, level: int) -> JSONResponse:
    print(country_code)
    print(level)
    try:
        name = pygadm.Names(admin=country_code, content_level=level, complete=True)

        return JSONResponse(content=json.loads(name.to_json(orient='records')))
        # names.extend(name["NAME_"+str(level)].tolist())
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except:
        raise HTTPException(status_code=500, detail="Error processing provided areas.")
        
    # names_df = pd.DataFrame(columns=["gid", "name"])
    # names_df["gid"] = areas
    # names_df["name"] = names
    return JSONResponse(content = [])