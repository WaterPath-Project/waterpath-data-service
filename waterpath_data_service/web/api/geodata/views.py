import io, os, pandas as pd, json
from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
import pygadm
from fastapi import APIRouter, HTTPException, Query
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from waterpath_data_service.settings import settings

router = APIRouter()

_DATA_DIR: Path = settings.data_dir

_PREVIEW_FILES = {
    "population-distribution",
    "livestock-distribution",
    "hydrology-flow",
    "hydrology-river_temperature",
    "hydrology-ssrd",
    "hydrology-runoff",
    "risk-treatment",
}
_HYDROLOGY_VARIABLES = {
    "hydrology-flow": "discharge",
    "hydrology-river_temperature": "river_temperature",
    "hydrology-ssrd": "ssrd",
    "hydrology-runoff": "runoff",
}
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _preview_data_dir(
    session_dir: Path,
    ssp: str | None,
    year: int | None,
) -> Path:
    if (ssp is None) != (year is None):
        raise HTTPException(status_code=422, detail="SSP and year must be provided together.")
    if ssp is None:
        return session_dir / "baseline"

    ssp_norm = ssp.strip().upper()
    if ssp_norm not in {f"SSP{i}" for i in range(1, 6)}:
        raise HTTPException(status_code=422, detail="Invalid SSP. Allowed: SSP1..SSP5")
    scenario_dir = session_dir / "scenarios" / f"{ssp_norm}_{year}"
    if not scenario_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"No preview data found for {ssp_norm} and year {year}.",
        )
    return scenario_dir


def _monthly_hydrology_path(
    data_dir: Path,
    variable: str,
    dimension: str | None,
) -> Path | None:
    variable_dir = data_dir / "hydrology" / variable
    if dimension is None:
        return None
    month = _MONTHS.get(dimension.strip().lower())
    if month is None:
        raise HTTPException(
            status_code=422,
            detail="Hydrology dimension must be a three-letter month (jan..dec).",
        )
    return variable_dir / f"{variable}_m{month:02d}.tif"


def _average_hydrology_rasters(variable_dir: Path, variable: str) -> bytes:
    paths = [variable_dir / f"{variable}_m{month:02d}.tif" for month in range(1, 13)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Missing monthly {variable} raster(s): {', '.join(missing)}",
        )

    arrays = []
    profile = None
    for path in paths:
        with rasterio.open(path) as source:
            array = source.read(1).astype(np.float64)
            if profile is None:
                profile = source.profile.copy()
            if source.nodata is not None:
                array[array == source.nodata] = np.nan
        array[array < 0] = np.nan
        arrays.append(array)

    monthly_stack = np.stack(arrays, axis=0)
    valid_counts = np.sum(np.isfinite(monthly_stack), axis=0)
    valid_sums = np.nansum(monthly_stack, axis=0)
    average = np.divide(
        valid_sums,
        valid_counts,
        out=np.full(valid_sums.shape, np.nan, dtype=np.float64),
        where=valid_counts > 0,
    )
    output_nodata = -9999.0
    average = np.where(np.isfinite(average), average, output_nodata)
    profile.update(dtype="float64", count=1, nodata=output_nodata, compress="lzw")

    with MemoryFile() as memory_file:
        with memory_file.open(**profile) as destination:
            destination.write(average, 1)
        return memory_file.read()


def _preview_raster_path(
    data_dir: Path,
    file_name: str,
    dimension: str | None,
) -> tuple[Path | None, bytes | None, str]:
    if file_name == "population-distribution":
        if dimension is not None:
            raise HTTPException(status_code=422, detail="population-distribution does not accept dimension.")
        baseline_path = data_dir / "human_emissions" / "pop_urban.tif"
        path = baseline_path if baseline_path.is_file() else data_dir / "pop_urban.tif"
        return path, None, "pop_urban.tif"

    if file_name == "livestock-distribution":
        animal = (dimension or "").strip().lower()
        if not animal or not animal.replace("_", "").isalnum():
            raise HTTPException(
                status_code=422,
                detail="livestock-distribution requires an animal dimension.",
            )
        filename = f"{animal}_heads.tif"
        return data_dir / "livestock_emissions" / "animals" / filename, None, filename

    if file_name in _HYDROLOGY_VARIABLES:
        variable = _HYDROLOGY_VARIABLES[file_name]
        monthly_path = _monthly_hydrology_path(data_dir, variable, dimension)
        if monthly_path is not None:
            return monthly_path, None, monthly_path.name
        filename = f"{variable}_average.tif"
        content = _average_hydrology_rasters(data_dir / "hydrology" / variable, variable)
        return None, content, filename

    if dimension is not None:
        raise HTTPException(status_code=422, detail="risk-treatment does not accept dimension.")
    return data_dir / "qmra" / "treatment.tif", None, "treatment.tif"


@router.post("/geometries")
async def polygons(admin: str, level: int) -> JSONResponse:

    # project_folder = Path(__file__).parent.parent.parent.parent

    # if os.path.isdir(project_folder / "data" / session_id):
    #     session_folder = project_folder / "data" / session_id
    # else:
    #     raise HTTPException(status_code=404, detail="Session ID not found.")

    areas = [x.strip() for x in admin.split(",") if x.strip()]

    gdf = pygadm.Items(admin=list(areas), content_level=level)
    geometries = gdf.to_geo_dict()
    return JSONResponse(content=geometries)

@router.post("/names")
def geonames(admin: str) -> JSONResponse:

    areas = [x.strip() for x in admin.split(",") if x.strip()]
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
def preview(
    session_id: str,
    year: int | None = None,
    ssp: str | None = Query(None, alias="SSP"),
    file: str | None = None,
    dimension: str | None = None,
) -> Response:
    session_dir = _DATA_DIR / session_id
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail="Session ID not found.")

    if file is None:
        if any(value is not None for value in (ssp, year, dimension)):
            raise HTTPException(
                status_code=422,
                detail="file is required when SSP, year, or dimension is provided.",
            )
        shapefile_path = session_dir / "baseline" / "geodata" / "geodata.shp"
        if not shapefile_path.is_file():
            shapefile_path = session_dir / "geodata" / "geodata.shp"
        if not shapefile_path.is_file():
            raise HTTPException(status_code=404, detail="Session geodata shapefile not found.")
        geodata = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
        return JSONResponse(content=json.loads(geodata.to_json()), headers=_NO_CACHE_HEADERS)

    file_name = file.strip().lower()
    if file_name not in _PREVIEW_FILES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file. Allowed: {sorted(_PREVIEW_FILES)}",
        )

    data_dir = _preview_data_dir(session_dir, ssp, year)
    raster_path, raster_bytes, filename = _preview_raster_path(
        data_dir,
        file_name,
        dimension,
    )
    if raster_bytes is not None:
        return StreamingResponse(
            io.BytesIO(raster_bytes),
            media_type="image/tiff",
            headers={
                **_NO_CACHE_HEADERS,
                "Content-Disposition": f'inline; filename="{filename}"',
            },
        )
    if raster_path is None or not raster_path.is_file():
        raise HTTPException(status_code=404, detail=f"Preview raster not found: {filename}")
    return FileResponse(
        raster_path,
        media_type="image/tiff",
        filename=filename,
        content_disposition_type="inline",
        headers=_NO_CACHE_HEADERS,
    )
    
    
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