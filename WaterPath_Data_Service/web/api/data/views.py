import json, os, shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import FileResponse
from frictionless import Package, Schema, checks, validate

router = APIRouter()

schemas = ["population", "sanitation", "waste_management", "treatment"]


@router.get("/input/download")
async def download_input_data(session_id: str, file_id: str | None = None):
    path = "data/" + session_id
    project_folder = Path(__file__).parent.parent.parent.parent
    default_folder = "default/"
    if os.path.isdir(project_folder / path):
        if file_id is not None:
            file_path = file_id + ".csv"
            if os.path.isfile(project_folder / path / file_path):
                return FileResponse(
                    path=project_folder / path / file_path,
                    filename="datapackage.json",
                    media_type="text/csv",
                )
            elif os.path.isfile(project_folder / path / default_folder / file_path):
                return FileResponse(
                    path=project_folder / path / default_folder / file_path,
                    filename=file_id + ".csv",
                    media_type="text/csv",
                )
            else:
                raise HTTPException(status_code=500, detail="Invalid File ID provided.")
        else:
            return FileResponse(
                path=project_folder / path / "datapackage.json",
                filename="datapackage.json",
                media_type="text/json",
            )
    else:
        raise HTTPException(status_code=500, detail="Invalid Session ID provided.")


@router.post("/input/upload")
async def upload_input_data(session_id: str, file_id: str, file: UploadFile) -> None:
    path = "data/" + session_id
    project_folder = Path(__file__).parent.parent.parent.parent
    default_folder = "default/"
    if os.path.isdir(project_folder / path):
        print(file_id in schemas)
        if file_id is not None and file_id in schemas:
            file_path = file_id + ".csv"
            try:
                contents = file.file.read()
                with open(project_folder / path / file_path, "wb") as f:
                    f.write(contents)
            except HTTPException:
                raise HTTPException(status_code=500, detail="File was not uploaded.")
            finally:
                file.file.close()
        else:
            raise HTTPException(status_code=500, detail="Invalid File ID provided.")
    else:
        raise HTTPException(status_code=500, detail="Invalid Session ID provided.")


@router.post("/input/generate")
async def generate_input_data_package(session_id: str, gids: str):

    areas = [x.strip(" ") for x in gids.split(",")]

    path = "data/" + session_id
    templates_path = "static/"
    project_folder = Path(__file__).parent.parent.parent.parent

    if os.path.isdir(project_folder / path):
        default_path = "default/"
        schemas_path = "schemas/"
        package_path = "templates/datapackage.json"

        if not os.path.isdir(project_folder / path / default_path):
            os.makedirs(project_folder / path / default_path)
        try:
            resource_descriptions = []
            for schema in schemas:
                filepath = os.path.join(
                    project_folder / path / default_path,
                    schema + ".csv",
                )
                f = open(filepath, "w")
                f.close()
                # schema_name = schema +".json"
                # shutil.copyfile(project_folder / templates_path / schemas_path / schema_name, project_folder / path / schemas_path / schema_name)
                file_name = schema + ".csv"
                description = {
                    "name": schema,
                    "type": "table",
                    "path": str(
                        Path(
                            project_folder / path / default_path / file_name,
                        ).relative_to(project_folder / path),
                    ),
                    "format": "csv",
                    "mediatype": "text/csv",
                    "scheme": "file",
                }
                resource_descriptions.append(description)
            resource_input = {"resources": resource_descriptions}
            package = Package(*resource_input)
            package.to_json(str(project_folder / path / "datapackage.json"))
            # for r in package.resources:
            #     r.schema = str(Path(project_folder / path / schemas_path / schema_name).relative_to(project_folder / path))
        except:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate input data template files.",
            )
    else:
        raise HTTPException(status_code=500, detail="Invalid Session ID provided.")
    with open(str(project_folder / path / "datapackage.json")) as f:
        d = json.load(f)
        return d


@router.get("/input/validate")
async def validate_input_data_package(session_id: str, file_id: str) -> list:

    path = "data/" + session_id
    templates_path = "static/"
    schemas_path = "schemas/"
    project_folder = Path(__file__).parent.parent.parent.parent

    if not os.path.isdir(project_folder / path / schemas_path):
        shutil.copytree(
            project_folder / templates_path / schemas_path,
            project_folder / path / schemas_path,
        )

    if file_id not in schemas:
        raise HTTPException(status_code=500, detail="Invalid File ID provided.")

    if os.path.isdir(project_folder / path):
        package_path = str(project_folder / path)
        package = Package(*package_path)
        resource = package.get_resource(file_id)
        resource_schema = resource.name + ".json"

        # resource.schema = Schema(schema_path)
        schema_path = str(
            Path(project_folder / path / schemas_path / resource_schema).relative_to(
                project_folder / path,
            ),
        )
        resource.schema = Schema(*schema_path)

        if resource.name == "sanitation":
            report = validate(
                resource,
                checks=[
                    checks.row_constraint(
                        formula="flushSewer_rur + flushSeptic_rur + flushPit_rur + flushOpen_rur + flushUnknown_rur + pitSlab_rur + pitNoSlab_rur + compostingToilet_rur + bucketLatrine_rur + containerBased_rur + hangingToilet_rur + openDefecation_rur + other_rur == 1",
                    ),
                    checks.row_constraint(
                        formula="flushSewer_urb + flushSeptic_urb + flushPit_urb + flushOpen_urb + flushUnknown_urb + pitSlab_urb + pitNoSlab_urb + compostingToilet_urb + bucketLatrine_urb + containerBased_urb + hangingToilet_urb + openDefecation_urb + other_urb == 1",
                    ),
                ],
            )
        elif resource.name == "waste_management":
            report = validate(resource)
        else:
            report = validate(resource)
    else:
        raise HTTPException(status_code=500, detail="Invalid Session ID provided.")
    # print(type(report.flatten(['fieldName', 'fieldNumber', 'rowNumber', 'type', 'message'])))
    errors = report.flatten(
        ["fieldName", "fieldNumber", "rowNumber", "type", "message"],
    )
    normalized_errors = []
    for err in errors:
        norm = {
            "field": err[0],
            "col": err[1],
            "row": err[2],
            "type": err[3],
            "msg": err[4],
        }
        normalized_errors.append(norm)
    return normalized_errors
