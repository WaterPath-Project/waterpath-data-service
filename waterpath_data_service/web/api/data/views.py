import json, os, shutil, httpx, pandas as pd, io
import zipfile
from pathlib import Path
from waterpath_data_service.services.geodata import geonames, shapefile, resample_raster, geofilter
from waterpath_data_service.services.prepare_spatial import prepare_spatial_inputs
from waterpath_data_service.services.projections import (
    generate_baseline_csv_projection,
    generate_population_isoraster,
    update_human_emissions_population,
)
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from frictionless import Package, Schema, Resource, checks, validate

router = APIRouter()

schemas = ["population", "sanitation", "treatment"]

population_cols = {
    "name": "subarea",
    "iso": "iso",
    "alpha3": "gid",
    "totalPopulation": "population",
    "fractionUrban": "fraction_urban_pop",
    "fractionUnderFive": "fraction_pop_under5",
    "hdi": "hdi",
}


def _get_session_default_dir(project_folder: Path, session_id: str) -> Path:
    """Return the session's baseline directory."""

    return project_folder / "data" / session_id / "baseline"


def ensure_human_emissions_csv(session_id: str) -> Path:
    """Merge population.csv + sanitation.csv into human_emissions/isodata.csv.

    Returns the path to the written isodata.csv.
    """

    project_folder = Path(__file__).parent.parent.parent.parent
    default_dir = _get_session_default_dir(project_folder, session_id)
    if not default_dir.is_dir():
        raise HTTPException(status_code=500, detail="Invalid Session ID provided.")

    human_emissions_dir = default_dir / "human_emissions"
    out_path = human_emissions_dir / "isodata.csv"

    pop_path = human_emissions_dir / "population.csv"
    san_path = human_emissions_dir / "sanitation.csv"
    if not pop_path.is_file() or not san_path.is_file():
        raise HTTPException(
            status_code=500,
            detail="Baseline input data missing (population.csv/sanitation.csv).",
        )

    pop_df = pd.read_csv(pop_path)
    san_df = pd.read_csv(san_path)

    # Normalize join keys.
    if "gid" not in pop_df.columns:
        raise HTTPException(
            status_code=500,
            detail="Baseline population.csv missing required 'gid' column.",
        )

    # Some baseline sanitation files are keyed by `alpha3` rather than `gid`.
    if "gid" not in san_df.columns:
        if "alpha3" in san_df.columns:
            san_df = san_df.rename(columns={"alpha3": "gid"})
        elif "iso3" in san_df.columns:
            san_df = san_df.rename(columns={"iso3": "gid"})
        else:
            raise HTTPException(
                status_code=500,
                detail="Baseline sanitation.csv missing required 'gid' (or 'alpha3') column.",
            )

    pop_df["gid"] = pop_df["gid"].astype(str).str.strip()
    san_df["gid"] = san_df["gid"].astype(str).str.strip()

    # Avoid accidental row multiplication when sanitation has duplicate keys.
    if san_df["gid"].duplicated().any():
        san_df = san_df.drop_duplicates(subset=["gid"], keep="first")

    # `human_emissions.csv` is the merged baseline table used by the GloWPa model.
    # Join key is `gid` (country/subarea identifier).
    merged = pd.merge(pop_df, san_df, how="left", on="gid")
    merged.to_csv(out_path, sep=",", encoding="utf-8", index=False)
    return out_path


@router.get("/input/download")
async def download_input_data(session_id: str, file_id: str | None = None):
    path = "data/" + session_id
    project_folder = Path(__file__).parent.parent.parent.parent
    default_folder = "baseline/"
    if os.path.isdir(project_folder / path):
        if file_id is not None:
            file_path = file_id + ".csv"
            human_emissions_dir = project_folder / path / default_folder / "human_emissions"
            if os.path.isfile(project_folder / path / file_path):
                return FileResponse(
                    path=project_folder / path / file_path,
                    filename="datapackage.json",
                    media_type="text/csv",
                )
            elif os.path.isfile(human_emissions_dir / file_path):
                return FileResponse(
                    path=human_emissions_dir / file_path,
                    filename=file_id + ".csv",
                    media_type="text/csv",
                )
            else:
                raise HTTPException(status_code=500, detail="Invalid File ID provided.")
        else:
            try:
                # Merge pop + sanitation → human_emissions/isodata.csv
                ensure_human_emissions_csv(session_id)

                default_dir = _get_session_default_dir(project_folder, session_id)
                geodata_dir = default_dir / "geodata"
                human_emissions_dir = default_dir / "human_emissions"
                treatment_path = human_emissions_dir / "treatment.csv"

                if not treatment_path.is_file():
                    raise HTTPException(
                        status_code=500,
                        detail="Baseline input data missing (treatment.csv).",
                    )

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Add geodata/ folder
                    if geodata_dir.is_dir():
                        zipf.writestr(zipfile.ZipInfo("geodata/"), "")
                        for item in geodata_dir.iterdir():
                            if item.is_file():
                                zipf.write(item, arcname=f"geodata/{item.name}")

                    # Add human_emissions/ folder (isodata.csv, treatment.csv, tifs)
                    # Exclude raw source files that have been merged into isodata.csv.
                    _excluded = {"population.csv", "sanitation.csv"}
                    zipf.writestr(zipfile.ZipInfo("human_emissions/"), "")
                    for item in human_emissions_dir.iterdir():
                        if item.is_file() and item.suffix in (".csv", ".tif") and item.name not in _excluded:
                            zipf.write(item, arcname=f"human_emissions/{item.name}")

                zip_buffer.seek(0)

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Input data not generated: {e}")
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="GloWPa_input.zip"'},
            )
    else:
        raise HTTPException(status_code=500, detail="Invalid Session ID provided.")


@router.post("/input/upload")
async def upload_input_data(session_id: str, file_id: str, file: UploadFile) -> None:
    path = "data/" + session_id
    project_folder = Path(__file__).parent.parent.parent.parent
    default_folder = "baseline/"
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


@router.post("/projections/generate")
async def generate_projection_data(
    session_id: str,
    schema: str = Query(
        ...,
        description="Which schema to generate projections for (population, sanitation, treatment)",
        examples=["population"],
    ),
    year: int = Query(..., description="Projection year", examples=[2025, 2030, 2050, 2100]),
    ssp: str = Query(..., description="SSP scenario (SSP1..SSP5)", examples=["SSP1"]),
):
    """Generate projected data for a given schema, year, and SSP scenario."""

    schema_norm = schema.strip().lower()
    allowed_schema = {"population", "sanitation", "treatment"}
    if schema_norm not in allowed_schema:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid schema. Allowed: {sorted(allowed_schema)}",
        )

    allowed_years = {2025, 2030, 2050, 2100}
    if year not in allowed_years:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid year. Allowed: {sorted(allowed_years)}",
        )

    ssp_norm = ssp.strip().upper()
    allowed_ssp = {f"SSP{i}" for i in range(1, 6)}
    if ssp_norm not in allowed_ssp:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid SSP. Allowed: {sorted(allowed_ssp)}",
        )

    project_folder = Path(__file__).parent.parent.parent.parent
    session_dir = project_folder / "data" / session_id

    if not session_dir.is_dir():
        raise HTTPException(status_code=500, detail="Invalid Session ID provided.")

    # Output folder sits alongside `baseline/` under a `scenarios/` container:
    # data/<session_id>/scenarios/<SSP>_<year>/human_emissions.csv
    scenario_folder_name = f"{ssp_norm}_{year}"
    scenario_dir = session_dir / "scenarios" / scenario_folder_name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    # Ensure we always have scenario-local isodata.csv available.
    baseline_human_emissions_path = ensure_human_emissions_csv(session_id)
    scenario_human_emissions_path = scenario_dir / "isodata.csv"
    shutil.copyfile(baseline_human_emissions_path, scenario_human_emissions_path)

    static_data_dir = project_folder / "static" / "data"

    if schema_norm == "population":
        out_tif = generate_population_isoraster(
            session_dir=session_dir,
            scenario_dir=scenario_dir,
            static_data_dir=static_data_dir,
            ssp=ssp_norm,
            year=year,
        )
        
        # Update human_emissions.csv with population values calculated from isoraster.tif
        shapefile_path = session_dir / "baseline" / "geodata" / "geodata.shp"
        if not shapefile_path.is_file():
            shapefile_path = session_dir / "geodata" / "geodata.shp"
        
        update_human_emissions_population(
            human_emissions_path=scenario_human_emissions_path,
            isoraster_path=out_tif,
            scenario_dir=scenario_dir,
            shapefile_path=shapefile_path,
        )
        
        return {
            "session_id": session_id,
            "schema": schema_norm,
            "year": year,
            "ssp": ssp_norm,
            "scenario_isodata_csv": str(scenario_human_emissions_path),
            "isoraster_tif": str(out_tif),
            "status": "written",
        }

    # Placeholder behavior for sanitation/treatment: copy baseline CSV into the scenario folder.
    try:
        out_csv = generate_baseline_csv_projection(
            session_dir=session_dir,
            schema=schema_norm,
            scenario_dir=scenario_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "session_id": session_id,
        "schema": schema_norm,
        "year": year,
        "ssp": ssp_norm,
        "scenario_isodata_csv": str(scenario_human_emissions_path),
        "projected_csv": str(out_csv),
        "status": "written",
    }


@router.get("/projections/validate")
async def validate_projection_population(
    session_id: str,
    year: int = Query(..., description="Projection year", examples=[2025, 2030, 2050, 2100]),
    ssp: str = Query(..., description="SSP scenario (SSP1..SSP5)", examples=["SSP1"]),
):
    """Compare baseline vs projected population per area for a given SSP/year scenario.

    Returns per-area rows with baseline population, projected population, absolute
    difference, and percentage change — making it easy to spot implausible values.
    Also returns aggregate totals for a quick sanity check.
    """
    ssp_norm = ssp.strip().upper()
    project_folder = Path(__file__).parent.parent.parent.parent
    session_dir = project_folder / "data" / session_id

    if not session_dir.is_dir():
        raise HTTPException(status_code=500, detail="Invalid Session ID provided.")

    baseline_csv = session_dir / "baseline" / "human_emissions" / "isodata.csv"
    scenario_csv = session_dir / "scenarios" / f"{ssp_norm}_{year}" / "isodata.csv"

    if not baseline_csv.is_file():
        raise HTTPException(
            status_code=500,
            detail="Baseline isodata.csv not found. Run /input/generate first.",
        )
    if not scenario_csv.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Projected isodata.csv not found for {ssp_norm}/{year}. Run /projections/generate first.",
        )

    baseline_df = pd.read_csv(baseline_csv)
    scenario_df = pd.read_csv(scenario_csv)

    gid_col = next((c for c in ["gid", "alpha3", "iso_country"] if c in baseline_df.columns), None)
    if gid_col is None:
        raise HTTPException(status_code=500, detail="No GID column found in baseline isodata.csv.")

    name_col = "subarea" if "subarea" in baseline_df.columns else None

    merged = pd.merge(
        baseline_df[[gid_col] + (["subarea"] if name_col else []) + ["population"]],
        scenario_df[[gid_col, "population"]],
        on=gid_col,
        suffixes=("_baseline", "_projected"),
    )

    merged["diff"] = merged["population_projected"] - merged["population_baseline"]
    merged["pct_change"] = (
        (merged["diff"] / merged["population_baseline"].replace(0, float("nan"))) * 100
    ).round(2)

    rows = merged.rename(columns={
        gid_col: "gid",
        "population_baseline": "population_baseline",
        "population_projected": "population_projected",
    }).to_dict(orient="records")

    total_baseline = int(merged["population_baseline"].sum())
    total_projected = int(merged["population_projected"].sum())
    total_diff = total_projected - total_baseline
    total_pct = round((total_diff / total_baseline * 100) if total_baseline else 0, 2)

    return {
        "session_id": session_id,
        "ssp": ssp_norm,
        "year": year,
        "summary": {
            "total_baseline": total_baseline,
            "total_projected": total_projected,
            "total_diff": total_diff,
            "total_pct_change": total_pct,
            "n_areas": len(rows),
        },
        "areas": rows,
    }


@router.post("/projections/download")
async def download_projection(
    schema: str = Query(
        ...,
        description="Which schema to project (population, sanitation, treatment)",
        examples=["population"],
    ),
    year: int = Query(..., description="Projection year", examples=[2025, 2030, 2050, 2100]),
    ssp: str = Query(..., description="SSP scenario (SSP1..SSP5)", examples=["SSP3"]),
    file: UploadFile = File(..., description="Baseline CSV (must contain gid/alpha3 + population columns)"),
):
    """Generate a projected scenario from an uploaded baseline CSV.

    Returns an in-memory zip archive containing:
    - ``isodata.csv``       – updated with projected population values
    - ``isoraster.tif``     – zone-index raster clipped to the study area  (population only)
    - ``pop_urban.tif``     – projected urban population raster             (population only)
    - ``pop_rural.tif``     – projected rural population raster             (population only)
    - ``summary.json``      – per-area and aggregate statistics

    No data is stored on the server.
    """
    import tempfile

    schema_norm = schema.strip().lower()
    if schema_norm not in {"population", "sanitation", "treatment"}:
        raise HTTPException(status_code=422, detail=f"Invalid schema. Allowed: population, sanitation, treatment")

    allowed_years = {2025, 2030, 2050, 2100}
    if year not in allowed_years:
        raise HTTPException(status_code=422, detail=f"Invalid year. Allowed: {sorted(allowed_years)}")

    ssp_norm = ssp.strip().upper()
    if ssp_norm not in {f"SSP{i}" for i in range(1, 6)}:
        raise HTTPException(status_code=422, detail="Invalid SSP. Allowed: SSP1..SSP5")

    project_folder = Path(__file__).parent.parent.parent.parent
    static_data_dir = project_folder / "static" / "data"

    # Read the uploaded CSV.
    raw = await file.read()
    try:
        baseline_df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse uploaded CSV: {exc}")

    gid_col = next((c for c in ["gid", "alpha3", "iso_country"] if c in baseline_df.columns), None)
    if gid_col is None:
        raise HTTPException(status_code=400, detail="CSV must contain a 'gid' or 'alpha3' column.")
    if "population" not in baseline_df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'population' column.")

    gids_list = baseline_df[gid_col].astype(str).str.strip().tolist()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Build shapefile on the fly from the GIDs in the uploaded CSV.
        try:
            shapefile(gids_list, str(tmp_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to build shapefile for provided GIDs: {exc}")
        shp_path = tmp_path / "geodata" / "geodata.shp"

        # Write baseline CSV to temp dir.
        baseline_csv_path = tmp_path / "isodata.csv"
        baseline_df.to_csv(baseline_csv_path, index=False)

        scenario_dir = tmp_path / "scenario"
        scenario_dir.mkdir()
        projected_csv_path = scenario_dir / "isodata.csv"
        shutil.copyfile(baseline_csv_path, projected_csv_path)

        summary: dict = {}

        if schema_norm == "population":
            from waterpath_data_service.services.projections import (
                _population_tif_path,
                update_human_emissions_population,
            )
            try:
                tif_path = _population_tif_path(static_data_dir, ssp_norm, year)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=500, detail=str(exc))

            try:
                paths = prepare_spatial_inputs(
                    geodata_path=str(shp_path),
                    isodata_path=str(baseline_csv_path),
                    pop_raster_path=str(tif_path),
                    out_dir=str(scenario_dir),
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"prepare_spatial_inputs failed: {exc}")

            isoraster_path = Path(paths["isoraster"])

            try:
                update_human_emissions_population(
                    human_emissions_path=projected_csv_path,
                    isoraster_path=isoraster_path,
                    scenario_dir=scenario_dir,
                    shapefile_path=shp_path,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Population update failed: {exc}")

        # Compute per-area and aggregate statistics.
        projected_df = pd.read_csv(projected_csv_path)
        merged_stats = pd.merge(
            baseline_df[[gid_col, "population"]],
            projected_df[[gid_col, "population"]],
            on=gid_col,
            suffixes=("_baseline", "_projected"),
        )
        merged_stats["diff"] = merged_stats["population_projected"] - merged_stats["population_baseline"]
        merged_stats["pct_change"] = (
            (merged_stats["diff"] / merged_stats["population_baseline"].replace(0, float("nan"))) * 100
        ).round(2)

        area_rows = merged_stats.rename(columns={gid_col: "gid"}).to_dict(orient="records")
        total_baseline = int(merged_stats["population_baseline"].sum())
        total_projected = int(merged_stats["population_projected"].sum())
        total_diff = total_projected - total_baseline
        summary = {
            "schema": schema_norm,
            "ssp": ssp_norm,
            "year": year,
            "total_baseline_population": total_baseline,
            "total_projected_population": total_projected,
            "total_diff": total_diff,
            "total_pct_change": round((total_diff / total_baseline * 100) if total_baseline else 0, 2),
            "n_areas": len(area_rows),
            "areas": area_rows,
        }

        # Build the zip entirely in memory.
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(projected_csv_path, arcname="isodata.csv")
            zipf.writestr("summary.json", json.dumps(summary, indent=2))
            # Add rasters produced by prepare_spatial_inputs (population schema only).
            for tif_name in ("isoraster.tif", "pop_urban.tif", "pop_rural.tif"):
                tif_file = scenario_dir / tif_name
                if tif_file.is_file():
                    zipf.write(tif_file, arcname=tif_name)

        zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="projection_{ssp_norm}_{year}_{schema_norm}.zip"'
        },
    )


@router.post("/input/generate")
async def generate_input_data_package(session_id: str, gids: str):
    areas = [x.strip(" ") for x in gids.split(",")]

    path = "data/" + session_id
    templates_path = "static/"
    project_folder = Path(__file__).parent.parent.parent.parent

    if os.path.isdir(project_folder / path):
        default_path = "baseline/"
        schemas_path = "schemas/"
        # Note: schemas are stored in the session folder so the generated
        # datapackage is self-contained and can be validated later.

        human_emissions_output_path = project_folder / path / default_path / "human_emissions"
        if not os.path.isdir(project_folder / path / default_path):
            os.makedirs(project_folder / path / default_path)
        os.makedirs(human_emissions_output_path, exist_ok=True)
        try:
            resources = []
            data = await generateData(areas, project_folder / path / default_path)

            # Country-level requests use the compact 3-fraction treatment schema;
            # sub-area requests use the high-resolution WWTP point schema.
            is_country_level = len(areas[0]) == 3

            for schema in schemas:
                if schema in data:
                    file_name = schema + ".csv"

                    # Write the generated CSV to the human_emissions subfolder.
                    # `generateData` returns CSV as a string.
                    csv_path = human_emissions_output_path / file_name
                    with open(csv_path, "w", encoding="utf-8", newline="") as f:
                        f.write(data[schema])

                    # Select the correct schema JSON for treatment depending on
                    # whether this is a country-level or sub-area request.
                    if schema == "treatment":
                        src_schema_name = "treatment.json" if is_country_level else "treatment_high_resolution.json"
                    else:
                        src_schema_name = schema + ".json"
                    dst_schema_name = schema + ".json"  # always written as treatment.json

                    if not os.path.isdir(project_folder / path / schemas_path):
                        os.mkdir(project_folder / path / schemas_path)
                    if not os.path.isfile(
                        project_folder / path / schemas_path / dst_schema_name
                    ):
                        shutil.copyfile(
                            project_folder / templates_path / schemas_path / src_schema_name,
                            project_folder / path / schemas_path / dst_schema_name,
                        )

                    # Frictionless resources should use paths relative to the package
                    # location (session folder), not absolute OS paths.
                    resource_rel_path = str(Path(default_path) / "human_emissions" / file_name)
                    resource = Resource(name=schema, path=resource_rel_path)
                    resources.append(resource)

            package = Package(resources=resources)
            package.to_json(str(project_folder / path / "datapackage.json"))

            # Generate baseline spatial rasters from the population CSV + shapefile.
            # Uses the 2025 SSP3 1 km raster as the baseline population surface.
            geodata_shp = project_folder / path / default_path / "geodata" / "geodata.shp"
            isodata_csv = human_emissions_output_path / "population.csv"
            pop_raster = project_folder / "static" / "data" / "global_pop_2025_CN_1km_R2025A_UA_v1.tif"

            if geodata_shp.is_file() and isodata_csv.is_file() and pop_raster.is_file():
                try:
                    prepare_spatial_inputs(
                        geodata_path=str(geodata_shp),
                        isodata_path=str(isodata_csv),
                        pop_raster_path=str(pop_raster),
                        out_dir=str(human_emissions_output_path),
                    )
                except Exception as spatial_err:
                    # Non-fatal: CSV generation already succeeded; log and continue.
                    import logging
                    logging.getLogger(__name__).warning(
                        "prepare_spatial_inputs failed for session %s: %s", path, spatial_err
                    )

        except Exception as e:
            import traceback, logging
            tb = traceback.format_exc()
            logging.getLogger(__name__).error(
                "generate_input_data_package failed:\n%s", tb
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate input data template files: {e}\n{tb}",
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
        package_path = str(project_folder / path / 'datapackage.json')
        package = Package(source=package_path)
        resource = package.get_resource(file_id)
        resource_schema = resource.name + ".json"

        # resource.schema = Schema(schema_path)
        schema_path = str(
            Path(project_folder / path / schemas_path / resource_schema).relative_to(
                project_folder / path,
            ),
        )
        resource.schema = Schema.from_descriptor(project_folder / path / schemas_path / resource_schema)

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


async def generateData(gids, path):
    """Generate schema-specific CSV payloads for the requested areas.

    Returns a dict mapping schema name -> CSV text.

    Notes:
    - `gids` can be either country codes (alpha3, length 3) or GADM-style IDs
      for sub-areas (contain '.'), e.g. "UGA.1_1".
    - `shapefile(...)` is called for its side effects (it writes the boundary
      files under the session folder, used by /input/download zipping).
    """
    resObj = dict()
    schemas_path = "schemas/"
    templates_path = "static/"
    project_folder = Path(__file__).parent.parent.parent.parent

    # Side effect: writes/ensures session geodata exists.
    shapefile(gids, path)

    # `population_output` is reused by sanitation for sub-area requests.
    population_output = None

    # try:
    #     rasterfile = resample_raster(path)
    # except Exception as e:
    #     print(e)
    async with httpx.AsyncClient() as client:
        for schema in schemas:
            schema_name = schema + ".json"

            # The schema JSON is used to enforce column ordering and naming.
            schema_descriptor_path = (
                project_folder / templates_path / schemas_path / schema_name
            )

            if schema == "population":
                if len(gids[0]) == 3:
                    # Admin level 0: gids are ISO alpha-3 country codes.
                    response = await client.get(
                        "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data/refs/heads/main/world_population/data/world-population.csv"
                    )
                    df = pd.read_table(io.StringIO(response.text), sep=",")
                    output = df.loc[df["alpha3"].isin(gids)]

                    response_hdi = await client.get(
                        "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data/refs/heads/main/hdi/data/hdi.csv"
                    )
                    df_hdi = pd.read_table(io.StringIO(response_hdi.text), sep=",")

                    with open(schema_descriptor_path, encoding="utf-8") as f:
                        d = json.load(f)
                        fields = [item["name"] for item in d["fields"]]

                    output = (
                        pd.merge(
                            output,
                            df_hdi,
                            left_on="alpha3",
                            right_on="alpha3",
                            how="left",
                        )
                        .rename(columns=population_cols)
                        .reindex(columns=fields)
                    )

                    # For historical reasons, iso_country is the same alpha3 used in gid.
                    output["iso_country"] = output["gid"]

                    # Requested behavior: fill `iso` with the same identifier as `gid`.
                    # (Some legacy datasets use numeric ISO codes here; this forces a
                    # consistent string identifier for generated packages.)
                    output["iso"] = output["gid"]
                    population_output = output
                    resObj[schema] = output.to_csv(index=False, lineterminator="\n")

                else:
                    # Admin level > 0: gids are GADM area IDs (contain '.'), e.g. UGA.1_1.
                    # `level` is derived from the number of '.' characters.
                    if gids[0].count(".") > 0:
                        level = str(gids[0].count("."))
                        names = geonames(",".join(gids), int(level))

                        response = await client.get(
                            "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data/refs/heads/main/world_admin_units_urbanisation_degree/data/world_urbanisation_level"
                            + level
                            + ".csv"
                        )
                        df = pd.read_table(io.StringIO(response.text), sep=",")
                        output = df.loc[df["gid"].isin(gids)]

                        # fractionUnderFive comes from the country-level dataset.
                        response = await client.get(
                            "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data/refs/heads/main/world_population/data/world-population.csv"
                        )
                        wp_df = pd.read_table(io.StringIO(response.text), sep=",")
                        wp_output = wp_df.loc[
                            wp_df["alpha3"].isin(output["alpha3"].tolist())
                        ]

                        response_hdi = await client.get(
                            "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data/refs/heads/main/hdi/data/hdi.csv"
                        )
                        df_hdi = pd.read_table(io.StringIO(response_hdi.text), sep=",")

                        output = pd.merge(
                            output,
                            wp_output[["alpha3", "fractionUnderFive"]],
                            left_on="alpha3",
                            right_on="alpha3",
                            how="left",
                        )
                        output = pd.merge(
                            output,
                            df_hdi,
                            left_on="alpha3",
                            right_on="alpha3",
                            how="left",
                        )
                        output = pd.merge(
                            output,
                            names,
                            left_on="gid",
                            right_on="gid",
                            how="left",
                        )

                        with open(schema_descriptor_path, encoding="utf-8") as f:
                            d = json.load(f)
                            fields = [item["name"] for item in d["fields"]]

                        # Important: do NOT overwrite the numeric `iso` column from
                        # the admin-units dataset. It is used in baseline packages.
                        output = (
                            output.reindex(
                                columns=[
                                    "name",
                                    "gid",
                                    "iso",
                                    "alpha3",
                                    "totalPopulation",
                                    "fractionUrban",
                                    "fractionUnderFive",
                                    "hdi",
                                ]
                            )
                            .rename(
                                columns={
                                    "name": "subarea",
                                    "alpha3": "iso_country",
                                    "totalPopulation": "population",
                                    "fractionUrban": "fraction_urban_pop",
                                    "fractionUnderFive": "fraction_pop_under5",
                                    "hdi": "hdi",
                                }
                            )
                            .reindex(columns=fields)
                        )

                        # Requested behavior: fill `iso` with the same identifier as `gid`.
                        output["iso"] = output["gid"]
                        population_output = output
                        resObj[schema] = output.to_csv(
                            index=False, lineterminator="\n"
                        )

            elif schema == "sanitation":
                response = await client.get(
                    "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data/refs/heads/main/jmp_household_surveys/data/sanitation_combined.csv"
                )

                df = pd.read_table(io.StringIO(response.text), sep=",")

                # Keep the original gids list intact: sanitation is keyed by alpha3.
                alpha3_gids = [s[0:3] for s in gids]
                filtered_sanitation = df.loc[df["alpha3"].isin(alpha3_gids)]

                # For sub-areas, create a gid column by joining sanitation (country)
                # with the population sub-area table.
                if (
                    population_output is not None
                    and not population_output.empty
                    and population_output["gid"].iloc[0].count(".") > 0
                ):
                    san_df = pd.merge(
                        filtered_sanitation,
                        population_output,
                        how="left",
                        left_on="alpha3",
                        right_on="iso_country",
                    )
                    san_df.drop(
                        columns=[
                            "alpha3",
                            "iso",
                            "subarea",
                            "hdi",
                            "population",
                            "fraction_urban_pop",
                            "fraction_pop_under5",
                            "iso_country",
                        ],
                        inplace=True,
                        errors="ignore",
                    )
                    san_df = san_df[["gid"] + [c for c in san_df.columns if c != "gid"]]
                    resObj[schema] = san_df.to_csv(index=False, lineterminator="\n")
                else:
                    filtered_sanitation = filtered_sanitation.rename(
                        columns={"alpha3": "gid"}
                    )
                    resObj[schema] = filtered_sanitation.to_csv(
                        index=False, lineterminator="\n"
                    )

            elif schema == "treatment":
                alpha3_gids = [s[0:3] for s in gids]

                if len(gids[0]) == 3:
                    # Country-level request: look up aggregate treatment fractions
                    # from the static lookup table (Van Drecht et al. 2009 / OECD 2016,
                    # penultimate column = 2010 values, format P-S-T /100).
                    fractions_path = project_folder / "static" / "data" / "treatment_fractions.csv"
                    fractions_df = pd.read_csv(fractions_path)
                    output = fractions_df[fractions_df["alpha3"].isin(alpha3_gids)].copy()
                    # Align with schema: rename alpha3 -> gid
                    output = output.rename(columns={"alpha3": "gid"})
                    resObj[schema] = output.to_csv(index=False, lineterminator="\n")
                else:
                    # Sub-area request: spatially filter high-resolution WWTP point data.
                    response = await client.get(
                        "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data/refs/heads/main/hydrosheds_treatment/data/wwtp.csv"
                    )
                    df = pd.read_table(io.StringIO(response.text), sep=",")
                    country_wwtps = df.loc[df["alpha3"].isin(alpha3_gids)]

                    geofiltered_areas = geofilter(country_wwtps, path)
                    if "treatment_type" in geofiltered_areas.columns:
                        geofiltered_areas = geofiltered_areas[
                            geofiltered_areas["treatment_type"].str.lower() != "ponds"
                        ].copy()
                        geofiltered_areas["treatment_type"] = geofiltered_areas["treatment_type"].str.capitalize()
                    resObj[schema] = geofiltered_areas.to_csv(
                        index=False, lineterminator="\n"
                    )

    return resObj
