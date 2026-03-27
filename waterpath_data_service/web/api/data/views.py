import json, logging, os, shutil, httpx, pandas as pd, io
import zipfile
from pathlib import Path
from waterpath_data_service.settings import settings
from waterpath_data_service.services.geodata import geonames, shapefile, resample_raster, geofilter
from waterpath_data_service.services.prepare_spatial import prepare_spatial_inputs
from waterpath_data_service.services.projections import (
    generate_baseline_csv_projection,
    generate_population_isoraster,
    update_human_emissions_population,
    update_isodata_projected_variables,
    fetch_treatment_fractions_csv,
    fetch_treatment_future_csv,
    fetch_livestock_future_csv,
    fetch_sanitation_projection,
    fetch_assumptions,
    read_schema_field_names,
)
from waterpath_data_service.services.temperature import (
    generate_temperature_tif,
    _baseline_temperature_path,
)
from waterpath_data_service.services.livestock import (
    generate_livestock_tabular_inputs,
    generate_livestock_projection_rasters,
    _load_iso_gid_mapping,
    _build_livestock_zone_template,
)
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from frictionless import Package, Schema, Resource, checks, validate

router = APIRouter()
logger = logging.getLogger(__name__)

# Root directory for user data (session packages).  Controlled by
# WATERPATH_DATA_SERVICE_DATA_DIR; defaults to the bundled data/ folder.
_DATA_DIR: Path = settings.data_dir

# Static assets (schemas, look-up tables) – always bundled with the container image.
_STATIC_DIR: Path = Path(__file__).parent.parent.parent.parent / "static"

# Field sets read from the livestock schema files, used to constrain which
# columns from livestock_future.csv are applied to each output CSV.
def _load_livestock_tabular_schema_fields() -> dict[str, set[str]]:
    schemas_dir = _STATIC_DIR / "schemas"
    mapping = {
        "manure_fractions": "livestock_manure_fractions.json",
        "production_systems": "livestock_production_systems.json",
    }
    result: dict[str, set[str]] = {}
    for key, fname in mapping.items():
        path = schemas_dir / fname
        if path.is_file():
            result[key] = read_schema_field_names(path)
    return result

# Maps livestock sub-schema names to generate_livestock_projection_rasters kwargs.
# Use schema=livestock for all outputs; use a sub-schema to restrict what is written.
_LIVESTOCK_SUB_SCHEMA_PARAMS: dict[str, dict] = {
    "livestock_isodata":            {"generate_rasters": True,  "tabular_outputs": set()},
    "livestock_manure_fractions":   {"generate_rasters": False, "tabular_outputs": {"manure_fractions"}},
    "livestock_production_systems": {"generate_rasters": False, "tabular_outputs": {"production_systems"}},
    "livestock_manure_management":  {"generate_rasters": False, "tabular_outputs": {"manure_management"}},
}

schemas = ["population", "sanitation", "treatment"]
livestock_schema_files = [
    "livestock_production_systems.json",
    "livestock_manure_fractions.json",
    "livestock_manure_management.json",
    "livestock_isodata.json",
]

population_cols = {
    "name": "subarea",
    "iso": "iso",
    "alpha3": "gid",
    "totalPopulation": "population",
    "fractionUrban": "fraction_urban_pop",
    "fractionUnderFive": "fraction_pop_under5",
    "hdi": "hdi",
}


def _get_session_default_dir(session_id: str) -> Path:
    """Return the session's baseline directory."""

    return _DATA_DIR / session_id / "baseline"


def ensure_human_emissions_csv(session_id: str) -> Path:
    """Merge population.csv + sanitation.csv into human_emissions/isodata.csv.

    Returns the path to the written isodata.csv.
    """

    default_dir = _get_session_default_dir(session_id)
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
    session_dir = _DATA_DIR / session_id
    default_folder = "baseline/"
    if os.path.isdir(session_dir):
        if file_id is not None:
            file_path = file_id + ".csv"
            human_emissions_dir = session_dir / default_folder / "human_emissions"
            if os.path.isfile(session_dir / file_path):
                return FileResponse(
                    path=session_dir / file_path,
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

                default_dir = _get_session_default_dir(session_id)
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

                    # Add livestock_emissions/ folder if present
                    livestock_emissions_dir = default_dir / "livestock_emissions"
                    if livestock_emissions_dir.is_dir():
                        zipf.writestr(zipfile.ZipInfo("livestock_emissions/"), "")
                        for item in livestock_emissions_dir.rglob("*"):
                            if item.is_file():
                                rel = item.relative_to(livestock_emissions_dir).as_posix()
                                zipf.write(item, arcname=f"livestock_emissions/{rel}")

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
    session_dir = _DATA_DIR / session_id
    default_folder = "baseline/"
    if os.path.isdir(session_dir):
        print(file_id in schemas)
        if file_id is not None and file_id in schemas:
            file_path = file_id + ".csv"
            try:
                contents = file.file.read()
                with open(session_dir / file_path, "wb") as f:
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
        description=(
            "Which schema to generate projections for. "
            "Human emissions: population, sanitation, treatment. "
            "Livestock (all outputs): livestock. "
            "Livestock sub-schemas: livestock_isodata (animal heads rasters only), "
            "livestock_manure_fractions, livestock_production_systems, livestock_manure_management. "
            "Use 'all' to generate population, sanitation, and treatment together."
        ),
        examples=["population"],
    ),
    year: int = Query(..., description="Projection year", examples=[2025, 2030, 2050, 2100]),
    ssp: str = Query(..., description="SSP scenario (SSP1..SSP5)", examples=["SSP1"]),
):
    """Generate projected data for a given schema, year, and SSP scenario."""

    schema_norm = schema.strip().lower()
    allowed_schema = {
        "population", "sanitation", "treatment", "all",
        "livestock", "livestock_isodata", "livestock_manure_fractions",
        "livestock_production_systems", "livestock_manure_management",
    }
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

    session_dir = _DATA_DIR / session_id

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

    static_data_dir = _STATIC_DIR / "data"

    # Detect admin level from baseline isodata to drive projection logic.
    _baseline_pop_csv = session_dir / "baseline" / "human_emissions" / "population.csv"
    _is_country_level: bool = True
    if _baseline_pop_csv.is_file():
        try:
            _bl_df = pd.read_csv(_baseline_pop_csv)
            _gid_vals = _bl_df.get("gid", _bl_df.get("alpha3", pd.Series(dtype=str)))
            _is_country_level = bool((_gid_vals.astype(str).str.len() <= 3).all())
        except Exception:
            pass

    schemas_to_generate = ["population", "sanitation", "treatment"] if schema_norm == "all" else [schema_norm]
    schema_results: list[dict] = []

    for _schema in schemas_to_generate:
        if _schema == "population":
            out_tif = generate_population_isoraster(
                session_dir=session_dir,
                scenario_dir=scenario_dir,
                static_data_dir=static_data_dir,
                ssp=ssp_norm,
                year=year,
            )

            shapefile_path = session_dir / "baseline" / "geodata" / "geodata.shp"
            if not shapefile_path.is_file():
                shapefile_path = session_dir / "geodata" / "geodata.shp"

            update_human_emissions_population(
                human_emissions_path=scenario_human_emissions_path,
                isoraster_path=out_tif,
                scenario_dir=scenario_dir,
                shapefile_path=shapefile_path,
            )

            assumptions = await update_isodata_projected_variables(
                isodata_path=scenario_human_emissions_path,
                ssp=ssp_norm,
                year=year,
                is_country_level=_is_country_level,
            )
            schema_results.append({
                "schema": "population",
                "isoraster_tif": str(out_tif),
                "assumptions": assumptions,
            })

        elif _schema == "sanitation":
            try:
                _iso_df = pd.read_csv(scenario_human_emissions_path)
                _gid_col = next((c for c in ["gid", "alpha3", "iso_country"] if c in _iso_df.columns), None)
                if _gid_col is None:
                    raise ValueError("No GID column found in isodata.csv.")
                _alpha3_list = _iso_df[_gid_col].astype(str).str[:3].unique().tolist()

                san_proj_df = await fetch_sanitation_projection(_alpha3_list, ssp_norm, year)

                # Sanitation future data is keyed by alpha3.  For sub-national rows the
                # parent country is in iso_country; for country-level rows gid IS alpha3.
                _join_src = "iso_country" if "iso_country" in _iso_df.columns else _gid_col
                _iso_df["_alpha3_join"] = _iso_df[_join_src].astype(str).str[:3]
                _san_map = san_proj_df.set_index("alpha3")
                for col in _san_map.columns:
                    col_map = _san_map[col].to_dict()
                    updated = _iso_df["_alpha3_join"].map(col_map)
                    if col in _iso_df.columns:
                        _iso_df[col] = updated.combine_first(_iso_df[col])
                    else:
                        _iso_df[col] = updated
                _iso_df = _iso_df.drop(columns=["_alpha3_join"])
                _iso_df.to_csv(scenario_human_emissions_path, index=False)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to generate sanitation projection: {exc}")

            assumptions = await fetch_assumptions(["sanitation"])
            schema_results.append({
                "schema": "sanitation",
                "assumptions": assumptions,
            })

        elif _schema == "treatment":
            try:
                _iso_df = pd.read_csv(scenario_human_emissions_path)
                _gid_col = next((c for c in ["gid", "alpha3", "iso_country"] if c in _iso_df.columns), None)
                if _gid_col is None:
                    raise ValueError("No GID column found in isodata.csv.")
                _alpha3_list = _iso_df[_gid_col].astype(str).str[:3].unique().tolist()
                treatment_df = await fetch_treatment_future_csv(_alpha3_list, ssp_norm, year)
                treatment_df = treatment_df.rename(columns={"alpha3": "gid"})
                out_csv = scenario_dir / "treatment.csv"
                treatment_df.to_csv(out_csv, index=False)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to generate treatment projection: {exc}")

            assumptions = await fetch_assumptions(["treatment_fractions"])
            schema_results.append({
                "schema": "treatment",
                "projected_csv": str(out_csv),
                "assumptions": assumptions,
            })

        elif _schema == "livestock" or _schema.startswith("livestock_"):
            baseline_livestock_dir = session_dir / "baseline" / "livestock_emissions"
            try:
                # Auto-generate baseline livestock inputs from isodata if not present.
                if not (baseline_livestock_dir / "animals").is_dir():
                    logger.info(
                        "Baseline livestock not found for session %s — generating from isodata.",
                        session_id,
                    )
                    generate_livestock_tabular_inputs(
                        session_dir=session_dir,
                        static_data_dir=static_data_dir,
                    )

                _iso_df = pd.read_csv(scenario_human_emissions_path)
                _gid_col = next((c for c in ["gid", "alpha3", "iso_country"] if c in _iso_df.columns), None)
                if _gid_col is None:
                    raise ValueError("No GID column found in isodata.csv.")
                _alpha3_list = _iso_df[_gid_col].astype(str).str[:3].unique().tolist()

                livestock_future_df = await fetch_livestock_future_csv(_alpha3_list, ssp_norm, year)
                _mapping = _load_iso_gid_mapping(session_dir)
                _zone_idx, _valid_mask, _zone_profile = _build_livestock_zone_template(
                    session_dir, static_data_dir, _mapping
                )
                ls_result = generate_livestock_projection_rasters(
                    baseline_livestock_dir=baseline_livestock_dir,
                    output_dir=scenario_dir / "livestock_emissions",
                    livestock_future_df=livestock_future_df,
                    mapping=_mapping,
                    zone_idx=_zone_idx,
                    valid_mask=_valid_mask,
                    zone_profile=_zone_profile,
                    tabular_schema_fields=_load_livestock_tabular_schema_fields(),
                    **_LIVESTOCK_SUB_SCHEMA_PARAMS.get(_schema, {}),
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to generate livestock projection: {exc}")

            schema_results.append({
                "schema": _schema,
                "output_dir": ls_result["output_dir"],
                "animal_heads": ls_result.get("animal_heads", {}),
            })

    # Write summary.json to the scenario directory.
    summary_data = {
        "session_id": session_id,
        "ssp": ssp_norm,
        "year": year,
        "schemas": schema_results,
    }
    with open(scenario_dir / "summary.json", "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf, indent=2)

    if schema_norm == "all":
        return {
            "session_id": session_id,
            "year": year,
            "ssp": ssp_norm,
            "scenario_isodata_csv": str(scenario_human_emissions_path),
            "status": "written",
            "schemas": schema_results,
        }

    result = schema_results[0]
    return {
        "session_id": session_id,
        "year": year,
        "ssp": ssp_norm,
        "scenario_isodata_csv": str(scenario_human_emissions_path),
        "status": "written",
        **result,
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
    session_dir = _DATA_DIR / session_id

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
        description=(
            "Which schema to project. "
            "Human emissions: population, sanitation, treatment. "
            "Livestock (all outputs): livestock. "
            "Livestock sub-schemas: livestock_isodata (animal heads rasters only), "
            "livestock_manure_fractions, livestock_production_systems, livestock_manure_management. "
            "Use 'all' to generate population, sanitation, treatment, and livestock together."
        ),
        examples=["population"],
    ),
    year: int = Query(..., description="Projection year", examples=[2025, 2030, 2050, 2100]),
    ssp: str = Query(..., description="SSP scenario (SSP1..SSP5)", examples=["SSP3"]),
    file: UploadFile = File(..., description="Baseline CSV (must contain gid/alpha3 + population columns)"),
):
    """Generate a projected scenario from an uploaded baseline CSV.

    Returns an in-memory zip archive containing:
    - ``isodata.csv``       – updated with projected values for all requested schemas
    - ``treatment.csv``     – projected treatment fractions             (treatment / all only)
    - ``isoraster.tif``     – zone-index raster clipped to the study area  (population / all only)
    - ``pop_urban.tif``     – projected urban population raster             (population / all only)
    - ``pop_rural.tif``     – projected rural population raster             (population / all only)
    - ``summary.json``      – per-area statistics and assumptions

    No data is stored on the server.
    """
    import tempfile

    schema_norm = schema.strip().lower()
    _allowed_dl = {
        "population", "sanitation", "treatment", "all",
        "livestock", "livestock_isodata", "livestock_manure_fractions",
        "livestock_production_systems", "livestock_manure_management",
    }
    if schema_norm not in _allowed_dl:
        raise HTTPException(status_code=422, detail=f"Invalid schema. Allowed: {sorted(_allowed_dl)}")

    allowed_years = {2025, 2030, 2050, 2100}
    if year not in allowed_years:
        raise HTTPException(status_code=422, detail=f"Invalid year. Allowed: {sorted(allowed_years)}")

    ssp_norm = ssp.strip().upper()
    if ssp_norm not in {f"SSP{i}" for i in range(1, 6)}:
        raise HTTPException(status_code=422, detail="Invalid SSP. Allowed: SSP1..SSP5")

    static_data_dir = _STATIC_DIR / "data"

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

        all_assumptions: list = []
        schema_entries: list = []

        # Detect admin level: 3-char gids are country codes; longer gids are sub-national.
        is_country_level = all(len(str(g)) <= 3 for g in gids_list)
        alpha3_list = [str(g)[:3] for g in gids_list]
        schemas_to_generate = ["population", "sanitation", "treatment", "livestock"] if schema_norm == "all" else [schema_norm]

        if "population" in schemas_to_generate:
            from waterpath_data_service.services.projections import (
                _population_tif_path,
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

            try:
                pop_assumptions = await update_isodata_projected_variables(
                    isodata_path=projected_csv_path,
                    ssp=ssp_norm,
                    year=year,
                    is_country_level=is_country_level,
                )
                all_assumptions.extend(pop_assumptions)
            except Exception as exc:
                logger.warning("update_isodata_projected_variables failed: %s", exc)

        if "sanitation" in schemas_to_generate:
            try:
                san_proj_df = await fetch_sanitation_projection(alpha3_list, ssp_norm, year)
                _san_df = pd.read_csv(projected_csv_path)
                _join_src = "iso_country" if "iso_country" in _san_df.columns else gid_col
                _san_df["_alpha3_join"] = _san_df[_join_src].astype(str).str[:3]
                _san_map = san_proj_df.set_index("alpha3")
                for col in _san_map.columns:
                    col_map = _san_map[col].to_dict()
                    updated = _san_df["_alpha3_join"].map(col_map)
                    if col in _san_df.columns:
                        _san_df[col] = updated.combine_first(_san_df[col])
                    else:
                        _san_df[col] = updated
                _san_df = _san_df.drop(columns=["_alpha3_join"])
                _san_df.to_csv(projected_csv_path, index=False)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to fetch sanitation projection: {exc}")
            san_assumptions = await fetch_assumptions(["sanitation"])
            all_assumptions.extend(san_assumptions)

        if "treatment" in schemas_to_generate:
            try:
                treatment_df = await fetch_treatment_future_csv(alpha3_list, ssp_norm, year)
                treatment_df = treatment_df.rename(columns={"alpha3": "gid"})
                treatment_csv_path = scenario_dir / "treatment.csv"
                treatment_df.to_csv(treatment_csv_path, index=False)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to fetch treatment projections: {exc}")
            treat_assumptions = await fetch_assumptions(["treatment_fractions"])
            all_assumptions.extend(treat_assumptions)

        _livestock_schema = next(
            (s for s in schemas_to_generate if s == "livestock" or s.startswith("livestock_")), None
        )
        if _livestock_schema is not None:
            _ls_sub_params = _LIVESTOCK_SUB_SCHEMA_PARAMS.get(_livestock_schema, {})
            try:
                # Write population CSV to the path expected by livestock helpers.
                _baseline_dir = tmp_path / "baseline"
                (_baseline_dir / "human_emissions").mkdir(parents=True, exist_ok=True)
                _pop_df = baseline_df.copy()
                if "iso" not in _pop_df.columns:
                    _gid_order = {g: i + 1 for i, g in enumerate(_pop_df[gid_col].astype(str).unique())}
                    _pop_df["iso"] = _pop_df[gid_col].astype(str).map(_gid_order)
                if "gid" not in _pop_df.columns:
                    _pop_df = _pop_df.rename(columns={gid_col: "gid"})
                _pop_df.to_csv(_baseline_dir / "human_emissions" / "population.csv", index=False)

                # Copy shapefile to baseline/geodata/ so livestock helpers find it.
                _gd_src = tmp_path / "geodata"
                _gd_dst = _baseline_dir / "geodata"
                if _gd_src.is_dir() and not _gd_dst.exists():
                    shutil.copytree(_gd_src, _gd_dst)

                _ls_mapping = _load_iso_gid_mapping(tmp_path)
                # Generate baseline livestock heads TIFs into tmp_path/baseline/livestock_emissions/
                generate_livestock_tabular_inputs(tmp_path, static_data_dir)

                livestock_future_df_dl = await fetch_livestock_future_csv(alpha3_list, ssp_norm, year)
                _ls_zone_idx, _ls_valid_mask, _ls_zone_profile = _build_livestock_zone_template(
                    tmp_path, static_data_dir, _ls_mapping
                )
                ls_result = generate_livestock_projection_rasters(
                    baseline_livestock_dir=_baseline_dir / "livestock_emissions",
                    output_dir=scenario_dir / "livestock_emissions",
                    livestock_future_df=livestock_future_df_dl,
                    mapping=_ls_mapping,
                    zone_idx=_ls_zone_idx,
                    valid_mask=_ls_valid_mask,
                    zone_profile=_ls_zone_profile,
                    tabular_schema_fields=_load_livestock_tabular_schema_fields(),
                    **_ls_sub_params,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Failed to generate livestock projection: {exc}")
            schema_entries.append({
                "schema": _livestock_schema,
                "ssp": ssp_norm,
                "year": year,
                "animal_heads": ls_result.get("animal_heads", {}),
            })

        # Compute per-area statistics.
        projected_df = pd.read_csv(projected_csv_path)
        for _schema in schemas_to_generate:
            if _schema == "population" and "population" in projected_df.columns and "population" in baseline_df.columns:
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
                schema_entries.append({
                    "schema": "population",
                    "ssp": ssp_norm,
                    "year": year,
                    "total_baseline_population": total_baseline,
                    "total_projected_population": total_projected,
                    "total_diff": total_diff,
                    "total_pct_change": round((total_diff / total_baseline * 100) if total_baseline else 0, 2),
                    "n_areas": len(area_rows),
                    "areas": area_rows,
                })
            elif _schema == "livestock" or _schema.startswith("livestock_"):
                # livestock stats are carried from the projection step above.
                ls_entry = next((e for e in schema_entries if e.get("schema") == _schema), None)
                if ls_entry is None:
                    schema_entries.append({"schema": _schema, "ssp": ssp_norm, "year": year})
            else:
                schema_entries.append({
                    "schema": _schema,
                    "ssp": ssp_norm,
                    "year": year,
                    "n_areas": len(projected_df),
                })
        summary = {"schemas": schema_entries, "assumptions": all_assumptions}

        # Build the zip entirely in memory.
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(projected_csv_path, arcname="isodata.csv")
            zipf.writestr("summary.json", json.dumps(summary, indent=2))
            # Add rasters produced by prepare_spatial_inputs (population / all).
            for tif_name in ("isoraster.tif", "pop_urban.tif", "pop_rural.tif"):
                tif_file = scenario_dir / tif_name
                if tif_file.is_file():
                    zipf.write(tif_file, arcname=tif_name)
            # Add treatment.csv if present.
            treatment_csv = scenario_dir / "treatment.csv"
            if treatment_csv.is_file():
                zipf.write(treatment_csv, arcname="treatment.csv")
            # Add projected livestock outputs if present.
            ls_dir = scenario_dir / "livestock_emissions"
            ls_animals_dir = ls_dir / "animals"
            if ls_animals_dir.is_dir():
                for tif_file in ls_animals_dir.glob("*.tif"):
                    zipf.write(tif_file, arcname=f"livestock_emissions/animals/{tif_file.name}")
                for isodata_csv in ls_animals_dir.glob("isodata_*.csv"):
                    zipf.write(isodata_csv, arcname=f"livestock_emissions/animals/{isodata_csv.name}")
            ls_isoraster = ls_dir / "animal_isoraster.tif"
            if ls_isoraster.is_file():
                zipf.write(ls_isoraster, arcname="livestock_emissions/animal_isoraster.tif")
            for ls_csv in ("manure_fractions.csv", "production_systems.csv", "manure_management.csv"):
                ls_csv_file = ls_dir / ls_csv
                if ls_csv_file.is_file():
                    zipf.write(ls_csv_file, arcname=f"livestock_emissions/{ls_csv}")

        zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="projection_{ssp_norm}_{year}_{schema_norm}.zip"'
        },
    )


@router.post("/input/generate")
async def generate_input_data_package(session_id: str, gids: str, include_livestock: bool = False):
    areas = [x.strip() for x in gids.split(",") if x.strip()]

    session_dir = _DATA_DIR / session_id

    if os.path.isdir(session_dir):
        default_path = "baseline/"
        schemas_path = "schemas/"
        # Note: schemas are stored in the session folder so the generated
        # datapackage is self-contained and can be validated later.

        human_emissions_output_path = session_dir / default_path / "human_emissions"
        if not os.path.isdir(session_dir / default_path):
            os.makedirs(session_dir / default_path)
        os.makedirs(human_emissions_output_path, exist_ok=True)
        try:
            resources = []
            data = await generateData(areas, session_dir / default_path)

            # Country-level requests use the compact 3-fraction treatment schema;
            # sub-area requests use the high-resolution WWTP point schema, unless
            # no WWTPs were found (in which case generateData falls back to fractions
            # and sets _treatment_used_fractions).
            is_country_level = len(areas[0]) == 3
            treatment_use_fractions = is_country_level or bool(data.get("_treatment_used_fractions"))

            for schema in schemas:
                if schema in data:
                    file_name = schema + ".csv"

                    # Write the generated CSV to the human_emissions subfolder.
                    # `generateData` returns CSV as a string.
                    csv_path = human_emissions_output_path / file_name
                    with open(csv_path, "w", encoding="utf-8", newline="") as f:
                        f.write(data[schema])

                    # Select the correct schema JSON for treatment depending on
                    # whether this is a country-level or sub-area request (or
                    # whether the sub-area fell back to fractions due to 0 WWTPs).
                    if schema == "treatment":
                        src_schema_name = "treatment.json" if treatment_use_fractions else "treatment_high_resolution.json"
                    else:
                        src_schema_name = schema + ".json"
                    dst_schema_name = schema + ".json"  # always written as treatment.json

                    if not os.path.isdir(session_dir / schemas_path):
                        os.mkdir(session_dir / schemas_path)
                    if not os.path.isfile(
                        session_dir / schemas_path / dst_schema_name
                    ):
                        shutil.copyfile(
                            _STATIC_DIR / schemas_path / src_schema_name,
                            session_dir / schemas_path / dst_schema_name,
                        )

                    # Frictionless resources should use paths relative to the package
                    # location (session folder), not absolute OS paths.
                    resource_rel_path = str(Path(default_path) / "human_emissions" / file_name)
                    resource = Resource(name=schema, path=resource_rel_path)
                    resources.append(resource)

            package = Package(resources=resources)
            package.to_json(str(session_dir / "datapackage.json"))

            # Fetch assumptions per schema and write a summary.json alongside
            # the datapackage.  Each schema entry lists only the assumption
            # datasets relevant to that schema so the structure is extensible
            # when more projection schemas are added in the future.
            try:
                use_treatment_fractions = is_country_level or bool(data.get("_treatment_used_fractions"))
                _schema_assumption_keys: dict[str, list[str]] = {
                    "population": ["urbanization", "population", "hdi"],
                    "sanitation": [],
                    "treatment": ["treatment_fractions"] if use_treatment_fractions else [],
                }
                schemas_list = []
                for sname in schemas:
                    keys = _schema_assumption_keys.get(sname, [])
                    if keys:
                        a = await fetch_assumptions(keys)
                        if is_country_level:
                            a = [
                                x for x in a
                                if x.get("admin_level", "all") in ("national", "all", "")
                            ]
                    else:
                        a = []
                    schemas_list.append({"schema": sname, "assumptions": a})
                summary_data = {
                    "session_id": session_id,
                    "schemas": schemas_list,
                }
                with open(str(session_dir / "summary.json"), "w", encoding="utf-8") as sf:
                    json.dump(summary_data, sf, indent=2)
            except Exception as assumption_err:
                logger.warning(
                    "Could not write summary.json for session %s: %s",
                    session_id,
                    assumption_err,
                )

            # Generate baseline spatial rasters from the population CSV + shapefile.
            # Uses the 2025 SSP3 1 km raster as the baseline population surface.
            geodata_shp = session_dir / default_path / "geodata" / "geodata.shp"
            isodata_csv = human_emissions_output_path / "population.csv"
            pop_raster = _STATIC_DIR / "data" / "worldpop_2025" / "global_pop_2025_CN_1km_R2025A_UA_v1.tif"

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
                    logger.warning(
                        "prepare_spatial_inputs failed for session %s: %s", session_id, spatial_err
                    )

            # Clip baseline temperature raster to the study area.
            if geodata_shp.is_file():
                try:
                    _baseline_temp = _baseline_temperature_path(_STATIC_DIR / "data")
                    _template_raster = human_emissions_output_path / "isoraster.tif"
                    generate_temperature_tif(
                        shapefile_path=geodata_shp,
                        source_raster_path=_baseline_temp,
                        out_dir=session_dir / default_path / "livestock_emissions",
                        template_raster_path=_template_raster if _template_raster.is_file() else None,
                    )
                except FileNotFoundError:
                    pass  # vic_watch ASC not present — skip silently
                except Exception as _temp_err:
                    logger.warning(
                        "Baseline temperature clip failed for session %s: %s",
                        session_id, _temp_err,
                    )

            if include_livestock:
                os.makedirs(session_dir / schemas_path, exist_ok=True)
                for schema_name in livestock_schema_files:
                    src = _STATIC_DIR / schemas_path / schema_name
                    dst = session_dir / schemas_path / schema_name
                    if src.is_file() and not dst.is_file():
                        shutil.copyfile(src, dst)
                try:
                    generate_livestock_tabular_inputs(
                        session_dir=session_dir,
                        static_data_dir=_STATIC_DIR / "data",
                    )
                except Exception as _ls_err:
                    logger.warning(
                        "Livestock baseline generation failed for session %s: %s",
                        session_id,
                        _ls_err,
                    )

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error("generate_input_data_package failed:\n%s", tb)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate input data template files: {e}\n{tb}",
            )
    else:
        raise HTTPException(status_code=500, detail="Invalid Session ID provided.")
    with open(str(session_dir / "datapackage.json")) as f:
        d = json.load(f)
    summary_path = session_dir / "summary.json"
    if summary_path.is_file():
        with open(summary_path) as sf:
            d["summary"] = json.load(sf)
    if include_livestock:
        try:
            d["livestock"] = generate_livestock_tabular_inputs(
                session_dir=session_dir,
                static_data_dir=_STATIC_DIR / "data",
            )
        except Exception as livestock_err:
            d["livestock"] = {
                "status": "failed",
                "error": str(livestock_err),
            }
    return d


@router.post("/input/generate/livestock")
async def generate_livestock_input_data(session_id: str):
    session_dir = _DATA_DIR / session_id
    if not session_dir.is_dir():
        raise HTTPException(status_code=500, detail="Invalid Session ID provided.")

    try:
        result = generate_livestock_tabular_inputs(
            session_dir=session_dir,
            static_data_dir=_STATIC_DIR / "data",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate livestock tabular inputs: {exc}")

    schemas_path = "schemas/"
    os.makedirs(session_dir / schemas_path, exist_ok=True)
    for schema_name in livestock_schema_files:
        src = _STATIC_DIR / schemas_path / schema_name
        dst = session_dir / schemas_path / schema_name
        if src.is_file() and not dst.is_file():
            shutil.copyfile(src, dst)

    return {
        "session_id": session_id,
        "status": "written",
        "livestock": result,
    }


@router.get("/input/validate")
async def validate_input_data_package(session_id: str, file_id: str) -> list:

    session_dir = _DATA_DIR / session_id
    schemas_path = "schemas/"

    if not os.path.isdir(session_dir / schemas_path):
        shutil.copytree(
            _STATIC_DIR / schemas_path,
            session_dir / schemas_path,
        )

    if file_id not in schemas:
        raise HTTPException(status_code=500, detail="Invalid File ID provided.")
    
    if os.path.isdir(session_dir):
        package_path = str(session_dir / 'datapackage.json')
        package = Package(source=package_path)
        resource = package.get_resource(file_id)
        resource_schema = resource.name + ".json"

        # resource.schema = Schema(schema_path)
        schema_path = str(
            Path(session_dir / schemas_path / resource_schema).relative_to(
                session_dir,
            ),
        )
        resource.schema = Schema.from_descriptor(session_dir / schemas_path / resource_schema)

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
                _STATIC_DIR / schemas_path / schema_name
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

                    # iso = sequential integer zone index (1-based, in gids list order).
                    # This integer is burned into isoraster.tif as the pixel value so
                    # that downstream code can join the raster back to isodata.csv rows
                    # via the iso column.  gid stays as the string identifier.
                    gid_order = {g: i + 1 for i, g in enumerate(gids)}
                    output["iso"] = output["gid"].map(gid_order)
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

                        # iso = sequential integer zone index (1-based, in gids list order)
                        # matched to the integer pixel values burned into isoraster.tif.
                        # gid remains the string GID identifier.
                        gid_order = {g: i + 1 for i, g in enumerate(gids)}
                        output["iso"] = output["gid"].map(gid_order)
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
                    # Country-level request: fetch treatment fractions from the
                    # WaterPath GitHub repository.
                    fractions_df = await fetch_treatment_fractions_csv(alpha3_gids)
                    # Align with schema: rename alpha3 -> gid
                    output = fractions_df.rename(columns={"alpha3": "gid"})
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

                    if geofiltered_areas.empty:
                        # No WWTPs found in the study area – fall back to the
                        # country-level fraction schema (fetched from GitHub) so
                        # the package is still usable.  Use the parent-country
                        # fractions, expanded once per sub-area GID.
                        country_fracs = await fetch_treatment_fractions_csv(alpha3_gids)
                        # Replicate country fractions for every requested GID so
                        # each sub-area row carries its parent country's fractions.
                        if population_output is not None and not population_output.empty:
                            frac_rows = (
                                population_output[["gid", "iso_country"]]
                                .merge(country_fracs, left_on="iso_country", right_on="alpha3", how="left")
                                .drop(columns=["iso_country", "alpha3"], errors="ignore")
                            )
                        else:
                            # Fallback: one row per unique alpha3 with gid = alpha3
                            frac_rows = country_fracs.rename(columns={"alpha3": "gid"})
                        resObj[schema] = frac_rows.to_csv(index=False, lineterminator="\n")
                        resObj["_treatment_used_fractions"] = True
                    else:
                        resObj[schema] = geofiltered_areas.to_csv(
                            index=False, lineterminator="\n"
                        )

    return resObj
