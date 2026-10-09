# WaterPath Data Service

This project is based on Python and Fast API.

## Poetry

This project uses poetry.

To run the project use this set of commands:

```bash
poetry install
poetry run python -m waterpath_data_service
```

This will start the server on the configured host.

You can find swagger documentation at `/api/docs`.

You can read more about poetry here: https://python-poetry.org/

## Docker

You can start the project with docker using this command:

```bash
docker-compose up --build
```

If you want to develop in docker with autoreload and exposed ports add `-f deploy/docker-compose.dev.yml` to your docker command.
Like this:

```bash
docker-compose -f docker-compose.yml -f deploy/docker-compose.dev.yml --project-directory . up --build
```

This command exposes the web application on port 8000, mounts current directory and enables autoreload.

But you have to rebuild image every time you modify `poetry.lock` or `pyproject.toml` with this command:

```bash
docker-compose build
```

## Project structure

```bash
$ tree "waterpath_data_service"
waterpath_data_service
├── conftest.py  # Fixtures for all tests.
├── __main__.py  # Startup script. Starts uvicorn.
├── services  # Package for different external services such as rabbit or redis etc.
├── settings.py  # Main configuration settings for project.
├── static  # Static content.
├── tests  # Tests for project.
└── web  # Package contains web server. Handlers, startup config.
    ├── api  # Package with all handlers.
    │   └── router.py  # Main router.
    ├── application.py  # FastAPI application configuration.
    └── lifespan.py  # Contains actions to perform on startup and shutdown.
```

## Configuration

This application can be configured with environment variables.

You can create `.env` file in the root directory and place all
environment variables here. 

All environment variables should start with "WATERPATH_DATA_SERVICE_" prefix.

For example if you see in your "waterpath_data_service/settings.py" a variable named like
`random_parameter`, you should provide the "WATERPATH_DATA_SERVICE_RANDOM_PARAMETER" 
variable to configure the value. This behaviour can be changed by overriding `env_prefix` property
in `waterpath_data_service.settings.Settings.Config`.

An example of .env file:
```bash
WATERPATH_DATA_SERVICE_RELOAD="True"
WATERPATH_DATA_SERVICE_PORT="8000"
WATERPATH_DATA_SERVICE_ENVIRONMENT="dev"
```

You can read more about BaseSettings class here: https://pydantic-docs.helpmanual.io/usage/settings/

## Pre-commit

To install pre-commit simply run inside the shell:
```bash
pre-commit install
```

pre-commit is very useful to check your code before publishing it.
It's configured using .pre-commit-config.yaml file.

By default it runs:
* black (formats your code);
* mypy (validates types);
* ruff (spots possible bugs);


You can read more about pre-commit here: https://pre-commit.com/


## Hydrology Inputs

Generated hydrology now stays on its native **0.5-degree grid** by default.
The existing emissions isoraster is the authoritative model grid and is never
coarsened to match hydrology. Preparation leaves compatible rasters untouched and
crops or resamples incompatible livestock, temperature and continuous inputs.
Hydrology remains on its own internally consistent grid, including its native D8
network and hydraulic fields. Preparation does not fabricate routing, resample
hydrology, or coarsen the isoraster. Downloads export stored inputs without requiring a shared
grid, including full, cached and scenario archives. The downstream tool or user is
responsible for two-grid coupling and model compatibility; a successful download
does not certify that the package is directly runnable in GloWPa.

Population and livestock head-count rasters use density/area resampling; cropping
does not redistribute excluded counts back into the study area. Categorical
rasters use mode when coarsening and nearest-neighbour
when refining. Continuous fields use average when coarsening and bilinear when
refining. Routing codes, cycles, downstream accumulation, and consistency of the
hydrology grid are checked without changing the native hydrology inputs.
Geometry compatibility alone is not scientific validation of a model run.

To substitute complete custom hydrology, use the existing upload endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/api/data/input/upload?session_id=case&file_id=hydrology" -F "file=@hydrology.zip"
curl -X POST "http://127.0.0.1:8000/api/data/input/upload?session_id=case&file_id=hydrology&ssp=SSP3&year=2050" -F "file=@hydrology.zip"
```

The session and selected baseline/scenario must already exist. `ssp` and `year`
must be supplied together; supported years are 2025, 2030, 2050 and 2100.
These parameters also target existing population, sanitation and treatment CSV
uploads. Scenario population/sanitation updates preserve unrelated isodata columns.
CSV uploads do not regenerate spatial rasters.

The ZIP must contain one `hydrology/` root with all 75 required rasters. It is
validated before replacing the destination hydrology directory. Uploaded grids
and raster bytes are preserved. Ordinary generation preserves custom hydrology;
scenarios without their own upload reuse custom baseline hydrology unchanged.
This reuse is recorded explicitly and is not a climate projection.

See [Hydrology upload format and experimental processing](docs/HYDROLOGY_ADDITIONAL_DOWNSCALING.md)
for filenames, units, validation rules and errors.

`hydrology_downscaling` is **HIGHLY EXPERIMENTAL**, disabled by default, and not
validated for production modelling. In `/input/generate`, `hydro_raster` replaces
the previous D8-specific upload parameter. It is reserved for a generic
`hydro_raster.tif`; supplying it currently returns 422 because its scientific
meaning has not been defined. Complete custom datasets use the ZIP workflow above.
The projection endpoint retains its existing experimental D8 interface.

## Running tests

If you want to run it in docker, simply run:

```bash
docker-compose run --build --rm api pytest -vv .
docker-compose down
```

For running tests on your local machine.


2. Run the pytest.
```bash
pytest -vv .
```