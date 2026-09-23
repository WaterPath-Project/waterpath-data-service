from pathlib import Path

import rasterio
from affine import Affine

from waterpath_data_service.services import hydrology


def test_canonical_hydrology_filenames_cover_each_month() -> None:
    assert hydrology._CANONICAL_MONTHLY_FILES == {
        variable: tuple(f"{variable}_m{month:02d}.tif" for month in range(1, 13))
        for variable in hydrology._MODEL_VARIABLE_DIRS
    }


def test_all_model_directories_have_monthly_source_files() -> None:
    models_dir = (
        Path(hydrology.__file__).parent.parent
        / "static"
        / "data"
        / "hydrology"
        / "models"
    )
    missing = {}
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for variable, canonical_files in hydrology._CANONICAL_MONTHLY_FILES.items():
            variable_dir = model_dir / variable
            actual_files = {path.name for path in variable_dir.glob("*.tif")}
            if actual_files != set(canonical_files):
                missing.setdefault(model_dir.name, []).append(variable)

    assert missing == {}


def test_model_directories_contain_no_legacy_monthly_names() -> None:
    models_dir = (
        Path(hydrology.__file__).parent.parent
        / "static"
        / "data"
        / "hydrology"
        / "models"
    )

    legacy_files = sorted(path.relative_to(models_dir) for path in models_dir.rglob("*monmean*.tif"))

    assert legacy_files == []


def test_model_rasters_use_standard_global_grid() -> None:
    models_dir = (
        Path(hydrology.__file__).parent.parent
        / "static"
        / "data"
        / "hydrology"
        / "models"
    )
    expected_transform = Affine(0.5, 0.0, -180.0, 0.0, -0.5, 90.0)
    invalid = []

    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
        for variable, canonical_files in hydrology._CANONICAL_MONTHLY_FILES.items():
            for canonical_name in canonical_files:
                raster_path = model_dir / variable / canonical_name
                with rasterio.open(raster_path) as source:
                    if not (
                        source.shape == (360, 720)
                        and source.dtypes == ("float32",)
                        and source.crs is not None
                        and source.crs.to_epsg() == 4326
                        and source.transform == expected_transform
                    ):
                        invalid.append(raster_path.relative_to(models_dir))

    assert invalid == []