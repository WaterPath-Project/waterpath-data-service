from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import from_origin

from waterpath_data_service.services import hydrology


def _write_test_raster(
    path: Path,
    values: np.ndarray,
    transform: Affine,
) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=values.shape[1],
        height=values.shape[0],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
    ) as destination:
        destination.write(values.astype(np.float32), 1)


def test_runoff_correction_preserves_area_weighted_coarse_depth(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    downscaled = tmp_path / "downscaled.tif"
    _write_test_raster(source, np.array([[10.0]]), from_origin(0.0, 0.1, 0.1, 0.1))
    _write_test_raster(
        downscaled,
        np.arange(1, 17, dtype=np.float32).reshape(4, 4),
        from_origin(0.0, 0.1, 0.025, 0.025),
    )

    hydrology._conserve_runoff_by_source_cell(source, downscaled)

    with rasterio.open(downscaled) as result:
        values = result.read(1)
        areas = hydrology._pixel_area_km2(result.transform, result.height, result.width)
    np.testing.assert_allclose(np.average(values, weights=areas), 10.0, rtol=1e-6)


def test_d8_flow_accumulation_counts_upstream_cells() -> None:
    flow_direction = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    downstream, order = hydrology._d8_network(
        flow_direction,
        np.ones(flow_direction.shape, dtype=bool),
    )

    result = hydrology._flow_accumulation(downstream, order, flow_direction.shape)

    np.testing.assert_array_equal(result, np.array([[1, 2, 3]], dtype=np.int32))


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