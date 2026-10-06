from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from waterpath_data_service.services import livestock
from waterpath_data_service.services.livestock import (
    _build_livestock_zone_template,
    _fao_country_total,
    _reproject_counts_to_zone_grid,
    _reproject_region_to_zone_grid,
)


def _write_single_pixel_raster(path: Path, value: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1,
        height=1,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 0.1, 0.1, 0.1),
        nodata=np.nan,
    ) as dst:
        dst.write(np.array([[value]], dtype=np.float32), 1)


def _fine_grid_profile() -> dict:
    return {
        "driver": "GTiff",
        "width": 4,
        "height": 4,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0.0, 0.1, 0.025, 0.025),
        "nodata": np.nan,
    }


def test_region_reprojection_uses_nearest_neighbour(tmp_path: Path) -> None:
    source = tmp_path / "animal_isoraster.tif"
    _write_single_pixel_raster(source, 7.0)

    result = _reproject_region_to_zone_grid(source, _fine_grid_profile())

    np.testing.assert_array_equal(result, np.full((4, 4), 7.0, dtype=np.float32))


def test_head_count_reprojection_conserves_total_when_splitting_cell(tmp_path: Path) -> None:
    source = tmp_path / "cattle_heads.tif"
    _write_single_pixel_raster(source, 100.0)

    result = _reproject_counts_to_zone_grid(source, _fine_grid_profile())

    assert float(np.nansum(result)) == pytest.approx(100.0, rel=1e-5)
    assert float(np.nanmean(result)) == pytest.approx(6.25, rel=1e-5)


def test_livestock_zone_template_matches_human_isoraster_for_large_area(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "case"
    shapefile = session_dir / "baseline" / "geodata" / "geodata.shp"
    shapefile.parent.mkdir(parents=True)
    shapefile.touch()
    reference = session_dir / "baseline" / "human_emissions" / "isoraster.tif"
    reference.parent.mkdir(parents=True)
    reference_profile = _fine_grid_profile()
    with rasterio.open(reference, "w", **reference_profile) as dst:
        dst.write(np.ones((4, 4), dtype=np.float32), 1)

    features = pd.DataFrame({"GID_0": ["TST"]})
    features["geometry"] = None
    features.at[0, "geometry"] = box(0.0, 0.0, 1.0, 1.0)
    features.__dict__["total_bounds"] = np.array([0.0, 0.0, 1.0, 1.0])
    monkeypatch.setattr(livestock.pyogrio, "read_dataframe", lambda _: features)
    monkeypatch.setattr(livestock, "_native_tif_resolution", lambda _: 0.1)

    _, _, result_profile = _build_livestock_zone_template(
        session_dir,
        tmp_path / "static",
        pd.DataFrame({"gid": ["TST"], "iso": [1]}),
    )

    assert result_profile["width"] == reference_profile["width"]
    assert result_profile["height"] == reference_profile["height"]
    assert result_profile["transform"] == reference_profile["transform"]
    assert result_profile["crs"] == rasterio.crs.CRS.from_epsg(4326)


def test_livestock_zone_template_prefers_projection_isoraster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_dir = tmp_path / "case"
    shapefile = session_dir / "baseline" / "geodata" / "geodata.shp"
    shapefile.parent.mkdir(parents=True)
    shapefile.touch()

    baseline_reference = session_dir / "baseline" / "human_emissions" / "isoraster.tif"
    baseline_reference.parent.mkdir(parents=True)
    with rasterio.open(baseline_reference, "w", **_fine_grid_profile()) as dst:
        dst.write(np.ones((4, 4), dtype=np.float32), 1)

    projection_reference = session_dir / "scenarios" / "SSP1_2050" / "isoraster.tif"
    projection_reference.parent.mkdir(parents=True)
    projection_profile = {
        **_fine_grid_profile(),
        "width": 3,
        "height": 2,
        "transform": from_origin(1.0, 2.0, 0.05, 0.05),
    }
    with rasterio.open(projection_reference, "w", **projection_profile) as dst:
        dst.write(np.ones((2, 3), dtype=np.float32), 1)

    features = pd.DataFrame({"GID_0": ["TST"]})
    features["geometry"] = None
    features.at[0, "geometry"] = box(1.0, 1.9, 1.15, 2.0)
    features.__dict__["total_bounds"] = np.array([1.0, 1.9, 1.15, 2.0])
    monkeypatch.setattr(livestock.pyogrio, "read_dataframe", lambda _: features)
    monkeypatch.setattr(livestock, "_native_tif_resolution", lambda _: 0.1)

    zone_idx, _, result_profile = _build_livestock_zone_template(
        session_dir,
        tmp_path / "static",
        pd.DataFrame({"gid": ["TST"], "iso": [1]}),
        reference_isoraster_path=projection_reference,
    )

    assert zone_idx.shape == (2, 3)
    assert result_profile["width"] == projection_profile["width"]
    assert result_profile["height"] == projection_profile["height"]
    assert result_profile["transform"] == projection_profile["transform"]


def test_fao_country_total_returns_matching_value() -> None:
    fao = pd.DataFrame(
        {
            "Area Code (ISO3)": ["UGA", "UGA", "KEN"],
            "Item": ["Asses", "Asses", "Asses"],
            "Year": [2020, 2019, 2020],
            "Value": [19373, 19243, 900000],
        }
    )

    assert _fao_country_total(fao, "uga", "Asses", 2020) == 19373.0


def test_fao_country_total_distinguishes_zero_from_missing() -> None:
    fao = pd.DataFrame(
        {
            "Area Code (ISO3)": ["UGA"],
            "Item": ["Ducks"],
            "Year": [2020],
            "Value": [0],
        }
    )

    assert _fao_country_total(fao, "UGA", "Ducks", 2020) == 0.0
    assert _fao_country_total(fao, "UGA", "Camels", 2020) is None