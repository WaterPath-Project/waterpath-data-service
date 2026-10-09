"""Prepare model inputs against an immutable emissions isoraster."""

from pathlib import Path
import json
import tempfile

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from waterpath_data_service.services.prepare_spatial import _cell_area_km2


def model_rasters(data_dir: Path) -> list[Path]:
    paths = list(data_dir.glob("*.tif"))
    for folder in ("human_emissions", "livestock_emissions", "qmra", "hydrology"):
        paths.extend((data_dir / folder).rglob("*.tif"))
    return sorted(path for path in paths if path.name != "fine_flowdir_source.tif")


def grid_signature(source):
    return source.crs, source.transform, source.width, source.height


def _validate_routing(paths: dict[Path, Path], data_dir: Path) -> None:
    from waterpath_data_service.services.hydrology import _d8_network

    direction = data_dir / "hydrology/routing/flowdir.tif"
    if direction not in paths:
        return
    with rasterio.open(paths[direction]) as source:
        values = source.read(1, masked=True).astype("float32").filled(np.nan)
    valid = np.isfinite(values)
    if not np.isin(values[valid], [0, 1, 2, 4, 8, 16, 32, 64, 128]).all():
        raise ValueError("Flow direction must use ESRI D8 codes.")
    downstream, order = _d8_network(values, valid)
    accumulation = data_dir / "hydrology/routing/flowacc.tif"
    if accumulation in paths:
        with rasterio.open(paths[accumulation]) as source:
            counts = source.read(1, masked=True).astype("float64").filled(np.nan)
        if np.any(valid & (~np.isfinite(counts) | (counts < 0) | (counts != np.floor(counts)))):
            raise ValueError("Flow accumulation must contain nonnegative integer counts across the routing network.")
        cells = np.asarray(order)
        connected = cells[downstream[cells] >= 0]
        if np.any(counts.ravel()[downstream[connected]] <= counts.ravel()[connected]):
            raise ValueError("Flow accumulation must increase downstream along the supplied routing network.")


def validate_model_grid(data_dir: Path) -> None:
    reference = data_dir / "human_emissions/isoraster.tif"
    if not reference.is_file():
        reference = data_dir / "isoraster.tif"
    if not reference.is_file():
        return
    with rasterio.open(reference) as source:
        expected = grid_signature(source)
    mismatches = []
    rasters = model_rasters(data_dir)
    hydrology = [path for path in rasters if path.is_relative_to(data_dir / "hydrology")]
    for path in (path for path in rasters if path not in hydrology):
        with rasterio.open(path) as source:
            if grid_signature(source) != expected:
                mismatches.append(str(path.relative_to(data_dir)))
    if mismatches:
        raise ValueError("Emission rasters do not match isoraster.tif: " + ", ".join(mismatches))
    if hydrology:
        with rasterio.open(hydrology[0]) as source:
            hydrology_grid = grid_signature(source)
        inconsistent = []
        for path in hydrology[1:]:
            with rasterio.open(path) as source:
                if grid_signature(source) != hydrology_grid:
                    inconsistent.append(str(path.relative_to(data_dir)))
        if inconsistent:
            raise ValueError("Hydrology rasters do not share one grid: " + ", ".join(inconsistent))
    _validate_routing({path: path for path in rasters}, data_dir)


def align_model_grid(data_dir: Path) -> None:
    template = data_dir / "human_emissions/isoraster.tif"
    if not template.is_file():
        template = data_dir / "isoraster.tif"
    if not template.is_file():
        return
    with rasterio.open(template) as reference:
        expected = grid_signature(reference)
        profile = reference.profile.copy()
    with tempfile.TemporaryDirectory(prefix=".model-grid-", dir=data_dir) as temporary:
        replacements = []
        for index, path in enumerate(model_rasters(data_dir)):
            if path.is_relative_to(data_dir / "hydrology"):
                continue
            with rasterio.open(path) as source:
                if grid_signature(source) == expected:
                    continue
                categorical = "isoraster" in path.name or path.parent.name in {"qmra", "routing"}
                counts = path.name in {"pop_urban.tif", "pop_rural.tif", "popurban.tif", "poprural.tif"} or path.name.endswith("_heads.tif")
                coarsening = abs(profile["transform"].a * profile["transform"].e) > abs(source.transform.a * source.transform.e)
                if categorical:
                    method = Resampling.mode if coarsening else Resampling.nearest
                elif counts:
                    method = Resampling.average
                else:
                    method = Resampling.average if coarsening else Resampling.bilinear
                output = np.full((source.count, profile["height"], profile["width"]), np.nan, dtype="float32")
                for band in range(source.count):
                    values = source.read(band + 1, masked=True).astype("float32").filled(np.nan)
                    before = np.nansum(values, dtype="float64")
                    if counts:
                        if source.crs.to_epsg() != 4326 or profile["crs"].to_epsg() != 4326:
                            raise ValueError(f"Count-raster preparation requires EPSG:4326 inputs: {path}")
                        values = np.nan_to_num(values, nan=0) / _cell_area_km2(source.transform, source.height, source.width)
                    reproject(
                        source=values, destination=output[band],
                        src_transform=source.transform, src_crs=source.crs, src_nodata=np.nan,
                        dst_transform=profile["transform"], dst_crs=profile["crs"], dst_nodata=np.nan,
                        resampling=method,
                    )
                    if counts:
                        output[band] *= _cell_area_km2(profile["transform"], profile["height"], profile["width"])
                        after = np.nansum(output[band], dtype="float64")
                        if before > 0 and after <= 0:
                            raise ValueError(f"Target grid loses all population/animal counts: {path}")
                        target_bounds = rasterio.transform.array_bounds(profile["height"], profile["width"], profile["transform"])
                        covers_source = (
                            target_bounds[0] <= source.bounds.left and target_bounds[1] <= source.bounds.bottom
                            and target_bounds[2] >= source.bounds.right and target_bounds[3] >= source.bounds.top
                        )
                        if after > 0 and covers_source:
                            output[band] *= before / after
                output_profile = profile.copy()
                output_profile.update(dtype="float32", nodata=np.nan, count=source.count)
                staged = Path(temporary) / f"{index}.tif"
                with rasterio.open(staged, "w", **output_profile) as destination:
                    destination.write(output)
                    destination.update_tags(**source.tags())
                replacements.append((staged, path))
        prepared = {path: path for path in model_rasters(data_dir)}
        prepared.update({path: staged for staged, path in replacements})
        _validate_routing(prepared, data_dir)
        for staged, path in replacements:
            staged.replace(path)
    validate_model_grid(data_dir)
    provenance = data_dir / "hydrology/source.json"
    if provenance.is_file():
        source_info = json.loads(provenance.read_text(encoding="utf-8"))
        source_info["emissions_grid_aligned"] = True
        source_info["hydrology_grid_preserved"] = True
        source_info["model_grid_aligned"] = False
        provenance.write_text(json.dumps(source_info, indent=2), encoding="utf-8")