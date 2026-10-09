"""Prepare model inputs against an immutable emissions isoraster."""

from __future__ import annotations

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


_POPULATION_RASTERS = ("pop_urban.tif", "pop_rural.tif", "popurban.tif", "poprural.tif")


def _routed_hydrology(data_dir: Path):
    """Return ``(routed, transform, crs)`` for generated hydrology routing, or ``None``.

    Custom uploaded hydrology is left to the uploader (see README), so only
    hydrology produced by this service is checked.
    """
    routing = data_dir / "hydrology/routing"
    direction = routing / "flowdir.tif"
    if not direction.is_file():
        return None
    provenance = data_dir / "hydrology/source.json"
    if provenance.is_file():
        try:
            source_name = str(json.loads(provenance.read_text(encoding="utf-8")).get("source", ""))
        except ValueError:
            source_name = ""
        if source_name.startswith("custom"):
            return None
    with rasterio.open(direction) as source:
        routed = np.isfinite(source.read(1, masked=True).astype("float64").filled(np.nan))
        transform, crs, shape = source.transform, source.crs, source.shape
    accumulation = routing / "flowacc.tif"
    if accumulation.is_file():
        with rasterio.open(accumulation) as source:
            if source.shape == shape and source.transform == transform:
                routed &= np.isfinite(source.read(1, masked=True).astype("float64").filled(np.nan))
    return routed, transform, crs


def _unrouted_mask(transform, height: int, width: int, crs, hydrology) -> np.ndarray | None:
    """Mark emission cells not fully covered by routed hydrology cells.

    Hydrology coupling distributes each emission cell over the hydrology cells
    it overlaps, so a cell is only safe when every overlapped hydrology cell
    lies inside the hydrology extent and has flow direction/accumulation data.
    """
    routed, routing_transform, routing_crs = hydrology
    if crs != routing_crs or transform.b or transform.d or routing_transform.b or routing_transform.d:
        return None
    eps = 1e-6

    def axis_ranges(edges, origin, size, cells):
        position = (edges - origin) / size
        lower = np.minimum(position[:-1], position[1:])
        upper = np.maximum(position[:-1], position[1:])
        first = np.floor(lower + eps).astype(int)
        last = np.ceil(upper - eps).astype(int) - 1
        inside = (first >= 0) & (last < cells) & (last >= first)
        return np.clip(first, 0, cells - 1), np.clip(last, 0, cells - 1), inside

    x_edges = transform.c + transform.a * np.arange(width + 1)
    y_edges = transform.f + transform.e * np.arange(height + 1)
    col_first, col_last, col_inside = axis_ranges(
        x_edges, routing_transform.c, routing_transform.a, routed.shape[1])
    row_first, row_last, row_inside = axis_ranges(
        y_edges, routing_transform.f, routing_transform.e, routed.shape[0])
    missing = np.pad((~routed).astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    r0, r1 = row_first[:, None], row_last[:, None] + 1
    c0, c1 = col_first[None, :], col_last[None, :] + 1
    unrouted_overlaps = missing[r1, c1] - missing[r0, c1] - missing[r1, c0] + missing[r0, c0]
    covered = row_inside[:, None] & col_inside[None, :] & (unrouted_overlaps == 0)
    return ~covered


def _zone_domain(values: np.ndarray, nodata) -> np.ndarray:
    domain = np.isfinite(values) & (values != 0)
    if nodata is not None and np.isfinite(nodata):
        domain &= values != nodata
    return domain


def _zone_rasters(data_dir: Path, template: Path) -> list[tuple[Path, list[Path]]]:
    """Pair each zone raster with the count rasters whose totals it defines."""
    human = [template.parent / name for name in _POPULATION_RASTERS if (template.parent / name).is_file()]
    heads = sorted((data_dir / "livestock_emissions/animals").glob("*_heads.tif"))
    animal = data_dir / "livestock_emissions/animal_isoraster.tif"
    if animal.is_file():
        return [(template, human), (animal, heads)]
    return [(template, human + heads)]


def _unrouted_emission_cells(data_dir: Path, template: Path) -> dict[Path, np.ndarray]:
    hydrology = _routed_hydrology(data_dir)
    if hydrology is None:
        return {}
    cells = {}
    for zone_path, _ in _zone_rasters(data_dir, template):
        with rasterio.open(zone_path) as source:
            unrouted = _unrouted_mask(source.transform, source.height, source.width, source.crs, hydrology)
            if unrouted is None:
                continue
            values = source.read(1).astype("float64")
            cells[zone_path] = _zone_domain(values, source.nodata) & unrouted
    return cells


def _exclude_unrouted_cells(data_dir: Path, template: Path) -> None:
    """Drop emission cells that hydrology cannot route, conserving zone totals.

    Coastal or edge cells whose hydrology cell has no flow direction (e.g. sea
    in the default 0.5-degree routing network) would otherwise receive
    emissions that can never reach the river network. Their population and
    livestock counts are moved to the remaining cells of the same zone.
    """
    hydrology = _routed_hydrology(data_dir)
    if hydrology is None:
        return
    updates: dict[Path, tuple[np.ndarray, dict, dict]] = {}
    stranded: set[int] = set()
    for zone_path, count_paths in _zone_rasters(data_dir, template):
        with rasterio.open(zone_path) as source:
            unrouted = _unrouted_mask(source.transform, source.height, source.width, source.crs, hydrology)
            if unrouted is None:
                continue
            zone_profile, zone_tags = source.profile.copy(), source.tags()
            zones = source.read(1)
            grid = grid_signature(source)
        domain = _zone_domain(zones.astype("float64"), zone_profile.get("nodata"))
        remove = domain & unrouted
        if not remove.any():
            continue
        zone_ids = np.unique(zones[remove])
        keep_by_zone = {zone: domain & ~remove & (zones == zone) for zone in zone_ids}
        stranded.update(int(zone) for zone, keep in keep_by_zone.items() if not keep.any())
        if stranded:
            continue
        for path in count_paths:
            with rasterio.open(path) as source:
                if grid_signature(source) != grid:
                    continue
                profile, tags = source.profile.copy(), source.tags()
                counts = source.read(1, masked=True).astype("float64").filled(np.nan)
            valid = np.isfinite(counts)
            for zone, keep in keep_by_zone.items():
                moved = np.nansum(np.where(remove & (zones == zone), counts, 0.0))
                if moved == 0:
                    continue
                base = np.where(keep & valid, np.clip(counts, 0, None), 0.0)
                weights = base / base.sum() if base.sum() > 0 else keep / keep.sum()
                counts = np.where(keep, np.where(valid, counts, 0.0) + moved * weights, counts)
            nodata = profile.get("nodata")
            counts[remove] = nodata if nodata is not None else 0.0
            if nodata is not None:
                counts[~np.isfinite(counts)] = nodata
            updates[path] = (counts, profile, tags)
        new_zones = zones.copy()
        zone_nodata = zone_profile.get("nodata")
        new_zones[remove] = zone_nodata if zone_nodata is not None else 0
        updates[zone_path] = (new_zones, zone_profile, zone_tags)
    if stranded:
        raise ValueError(
            "Emission zone(s) " + ", ".join(str(zone) for zone in sorted(stranded))
            + " lie entirely in hydrology cells without routing (e.g. sea or outside the "
            "hydrology extent) and cannot be represented with this hydrology."
        )
    for path, (values, profile, tags) in updates.items():
        with rasterio.open(path, "w", **profile) as destination:
            destination.write(values.astype(profile["dtype"]), 1)
            destination.update_tags(**tags)


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
    unrouted = {path: int(cells.sum()) for path, cells in _unrouted_emission_cells(data_dir, reference).items()}
    unrouted = {path: count for path, count in unrouted.items() if count}
    if unrouted:
        raise ValueError(
            "Emission cells lie in hydrology cells without routing or outside the hydrology extent: "
            + ", ".join(f"{path.relative_to(data_dir)} ({count} cells)" for path, count in unrouted.items())
        )


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
    _exclude_unrouted_cells(data_dir, template)
    validate_model_grid(data_dir)
    provenance = data_dir / "hydrology/source.json"
    if provenance.is_file():
        source_info = json.loads(provenance.read_text(encoding="utf-8"))
        source_info["emissions_grid_aligned"] = True
        source_info["hydrology_grid_preserved"] = True
        source_info["model_grid_aligned"] = False
        provenance.write_text(json.dumps(source_info, indent=2), encoding="utf-8")