from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject

logger = logging.getLogger(__name__)


_IPCC_ORDER = [
    "Africa",
    "Asia",
    "Europe",
    "Latin America",
    "NENA",
    "North America",
    "Oceania",
]

_NICE_RESOLUTIONS = [0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]

_FAOSTAT_STOCKS_CSV_URL = (
    "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data/"
    "refs/heads/main/faostat_stocks/data/FAOSTAT_data.csv"
)
_FAOSTAT_COUNTRY_GROUPS_CSV_URL = (
    "https://raw.githubusercontent.com/WaterPath-Project/waterpath-data/"
    "refs/heads/main/faostat_stocks/data/FAOSTAT_country_groups.csv"
)


def _native_tif_resolution(raster_path: Path) -> float:
    with rasterio.open(raster_path) as src:
        return abs(src.transform.a)


def _round_to_nice_res(resolution: float) -> float:
    return min(_NICE_RESOLUTIONS, key=lambda value: abs(value - resolution))


def _session_shapefile_path(session_dir: Path) -> Path:
    candidates = [
        session_dir / "baseline" / "geodata" / "geodata.shp",
        session_dir / "geodata" / "geodata.shp",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Session shapefile not found. Expected geodata.shp under "
        f"{candidates[0].parent} (or {candidates[1].parent})."
    )


def _feature_gid(feature: dict, fallback_iso: int) -> str:
    return str(
        feature.get("GID_3")
        or feature.get("GID_2")
        or feature.get("GID_1")
        or feature.get("GID_0")
        or fallback_iso
    ).strip()


def _build_livestock_zone_template(
    session_dir: Path,
    static_data_dir: Path,
    mapping: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, rasterio.profiles.Profile]:
    """Build a fresh zone grid from geodata.shp using the same grid-snapping logic as prepare.R."""
    shapefile_path = _session_shapefile_path(session_dir)
    features = pyogrio.read_dataframe(shapefile_path)

    if features.empty:
        raise ValueError(f"No features found in {shapefile_path}")

    gid_to_iso = mapping.drop_duplicates(subset=["gid"], keep="first").set_index("gid")["iso"].to_dict()

    xmin_data, ymin_data, xmax_data, ymax_data = features.total_bounds.tolist()

    # Match prepare.R resolution snapping, but use a livestock raster as the native-resolution floor.
    extent_x = xmax_data - xmin_data
    extent_y = ymax_data - ymin_data
    diagonal = math.hypot(extent_x, extent_y)
    native_raster = static_data_dir / "glw4_2020" / "GLW4-2020.D-DA.CTL.tif"
    src_native_res = _native_tif_resolution(native_raster)
    raw = max(src_native_res, min(0.5, diagonal / 100.0))
    res = _round_to_nice_res(raw)

    xmin = max(math.floor(xmin_data / res) * res - res, -180.0)
    ymin = max(math.floor(ymin_data / res) * res - res, -90.0)
    xmax = min(math.ceil(xmax_data / res) * res + res, 180.0)
    ymax = min(math.ceil(ymax_data / res) * res + res, 90.0)

    width = round((xmax - xmin) / res)
    height = round((ymax - ymin) / res)
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    shapes = [
        (row.geometry.__geo_interface__, int(gid_to_iso.get(_feature_gid(row, idx + 1), idx + 1)))
        for _, idx, row in sorted(
            (
                (
                    (row.geometry.bounds[2] - row.geometry.bounds[0])
                    * (row.geometry.bounds[3] - row.geometry.bounds[1]),
                    idx,
                    row,
                )
                for idx, (_, row) in enumerate(features.iterrows())
            ),
            key=lambda item: item[0],
            reverse=True,
        )
    ]
    zone_idx = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.int32,
        all_touched=True,
    )

    profile = {
        "driver": "GTiff",
        "dtype": "int32",
        "nodata": 0,
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
    }
    valid_mask = zone_idx > 0
    return zone_idx, valid_mask, profile


def _read_zone_index(isoraster_path: Path) -> tuple[np.ndarray, np.ndarray, rasterio.profiles.Profile]:
    with rasterio.open(isoraster_path) as src:
        zone_arr = src.read(1, masked=True)
        zone_idx = np.where(zone_arr.mask, 0, zone_arr.filled(0)).astype(np.int32)
        valid_mask = zone_idx > 0
        profile = src.profile.copy()
    return zone_idx, valid_mask, profile


def _reproject_region_to_zone_grid(source_raster: Path, zone_profile: rasterio.profiles.Profile) -> np.ndarray:
    """Reproject a categorical raster (e.g. region IDs) to the zone grid using nearest-neighbour."""
    dst = np.full((zone_profile["height"], zone_profile["width"]), np.nan, dtype=np.float32)
    with rasterio.open(source_raster) as src:
        src_arr = src.read(1)
        src_nodata = src.nodata
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=zone_profile["transform"],
            dst_crs=zone_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    return dst


def _clip_raster_to_zone_grid(
    source_raster: Path,
    zone_profile: rasterio.profiles.Profile,
    source_nodata: float | None = None,
) -> np.ndarray:
    """Reproject/clip a continuous raster (e.g. animal heads) to the zone grid using bilinear resampling."""
    dst = np.full((zone_profile["height"], zone_profile["width"]), np.nan, dtype=np.float32)
    with rasterio.open(source_raster) as src:
        src_arr = src.read(1).astype(np.float32)
        nd = source_nodata if source_nodata is not None else src.nodata
        # If the file has no nodata metadata but stores sentinel values such as
        # -9999, treat any negative pixel as nodata so bilinear resampling does
        # not bleed those sentinels into valid cells.
        if nd is None:
            src_arr = np.where(src_arr < 0, np.nan, src_arr)
            nd = np.nan
        else:
            src_arr = np.where(src_arr == nd, np.nan, src_arr)
            nd = np.nan
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=nd,
            dst_transform=zone_profile["transform"],
            dst_crs=zone_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    # Bilinear interpolation at nodata boundaries can still produce small
    # negative artefacts; clamp them because animal head counts must be >= 0.
    dst = np.where(dst < 0, np.nan, dst)
    return dst


def _write_float_raster(arr: np.ndarray, zone_profile: rasterio.profiles.Profile, out_path: Path) -> None:
    """Write a float32 array to a GeoTIFF using the zone grid CRS/transform."""
    profile = zone_profile.copy()
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(np.float32), 1)


def _aggregate_lookup_columns(
    zone_idx: np.ndarray,
    valid_mask: np.ndarray,
    region_idx: np.ndarray,
    lookup_df: pd.DataFrame,
    lookup_region_col: str,
    value_columns: list[str],
) -> pd.DataFrame:
    flat_zone = zone_idx.reshape(-1)
    flat_region = region_idx.reshape(-1)
    flat_mask = valid_mask.reshape(-1) & np.isfinite(flat_region)

    data = pd.DataFrame(
        {
            "iso": flat_zone[flat_mask].astype(np.int32),
            "region": flat_region[flat_mask].astype(np.int32),
        }
    )

    dominant_region = (
        data.groupby("iso", as_index=False)["region"]
        .agg(lambda values: values.value_counts().idxmax())
        .rename(columns={"region": "_region"})
    )

    out = dominant_region[["iso"]].copy()
    region_lookup = lookup_df.set_index(lookup_region_col)

    for col in value_columns:
        out[col] = dominant_region["_region"].map(region_lookup[col])

    return out


def _load_iso_gid_mapping(session_dir: Path) -> pd.DataFrame:
    pop_path = session_dir / "baseline" / "human_emissions" / "population.csv"
    if not pop_path.is_file():
        raise FileNotFoundError(f"population.csv not found at {pop_path}")

    pop = pd.read_csv(pop_path)
    if "iso" not in pop.columns or "gid" not in pop.columns:
        raise ValueError("population.csv must contain 'iso' and 'gid' columns")

    mapping = pop[["iso", "gid"]].copy()
    mapping["iso"] = pd.to_numeric(mapping["iso"], errors="coerce").astype("Int64")
    mapping = mapping.dropna(subset=["iso"])
    mapping["iso"] = mapping["iso"].astype(np.int32)
    mapping["gid"] = mapping["gid"].astype(str)
    return mapping.drop_duplicates(subset=["iso"], keep="first")


def _generate_production_systems(
    static_data_dir: Path,
    output_dir: Path,
    mapping: pd.DataFrame,
    zone_idx: np.ndarray,
    valid_mask: np.ndarray,
    zone_profile: rasterio.profiles.Profile,
) -> Path:
    nutdata_raster = static_data_dir / "vermeulen_2017" / "nutdata_2000_iso.tif"
    nutdata_csv = static_data_dir / "vermeulen_2017" / "nutdata_2000_intensive_extensive.csv"

    region_idx = _reproject_region_to_zone_grid(nutdata_raster, zone_profile)

    src = pd.read_csv(nutdata_csv)
    livestock = [
        "meat",
        "dairy",
        "buffaloes",
        "pigs",
        "poultry",
        "sheep",
        "goats",
        "horses",
        "asses",
        "mules",
        "camels",
    ]
    value_columns = [f"{name}_{sys}" for name in livestock for sys in ("i", "e")]

    agg = _aggregate_lookup_columns(
        zone_idx=zone_idx,
        valid_mask=valid_mask,
        region_idx=region_idx,
        lookup_df=src,
        lookup_region_col="iso",
        value_columns=value_columns,
    )

    out = mapping.merge(agg, on="iso", how="left")
    out_path = output_dir / "production_systems.csv"
    out.to_csv(out_path, index=False)
    return out_path


def _generate_manure_fractions(
    static_data_dir: Path,
    output_dir: Path,
    mapping: pd.DataFrame,
    zone_idx: np.ndarray,
    valid_mask: np.ndarray,
    zone_profile: rasterio.profiles.Profile,
) -> Path:
    nutdata_raster = static_data_dir / "vermeulen_2017" / "nutdata_2000_iso.tif"
    fractions_csv = static_data_dir / "vermeulen_2017" / "nutdata_2000_fractions_mm.csv"

    region_idx = _reproject_region_to_zone_grid(nutdata_raster, zone_profile)

    src = pd.read_csv(fractions_csv)
    livestock = [
        "meat",
        "dairy",
        "buffaloes",
        "pigs",
        "poultry",
        "sheep",
        "goats",
        "horses",
        "asses",
        "mules",
        "camels",
    ]
    value_columns = [f"{name}_{part}" for name in livestock for part in ("fgi", "fge", "foi", "foe")]

    agg = _aggregate_lookup_columns(
        zone_idx=zone_idx,
        valid_mask=valid_mask,
        region_idx=region_idx,
        lookup_df=src,
        lookup_region_col="iso",
        value_columns=value_columns,
    )

    out = mapping.merge(agg, on="iso", how="left")
    out_path = output_dir / "manure_fractions.csv"
    out.to_csv(out_path, index=False)
    return out_path


def _generate_manure_management(
    static_data_dir: Path,
    output_dir: Path,
    mapping: pd.DataFrame,
) -> Path:
    mm_csv = static_data_dir / "vermeulen_2017" / "manure_management_systems.csv"

    src = pd.read_csv(mm_csv, encoding="latin-1")
    groups = pd.read_csv(_FAOSTAT_COUNTRY_GROUPS_CSV_URL)

    groups["M49 Code"] = pd.to_numeric(groups["M49 Code"], errors="coerce")
    groups = groups.dropna(subset=["M49 Code", "ISO3 Code"]).copy()
    groups["M49 Code"] = groups["M49 Code"].astype(np.int32)
    groups["ISO3 Code"] = groups["ISO3 Code"].astype(str)

    iso3_to_m49 = groups.drop_duplicates(subset=["ISO3 Code"], keep="first").set_index("ISO3 Code")["M49 Code"].to_dict()

    out = mapping.copy()
    out["iso3"] = out["gid"].str[:3]
    out["m49"] = out["iso3"].map(iso3_to_m49)

    src["iso"] = pd.to_numeric(src["iso"], errors="coerce")
    src = src.dropna(subset=["iso"]).copy()
    src["iso"] = src["iso"].astype(np.int32)

    value_columns = [
        c
        for c in src.columns
        if c not in ("country", "iso", "CC")
        and "_" in c
        and not c.startswith("Tot")
        and not c.startswith("Tot2")
    ]

    mm_lookup = src.drop_duplicates(subset=["iso"], keep="first").set_index("iso")
    for col in value_columns:
        out[col] = out["m49"].map(mm_lookup[col])

    out = out.drop(columns=["iso3", "m49"])

    out_path = output_dir / "manure_management.csv"
    out.to_csv(out_path, index=False)
    return out_path


def _generate_animal_isodata(
    static_data_dir: Path,
    output_dir: Path,
    active_region_ids: list[int],
) -> dict[str, str]:
    animals_csv = static_data_dir / "vermeulen_2017" / "animals.csv"
    region_animal_csv = static_data_dir / "vermeulen_2017" / "ippc_region_animal.csv"

    animals = pd.read_csv(animals_csv)
    region_animal = pd.read_csv(region_animal_csv)

    merged = region_animal.merge(animals, on="animal", how="left")

    regions = [r for r in _IPCC_ORDER if r in merged["ipcc_region"].unique().tolist()]
    regions += sorted([r for r in merged["ipcc_region"].unique().tolist() if r not in regions])
    region_id = {name: idx + 1 for idx, name in enumerate(regions)}

    out_dir = output_dir / "animals"
    out_dir.mkdir(parents=True, exist_ok=True)

    created: dict[str, str] = {}
    for animal in sorted(merged["animal"].dropna().unique().tolist()):
        df = merged.loc[merged["animal"] == animal].copy()
        df["iso"] = df["ipcc_region"].map(region_id)
        df = df.rename(
            columns={
                "birth_weight": "mass_young",
                "frac_lt_3m": "frac_young",
            }
        )
        df["mass_young"] = pd.to_numeric(df["mass_young"], errors="coerce")
        df["mass_adult"] = pd.to_numeric(df["mass_adult"], errors="coerce")
        df["mass_young"] = df["mass_young"].fillna(df["mass_adult"])

        for col in ("excr_young", "excr_adult", "mass_young", "mass_adult", "manure_per_mass"):
            df[col] = pd.to_numeric(df[col], errors="coerce").round()

        cols = [
            "iso",
            "frac_young",
            "prev_young",
            "prev_adult",
            "excr_young",
            "excr_adult",
            "excr_day",
            "mass_young",
            "mass_adult",
            "manure_per_mass",
        ]
        df = df.loc[df["iso"].isin(active_region_ids), cols].sort_values("iso").reset_index(drop=True)

        out_path = out_dir / f"isodata_{animal}.csv"
        df.to_csv(out_path, index=False)
        created[animal] = str(out_path)

    return created


def _build_m49_to_ipcc_region(country_groups_csv: str | Path) -> dict[int, int]:
    """
    Build a mapping from M49 country code → IPCC livestock region integer ID.

    IPCC regions follow _IPCC_ORDER (1-based index). The mapping uses FAO geographic
    groups with NENA (Northern Africa + Western Asia) taking priority over the broader
    Africa and Asia groups respectively.
    """
    groups = pd.read_csv(country_groups_csv)
    groups = groups.dropna(subset=["M49 Code", "Country Group Code"]).copy()
    groups["M49 Code"] = pd.to_numeric(groups["M49 Code"], errors="coerce")
    groups["Country Group Code"] = pd.to_numeric(groups["Country Group Code"], errors="coerce")
    groups = groups.dropna(subset=["M49 Code", "Country Group Code"])
    groups["M49 Code"] = groups["M49 Code"].astype(int)
    groups["Country Group Code"] = groups["Country Group Code"].astype(int)

    def _m49_in_group(code: int) -> set[int]:
        return set(groups.loc[groups["Country Group Code"] == code, "M49 Code"])

    # NENA takes priority over Africa/Asia, so assign last (overrides earlier assignment)
    assignment_order = [
        (_m49_in_group(5100), "Africa"),         # Africa
        (_m49_in_group(5300), "Asia"),            # Asia
        (_m49_in_group(5400), "Europe"),          # Europe
        (_m49_in_group(5205) | _m49_in_group(5206), "Latin America"),  # Latin Am + Caribbean
        (_m49_in_group(5203), "North America"),   # Northern America
        (_m49_in_group(5500), "Oceania"),         # Oceania
        # NENA last so it overrides Africa/Asia for overlapping countries
        (_m49_in_group(5103) | _m49_in_group(5305), "NENA"),  # N. Africa + W. Asia
    ]

    region_id = {name: idx + 1 for idx, name in enumerate(_IPCC_ORDER)}
    m49_to_ipcc: dict[int, int] = {}
    for m49_set, region_name in assignment_order:
        ipcc_id = region_id.get(region_name)
        if ipcc_id is not None:
            for m49 in m49_set:
                m49_to_ipcc[m49] = ipcc_id

    return m49_to_ipcc


def _build_ipcc_region_array(
    static_data_dir: Path,
    zone_idx: np.ndarray,
    zone_profile: rasterio.profiles.Profile,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    nutdata_raster = static_data_dir / "vermeulen_2017" / "nutdata_2000_iso.tif"

    m49_arr = _reproject_region_to_zone_grid(nutdata_raster, zone_profile)
    m49_to_ipcc = _build_m49_to_ipcc_region(_FAOSTAT_COUNTRY_GROUPS_CSV_URL)
    ipcc_arr = np.full_like(m49_arr, np.nan, dtype=np.float32)
    for m49, ipcc_id in m49_to_ipcc.items():
        ipcc_arr[m49_arr == float(m49)] = float(ipcc_id)
    ipcc_arr[~valid_mask] = np.nan

    # Reduce to one IPCC region per session zone so rows align with the session iso/gid domain.
    zone_ipcc_arr = np.full_like(ipcc_arr, np.nan, dtype=np.float32)
    for iso_id in np.unique(zone_idx[valid_mask]):
        zone_mask = zone_idx == iso_id
        zone_values = ipcc_arr[zone_mask & np.isfinite(ipcc_arr)].astype(np.int32)
        if zone_values.size == 0:
            continue
        dominant = np.bincount(zone_values).argmax()
        zone_ipcc_arr[zone_mask] = float(dominant)

    active_region_ids = sorted(int(value) for value in np.unique(zone_ipcc_arr[np.isfinite(zone_ipcc_arr)]))
    return zone_ipcc_arr, active_region_ids


def _generate_animal_isoraster(
    static_data_dir: Path,
    output_dir: Path,
    zone_idx: np.ndarray,
    zone_profile: rasterio.profiles.Profile,
    valid_mask: np.ndarray,
) -> tuple[Path, list[int], np.ndarray]:
    """Write animal_isoraster.tif using the session geodata grid.

    Returns (path, active_region_ids, zone_ipcc_arr) so callers can reuse the
    already-computed per-pixel IPCC region array without re-running it.
    """
    ipcc_arr, active_region_ids = _build_ipcc_region_array(static_data_dir, zone_idx, zone_profile, valid_mask)

    out_path = output_dir / "animal_isoraster.tif"
    _write_float_raster(ipcc_arr, zone_profile, out_path)
    return out_path, active_region_ids, ipcc_arr


# IPCC region IDs follow _IPCC_ORDER (1-based): Africa=1, Asia=2, Europe=3,
# Latin America=4, NENA=5, North America=6, Oceania=7.
# Proxy species that only occur in specific regions must be masked accordingly
# so that using a sheep/goat spatial proxy does not produce biologically
# implausible occurrences (e.g. camels in Europe).
_PROXY_ALLOWED_IPCC_REGIONS: dict[str, frozenset[int]] = {
    "camels": frozenset({1, 2, 5}),  # Africa, Asia, NENA
}


def _generate_animal_heads_rasters(
    static_data_dir: Path,
    output_dir: Path,
    zone_profile: rasterio.profiles.Profile,
    valid_mask: np.ndarray,
    zone_ipcc_arr: np.ndarray,
) -> dict[str, str]:
    """
    Generate per-animal heads rasters clipped to the session extent.

    GLW4 2020 rasters cover cattle, buffaloes, chickens, goats, pigs, sheep.
    Ducks are taken from GLW4 2015 and scaled to 2020 using global FAOSTAT totals.
    Horses, donkeys, asses, mules, and camels use a sheep+goat spatial proxy scaled
    by the ratio of each species' FAOSTAT 2020 global total to the combined sheep+goat total.
    Proxy species listed in ``_PROXY_ALLOWED_IPCC_REGIONS`` are additionally
    masked to their plausible IPCC regions to avoid artefacts (e.g. camels in
    Europe due to the sheep/goat proxy).
    """
    glw4_2020_dir = static_data_dir / "glw4_2020"
    glw4_2015_dir = static_data_dir / "glw4_2015"
    faostat_csv = _FAOSTAT_STOCKS_CSV_URL

    animals_dir = output_dir / "animals"
    animals_dir.mkdir(parents=True, exist_ok=True)

    fao = pd.read_csv(faostat_csv)

    def _fao_total(item_name: str, year: int = 2020) -> float:
        sub = fao[(fao["Year"] == year) & (fao["Item"] == item_name)]
        return float(sub["Value"].sum())

    created: dict[str, str] = {}

    # GLW4 2020 direct mappings
    glw4_direct = {
        "cattle":    "GLW4-2020.D-DA.CTL.tif",
        "buffaloes": "GLW4-2020.D-DA.BFL.tif",
        "chickens":  "GLW4-2020.D-DA.CHK.tif",
        "goats":     "GLW4-2020.D-DA.GTS.tif",
        "pigs":      "GLW4-2020.D-DA.PGS.tif",
        "sheep":     "GLW4-2020.D-DA.SHP.tif",
    }
    for animal, fname in glw4_direct.items():
        arr = _clip_raster_to_zone_grid(glw4_2020_dir / fname, zone_profile)
        arr[~valid_mask] = np.nan
        out_path = animals_dir / f"{animal}_heads.tif"
        _write_float_raster(arr, zone_profile, out_path)
        created[animal] = str(out_path)
        logger.debug("Written %s", out_path)

    # Ducks: GLW4 2015 scaled to 2020 by global FAOSTAT ratio
    duck_2015_total = _fao_total("Ducks", 2015)
    duck_2020_total = _fao_total("Ducks", 2020)
    duck_scale = duck_2020_total / duck_2015_total if duck_2015_total > 0 else 1.0
    duck_arr = _clip_raster_to_zone_grid(glw4_2015_dir / "5_Dk_2015_Da.tif", zone_profile)
    duck_arr = np.where(np.isfinite(duck_arr), duck_arr * duck_scale, np.nan).astype(np.float32)
    duck_arr[~valid_mask] = np.nan
    out_path = animals_dir / "ducks_heads.tif"
    _write_float_raster(duck_arr, zone_profile, out_path)
    created["ducks"] = str(out_path)
    logger.debug("Written %s", out_path)

    # Proxy species (no GLW4 raster): distribute using SHP+GTS spatial pattern
    shp_arr = _clip_raster_to_zone_grid(glw4_2020_dir / "GLW4-2020.D-DA.SHP.tif", zone_profile)
    gts_arr = _clip_raster_to_zone_grid(glw4_2020_dir / "GLW4-2020.D-DA.GTS.tif", zone_profile)
    proxy_arr = np.where(np.isfinite(shp_arr) & np.isfinite(gts_arr), shp_arr + gts_arr, np.nan)
    proxy_total = _fao_total("Sheep") + _fao_total("Goats")

    proxy_species = {
        "horses":  "Horses",
        "donkeys": "Asses",
        "asses":   "Asses",
        "mules":   "Mules and hinnies",
        "camels":  "Camels",
    }
    for animal, fao_item in proxy_species.items():
        fao_count = _fao_total(fao_item)
        ratio = fao_count / proxy_total if proxy_total > 0 else 0.0
        arr = np.where(np.isfinite(proxy_arr), proxy_arr * ratio, np.nan).astype(np.float32)
        # Restrict species to their biologically plausible IPCC regions.
        allowed_regions = _PROXY_ALLOWED_IPCC_REGIONS.get(animal)
        if allowed_regions is not None:
            region_ok = np.zeros(zone_ipcc_arr.shape, dtype=bool)
            for rid in allowed_regions:
                region_ok |= zone_ipcc_arr == float(rid)
            arr = np.where(region_ok, arr, np.nan)
        arr[~valid_mask] = np.nan
        out_path = animals_dir / f"{animal}_heads.tif"
        _write_float_raster(arr, zone_profile, out_path)
        created[animal] = str(out_path)
        logger.debug("Written %s", out_path)

    return created


def generate_livestock_tabular_inputs(session_dir: Path, static_data_dir: Path) -> dict[str, str | dict[str, str]]:
    output_dir = session_dir / "baseline" / "livestock_emissions"
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = _load_iso_gid_mapping(session_dir)
    zone_idx, valid_mask, zone_profile = _build_livestock_zone_template(session_dir, static_data_dir, mapping)

    production = _generate_production_systems(
        static_data_dir=static_data_dir,
        output_dir=output_dir,
        mapping=mapping,
        zone_idx=zone_idx,
        valid_mask=valid_mask,
        zone_profile=zone_profile,
    )
    manure_fractions = _generate_manure_fractions(
        static_data_dir=static_data_dir,
        output_dir=output_dir,
        mapping=mapping,
        zone_idx=zone_idx,
        valid_mask=valid_mask,
        zone_profile=zone_profile,
    )
    manure_management = _generate_manure_management(
        static_data_dir=static_data_dir,
        output_dir=output_dir,
        mapping=mapping,
    )

    animal_isoraster, active_region_ids, zone_ipcc_arr = _generate_animal_isoraster(
        static_data_dir=static_data_dir,
        output_dir=output_dir,
        zone_idx=zone_idx,
        zone_profile=zone_profile,
        valid_mask=valid_mask,
    )
    animal_isodata = _generate_animal_isodata(
        static_data_dir=static_data_dir,
        output_dir=output_dir,
        active_region_ids=active_region_ids,
    )
    animal_heads = _generate_animal_heads_rasters(
        static_data_dir=static_data_dir,
        output_dir=output_dir,
        zone_profile=zone_profile,
        valid_mask=valid_mask,
        zone_ipcc_arr=zone_ipcc_arr,
    )

    logger.info("Generated livestock tabular inputs in %s", output_dir)
    return {
        "output_dir": str(output_dir),
        "production_systems": str(production),
        "manure_fractions": str(manure_fractions),
        "manure_management": str(manure_management),
        "animal_isodata": animal_isodata,
        "animal_isoraster": str(animal_isoraster),
        "animal_heads": animal_heads,
    }
