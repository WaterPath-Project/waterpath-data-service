from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


DRIVER_METRIC_DEFS = [
    {"key": "population_total", "driver": "Population", "label": "Total population", "delta_mode": "relative_pct", "value_format": "integer", "color_direction": "neutral"},
    {"key": "population_urban_mean_pct", "driver": "Population", "label": "Mean urban fraction", "delta_mode": "pp", "value_format": "percent", "color_direction": "neutral"},
    {"key": "population_under5_mean_pct", "driver": "Population", "label": "Mean under-5 fraction", "delta_mode": "pp", "value_format": "percent", "color_direction": "neutral"},
    {"key": "population_hdi_mean", "driver": "Population", "label": "Mean HDI", "delta_mode": "absolute", "value_format": "hdi", "color_direction": "positive_good"},
    {"key": "sanitation_improved_pct", "driver": "Sanitation", "label": "Improved %", "delta_mode": "pp", "value_format": "percent", "color_direction": "positive_good"},
    {"key": "sanitation_unimproved_pct", "driver": "Sanitation", "label": "Unimproved %", "delta_mode": "pp", "value_format": "percent", "color_direction": "negative_good"},
    {"key": "sanitation_open_defecation_pct", "driver": "Sanitation", "label": "Open defecation %", "delta_mode": "pp", "value_format": "percent", "color_direction": "negative_good"},
    {"key": "wastewater_sewage_treated_pct", "driver": "Wastewater treatment", "label": "Sewage treated %", "delta_mode": "pp", "value_format": "percent", "color_direction": "positive_good"},
    {"key": "wastewater_fecal_sludge_treated_pct", "driver": "Wastewater treatment", "label": "Fecal sludge treated %", "delta_mode": "pp", "value_format": "percent", "color_direction": "positive_good"},
    {"key": "wastewater_facility_count", "driver": "Wastewater treatment", "label": "Number of treatment facilities", "delta_mode": "relative_pct", "value_format": "integer", "color_direction": "positive_good"},
    {"key": "wastewater_total_capacity", "driver": "Wastewater treatment", "label": "Total treatment capacity", "delta_mode": "relative_pct", "value_format": "integer", "color_direction": "positive_good"},
    {"key": "wastewater_share_primary_pct", "driver": "Wastewater treatment", "label": "Share of Primary", "delta_mode": "pp", "value_format": "percent", "color_direction": "negative_good"},
    {"key": "wastewater_share_secondary_pct", "driver": "Wastewater treatment", "label": "Share of Secondary", "delta_mode": "pp", "value_format": "percent", "color_direction": "positive_good"},
    {"key": "wastewater_share_tertiary_pct", "driver": "Wastewater treatment", "label": "Share of Tertiary", "delta_mode": "pp", "value_format": "percent", "color_direction": "positive_good"},
    {"key": "wastewater_share_quaternary_pct", "driver": "Wastewater treatment", "label": "Share of Quaternary", "delta_mode": "pp", "value_format": "percent", "color_direction": "positive_good"},
    {"key": "livestock_mean_population_growth", "driver": "Livestock population", "label": "Mean Population growth (all animals)", "delta_mode": "relative_pct", "value_format": "integer", "color_direction": "neutral"},
    {"key": "manure_direct_land_application_pct", "driver": "Manure management", "label": "Directly applied to land %", "delta_mode": "pp", "value_format": "percent", "color_direction": "negative_good"},
    {"key": "manure_storage_pct", "driver": "Manure management", "label": "Stored before application %", "delta_mode": "pp", "value_format": "percent", "color_direction": "positive_good"},
    {"key": "manure_treated_pct", "driver": "Manure management", "label": "Digested or burned %", "delta_mode": "pp", "value_format": "percent", "color_direction": "positive_good"},
    {"key": "production_mean_progress_intensive_pct", "driver": "Production systems", "label": "Mean progress towards intensive", "delta_mode": "pp", "value_format": "percent", "color_direction": "neutral"},
    {"key": "hydrology_mean_annual_discharge", "driver": "Hydrology", "label": "Mean river discharge (m³/s)", "delta_mode": "relative_pct", "value_format": "decimal", "color_direction": "positive_good"},
    {"key": "hydrology_mean_annual_runoff", "driver": "Hydrology", "label": "Mean surface runoff (mm/day)", "delta_mode": "relative_pct", "value_format": "decimal", "color_direction": "negative_good"},
    {"key": "hydrology_mean_river_temperature", "driver": "Hydrology", "label": "Mean river temperature (°C)", "delta_mode": "absolute", "value_format": "decimal", "color_direction": "positive_good"},
    {"key": "hydrology_mean_ssrd", "driver": "Hydrology", "label": "Mean solar radiation (W/m²)", "delta_mode": "relative_pct", "value_format": "decimal", "color_direction": "positive_good"},
    {"key": "exposure_drinking_events_per_year", "driver": "Exposure pathways", "label": "Drinking", "delta_mode": "relative_pct", "value_format": "integer", "color_direction": "neutral"},
    {"key": "exposure_swimming_events_per_year", "driver": "Exposure pathways", "label": "Swimming", "delta_mode": "relative_pct", "value_format": "integer", "color_direction": "neutral"},
    {"key": "exposure_flooding_events_per_year", "driver": "Exposure pathways", "label": "Flooding", "delta_mode": "relative_pct", "value_format": "integer", "color_direction": "neutral"},
    {"key": "exposure_open_drain_events_per_year", "driver": "Exposure pathways", "label": "Open drain", "delta_mode": "relative_pct", "value_format": "integer", "color_direction": "neutral"},
    {"key": "exposure_playing_events_per_year", "driver": "Exposure pathways", "label": "Playing", "delta_mode": "relative_pct", "value_format": "integer", "color_direction": "neutral"},
    {"key": "exposure_washing_clothes_events_per_year", "driver": "Exposure pathways", "label": "Washing clothes", "delta_mode": "relative_pct", "value_format": "integer", "color_direction": "neutral"},
]

IMPROVED_SOURCES = ("flushSewer", "flushSeptic", "flushPit", "pitSlab", "compostingToilet", "containerBased")
UNIMPROVED_SOURCES = ("pitNoSlab", "bucketLatrine", "hangingToilet", "flushOpen", "flushUnknown", "other")
TREATMENT_FIELDS = (
    "FractionPrimarytreatment",
    "FractionSecondarytreatment",
    "FractionTertiarytreatment",
    "FractionQuaternarytreatment",
)
EXPOSURE_ROUTES = ("drinking", "swimming", "flooding", "open_drain", "playing", "washing_clothes")
DEFAULT_EXPOSURE_FREQUENCIES = {
    "drinking": {"type": "fixed", "value": 365},
    "swimming": {"type": "nbinom", "size": 0.4, "prob": 0.11},
    "flooding": {"type": "poisson", "lambda": 1.0},
    "open_drain": {"type": "poisson", "lambda": 200.0},
    "playing": {"type": "poisson", "lambda": 30.0},
    "washing_clothes": {"type": "poisson", "lambda": 200.0},
}
_SCENARIO_RE = re.compile(r"^(SSP[1-5])_(\d{4})$", re.IGNORECASE)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _table_path(data_dir: Path, filename: str) -> Path:
    direct = data_dir / filename
    if direct.is_file():
        return direct
    return data_dir / "human_emissions" / filename


def _numbers(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _population_weights(frame: pd.DataFrame) -> pd.Series:
    return _numbers(frame, "population").clip(lower=0.0)


def _weighted_mean(frame: pd.DataFrame, column: str) -> float | None:
    weights = _population_weights(frame)
    if frame.empty or weights.sum() <= 0:
        return None
    return float(np.average(_numbers(frame, column), weights=weights))


def _weighted_split(frame: pd.DataFrame, urban_field: str, rural_field: str) -> float | None:
    weights = _population_weights(frame)
    if frame.empty or weights.sum() <= 0:
        return None
    urban_share = _numbers(frame, "fraction_urban_pop").clip(0.0, 1.0)
    values = urban_share * _numbers(frame, urban_field) + (1.0 - urban_share) * _numbers(frame, rural_field)
    return float(np.average(values, weights=weights))


def _sanitation_share(frame: pd.DataFrame, sources: tuple[str, ...]) -> float | None:
    weights = _population_weights(frame)
    if frame.empty or weights.sum() <= 0:
        return None
    urban_share = _numbers(frame, "fraction_urban_pop").clip(0.0, 1.0)
    urban = sum((_numbers(frame, f"{source}_urb") for source in sources), start=pd.Series(0.0, index=frame.index))
    rural = sum((_numbers(frame, f"{source}_rur") for source in sources), start=pd.Series(0.0, index=frame.index))
    return float(np.average(urban_share * urban + (1.0 - urban_share) * rural, weights=weights))


def _treatment_metrics(data_dir: Path, isodata: pd.DataFrame) -> tuple[str, dict[str, float | None]]:
    treatment = _read_csv(_table_path(data_dir, "treatment.csv"))
    area_mode = "FractionPrimarytreatment" in treatment.columns or "FractionPrimarytreatment" in isodata.columns
    result: dict[str, float | None] = {
        "facility_count": None,
        "total_capacity": None,
        "primary": None,
        "secondary": None,
        "tertiary": None,
        "quaternary": None,
    }
    if not area_mode:
        points = treatment.dropna(subset=[column for column in ("lon", "lat") if column in treatment.columns])
        if "lon" not in treatment.columns or "lat" not in treatment.columns:
            points = treatment.iloc[0:0]
        result["facility_count"] = float(len(points))
        result["total_capacity"] = float(_numbers(points, "capacity").clip(lower=0.0).sum())
        return "point", result

    source = isodata if "FractionPrimarytreatment" in isodata.columns else treatment
    for label, field in zip(("primary", "secondary", "tertiary", "quaternary"), TREATMENT_FIELDS):
        if field not in source.columns:
            continue
        value = _weighted_mean(source, field) if source is isodata else float(_numbers(source, field).mean())
        result[label] = 100.0 * value if value is not None else None
    return "area", result


def _raster_values(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        values = source.read(1).astype(np.float64)
        if source.nodata is not None:
            values[values == source.nodata] = np.nan
    values[values < 0] = np.nan
    return values


def _livestock_mean_heads(data_dir: Path) -> float | None:
    totals = []
    for path in (data_dir / "livestock_emissions" / "animals").glob("*_heads.tif"):
        value = float(np.nansum(_raster_values(path)))
        if np.isfinite(value):
            totals.append(value)
    return float(np.mean(totals)) if totals else None


def _manure_metrics(data_dir: Path) -> dict[str, float | None]:
    frame = _read_csv(data_dir / "livestock_emissions" / "manure_management.csv")
    groups = {
        "direct": {"PP", "DS"},
        "storage": {"SS", "DL", "LS", "UAL", "Pl1", "Ph1", "SSDL"},
        "treated": {"AD", "BF"},
    }
    by_animal: dict[str, dict[str, str]] = {}
    for column in frame.columns:
        if column in {"iso", "gid"} or "_" not in column:
            continue
        system, animal = column.rsplit("_", 1)
        by_animal.setdefault(animal, {})[system] = column
    shares = {key: [] for key in groups}
    for _, row in frame.iterrows():
        for systems in by_animal.values():
            values = {system: max(0.0, float(pd.to_numeric(row[column], errors="coerce") or 0.0)) for system, column in systems.items()}
            total = sum(values.values())
            if total <= 0:
                continue
            for key, codes in groups.items():
                shares[key].append(sum(value for system, value in values.items() if system in codes) / total)
    return {key: (100.0 * float(np.mean(values)) if values else None) for key, values in shares.items()}


def _production_intensive(data_dir: Path) -> float | None:
    frame = _read_csv(data_dir / "livestock_emissions" / "production_systems.csv")
    columns = [column for column in frame.columns if column.endswith("_i")]
    if not columns:
        return None
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return 100.0 * float(np.nanmean(values)) if np.isfinite(values).any() else None


def _hydrology_metrics(data_dir: Path) -> dict[str, float | None]:
    definitions = {
        "hydrology_mean_annual_discharge": "discharge",
        "hydrology_mean_annual_runoff": "runoff",
        "hydrology_mean_river_temperature": "river_temperature",
        "hydrology_mean_ssrd": "ssrd",
    }
    result = {}
    for key, variable in definitions.items():
        monthly_means = []
        for path in sorted((data_dir / "hydrology" / variable).glob(f"{variable}_m??.tif")):
            values = _raster_values(path)
            valid = values[np.isfinite(values)]
            if valid.size:
                monthly_means.append(float(valid.mean()))
        result[key] = float(np.mean(monthly_means)) if monthly_means else None
    return result


def _load_exposure_config(data_dir: Path, baseline_dir: Path) -> dict:
    for path in (data_dir / "qmra" / "qmra_config.json", baseline_dir / "qmra" / "qmra_config.json"):
        if path.is_file():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def _exposure_metrics(data_dir: Path, baseline_dir: Path) -> dict[str, float | None]:
    config = _load_exposure_config(data_dir, baseline_dir)
    frequencies = {}
    if isinstance(config.get("pathways"), dict):
        frequencies = {route: (value or {}).get("frequency") for route, value in config["pathways"].items()}
    for group in config.get("exposure_groups", []):
        route = group.get("route") or group.get("name")
        if route:
            frequencies[route] = group.get("frequency")

    def expected_events(frequency) -> float | None:
        if isinstance(frequency, (int, float)):
            return float(frequency)
        if not isinstance(frequency, dict):
            return None
        frequency_type = frequency.get("type") or frequency.get("dist")
        if frequency_type == "fixed":
            return float(frequency.get("value", 0.0))
        if frequency_type == "poisson":
            return float(frequency.get("lambda", 0.0))
        if frequency_type == "nbinom":
            size = float(frequency.get("size", 0.0))
            probability = float(frequency.get("prob", 0.0))
            return size * (1.0 - probability) / probability if probability else None
        return None

    result = {}
    for route in EXPOSURE_ROUTES:
        frequency = frequencies.get(route, DEFAULT_EXPOSURE_FREQUENCIES[route])
        result[f"exposure_{route}_events_per_year"] = expected_events(frequency)
    return result


def _scenario_metrics(data_dir: Path, baseline_dir: Path) -> tuple[str, dict[str, float | None]]:
    isodata = _read_csv(_table_path(data_dir, "isodata.csv"))
    treatment_mode, treatment = _treatment_metrics(data_dir, isodata)
    manure = _manure_metrics(data_dir)
    population_total = float(_population_weights(isodata).sum()) if not isodata.empty else None
    metrics = {
        "population_total": population_total,
        "population_urban_mean_pct": _percentage(_weighted_mean(isodata, "fraction_urban_pop")),
        "population_under5_mean_pct": _percentage(_weighted_mean(isodata, "fraction_pop_under5")),
        "population_hdi_mean": _weighted_mean(isodata, "hdi"),
        "sanitation_improved_pct": _percentage(_sanitation_share(isodata, IMPROVED_SOURCES)),
        "sanitation_unimproved_pct": _percentage(_sanitation_share(isodata, UNIMPROVED_SOURCES)),
        "sanitation_open_defecation_pct": _percentage(_sanitation_share(isodata, ("openDefecation",))),
        "wastewater_sewage_treated_pct": _percentage(_weighted_split(isodata, "sewageTreated_urb", "sewageTreated_rur")),
        "wastewater_fecal_sludge_treated_pct": _percentage(_weighted_split(isodata, "fecalSludgeTreated_urb", "fecalSludgeTreated_rur")),
        "wastewater_facility_count": treatment["facility_count"],
        "wastewater_total_capacity": treatment["total_capacity"],
        "wastewater_share_primary_pct": treatment["primary"],
        "wastewater_share_secondary_pct": treatment["secondary"],
        "wastewater_share_tertiary_pct": treatment["tertiary"],
        "wastewater_share_quaternary_pct": treatment["quaternary"],
        "livestock_mean_population_growth": _livestock_mean_heads(data_dir),
        "manure_direct_land_application_pct": manure["direct"],
        "manure_storage_pct": manure["storage"],
        "manure_treated_pct": manure["treated"],
        "production_mean_progress_intensive_pct": _production_intensive(data_dir),
        **_hydrology_metrics(data_dir),
        **_exposure_metrics(data_dir, baseline_dir),
    }
    return treatment_mode, metrics


def _percentage(value: float | None) -> float | None:
    return 100.0 * value if value is not None else None


def summarize_session(session_dir: Path) -> dict:
    baseline_dir = session_dir / "baseline"
    baseline_mode, baseline_metrics = _scenario_metrics(baseline_dir, baseline_dir)
    scenarios = [{
        "id": "baseline",
        "name": "Baseline",
        "year": "2025",
        "ssp": "",
        "is_baseline": True,
        "wwtp_mode": baseline_mode,
        "metrics": baseline_metrics,
    }]

    scenarios_dir = session_dir / "scenarios"
    scenario_entries = []
    if scenarios_dir.is_dir():
        for path in scenarios_dir.iterdir():
            match = _SCENARIO_RE.fullmatch(path.name)
            if not path.is_dir() or match is None:
                continue
            ssp, year = match.groups()
            treatment_mode, metrics = _scenario_metrics(path, baseline_dir)
            scenario_entries.append({
                "id": path.name,
                "name": path.name,
                "year": year,
                "ssp": ssp.upper(),
                "is_baseline": False,
                "wwtp_mode": treatment_mode,
                "metrics": metrics,
            })
    scenarios.extend(sorted(scenario_entries, key=lambda item: (int(item["year"]), item["name"])))
    return {
        "baseline_scenario_id": "baseline",
        "metrics": DRIVER_METRIC_DEFS,
        "scenarios": scenarios,
    }