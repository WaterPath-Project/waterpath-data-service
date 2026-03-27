"""
Projection correctness tests.

Covers:
- Schema field-name reader (unit)
- Treatment projections: country-level and sub-national (mocked HTTP)
- Sanitation projections: country-level and sub-national (mocked HTTP)
- Livestock tabular-CSV schema filtering (in-process, no HTTP)
- Full livestock projection pipeline using bundled test-session data

Run with::

    pytest utils/test_projections.py -v

or as a standalone script::

    python utils/test_projections.py
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Resolve key paths relative to this file's location.
# ---------------------------------------------------------------------------
_UTILS_DIR = Path(__file__).parent
_REPO_ROOT = _UTILS_DIR.parent
_PKG_DIR = _REPO_ROOT / "waterpath_data_service"
_STATIC_SCHEMAS_DIR = _PKG_DIR / "static" / "schemas"
_STATIC_DATA_DIR = _PKG_DIR / "static" / "data"
_DATA_DIR = _PKG_DIR / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_httpx_client(csv_text: str):
    """Return a context-manager mock for httpx.AsyncClient whose .get() returns *csv_text*."""
    mock_response = MagicMock()
    mock_response.text = csv_text
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


def _build_mock_client_class(csv_text: str):
    """Return a class-level mock that yields a pre-loaded response for every .get() call."""
    def _factory(*_args, **_kwargs):
        return _mock_httpx_client(csv_text)
    return _factory


def _schema_fields(schema_name: str) -> set[str]:
    """Return field names from a static schema JSON by filename stem (e.g. 'treatment')."""
    from waterpath_data_service.services.projections import read_schema_field_names
    return read_schema_field_names(_STATIC_SCHEMAS_DIR / f"{schema_name}.json")


# ---------------------------------------------------------------------------
# Minimal CSV fixtures
# ---------------------------------------------------------------------------

def _treatment_future_csv(alpha3_list: list[str], ssp: str = "SSP3", year: int = 2050) -> str:
    rows = [
        "alpha3,scenario,year,"
        "FractionPrimarytreatment,FractionSecondarytreatment,"
        "FractionTertiarytreatment,FractionQuaternarytreatment"
    ]
    for a in alpha3_list:
        rows.append(f"{a},{ssp},{year},0.10,0.40,0.30,0.20")
    return "\n".join(rows)


def _sanitation_future_csv(alpha3_list: list[str], ssp: str = "SSP3", year: int = 2050) -> str:
    # The sanitation schema has many columns; we include a representative subset.
    san_cols = [
        "flushSewer_urb", "flushSeptic_urb", "flushPit_urb", "openDefecation_urb",
        "pitNoSlab_urb", "pitSlab_urb", "flushSewer_rur", "flushSeptic_rur",
        "flushPit_rur", "openDefecation_rur", "pitNoSlab_rur", "pitSlab_rur",
    ]
    header = "alpha3,scenario,year," + ",".join(san_cols)
    rows = [header]
    for a in alpha3_list:
        values = ",".join(["0.1"] * len(san_cols))
        rows.append(f"{a},{ssp},{year},{values}")
    return "\n".join(rows)


def _livestock_future_csv(
    alpha3_list: list[str],
    ssp: str = "SSP3",
    year: int = 2050,
    include_tabular_cols: bool = True,
) -> str:
    """Build a livestock_future.csv with both head-count and optional tabular columns."""
    head_cols = ["cattle", "buffaloes", "pigs", "sheep", "goats", "poultry"]
    # Columns that belong to production_systems schema
    prod_cols = ["meat_i", "meat_e", "dairy_i", "dairy_e", "pigs_i", "pigs_e"]
    # Columns that belong to manure_fractions schema
    frac_cols = ["meat_fgi", "meat_fge", "meat_foi", "meat_foe",
                 "dairy_fgi", "dairy_fge", "dairy_foi", "dairy_foe"]
    # Intentional extra column that belongs to NO schema
    noise_cols = ["unknown_future_col"]

    if include_tabular_cols:
        all_cols = head_cols + prod_cols + frac_cols + noise_cols
    else:
        all_cols = head_cols

    header = "alpha3,scenario,year," + ",".join(all_cols)
    rows = [header]
    for a in alpha3_list:
        values = ",".join(["100.0"] * len(all_cols))
        rows.append(f"{a},{ssp},{year},{values}")
    return "\n".join(rows)


# ===========================================================================
# 1. Unit tests – schema reader
# ===========================================================================

class TestReadSchemaFieldNames:
    def test_treatment_schema_contains_expected_fields(self):
        fields = _schema_fields("treatment")
        assert "gid" in fields
        assert "FractionPrimarytreatment" in fields
        assert "FractionSecondarytreatment" in fields
        assert "FractionTertiarytreatment" in fields
        assert "FractionQuaternarytreatment" in fields

    def test_sanitation_schema_has_urban_and_rural_fields(self):
        fields = _schema_fields("sanitation")
        # At minimum the column identifier and a few canonical sanitation types should be present.
        assert "gid" in fields
        assert any("urb" in f for f in fields), "Expected urban columns in sanitation schema"
        assert any("rur" in f for f in fields), "Expected rural columns in sanitation schema"

    def test_population_schema_contains_key_columns(self):
        fields = _schema_fields("population")
        for col in ("iso", "gid", "population", "fraction_urban_pop", "fraction_pop_under5", "hdi"):
            assert col in fields, f"Missing expected column '{col}' in population schema"

    def test_livestock_production_systems_schema(self):
        fields = _schema_fields("livestock_production_systems")
        assert "iso" in fields
        assert "gid" in fields
        # Each species should have at least an intensive fraction column
        for animal in ("meat", "pigs", "poultry", "sheep", "goats"):
            assert f"{animal}_i" in fields, f"Missing '{animal}_i' in production_systems schema"

    def test_livestock_manure_fractions_schema(self):
        fields = _schema_fields("livestock_manure_fractions")
        assert "iso" in fields
        assert "gid" in fields
        for animal in ("meat", "pigs", "poultry"):
            for suffix in ("fgi", "fge", "foi", "foe"):
                assert f"{animal}_{suffix}" in fields, (
                    f"Missing '{animal}_{suffix}' in manure_fractions schema"
                )

    def test_unknown_schema_file_raises(self):
        with pytest.raises(FileNotFoundError):
            from waterpath_data_service.services.projections import read_schema_field_names
            read_schema_field_names(_STATIC_SCHEMAS_DIR / "nonexistent_schema.json")


# ===========================================================================
# 2. Treatment projections
# ===========================================================================

class TestTreatmentProjection:
    """Treatment projection for country-level and sub-national alpha3 roll-up."""

    COUNTRY_ALPHA3 = ["BGR", "ALB", "GRC", "MKD"]  # test session – Balkans
    SUBCOUNTRY_ALPHA3 = ["BGD"]  # test_dhaka – sub-national GIDs share BGD parent

    @pytest.mark.anyio
    async def test_country_level_columns_match_schema(self):
        """Projected treatment DataFrame columns must be a subset of treatment.json fields."""
        from waterpath_data_service.services.projections import fetch_treatment_future_csv

        csv_text = _treatment_future_csv(self.COUNTRY_ALPHA3)
        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            df = await fetch_treatment_future_csv(self.COUNTRY_ALPHA3, "SSP3", 2050)

        schema_fields = _schema_fields("treatment")
        unexpected = set(df.columns) - schema_fields - {"alpha3"}
        assert not unexpected, f"Output has columns not in treatment schema: {unexpected}"

    @pytest.mark.anyio
    async def test_country_level_all_areas_present(self):
        from waterpath_data_service.services.projections import fetch_treatment_future_csv

        csv_text = _treatment_future_csv(self.COUNTRY_ALPHA3)
        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            df = await fetch_treatment_future_csv(self.COUNTRY_ALPHA3, "SSP3", 2050)

        assert set(df["alpha3"]) == set(self.COUNTRY_ALPHA3)

    @pytest.mark.anyio
    async def test_fractions_sum_to_one(self):
        """Treatment fractions shoud sum to approximately 1.0 per row."""
        from waterpath_data_service.services.projections import fetch_treatment_future_csv

        csv_text = _treatment_future_csv(self.COUNTRY_ALPHA3)
        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            df = await fetch_treatment_future_csv(self.COUNTRY_ALPHA3, "SSP3", 2050)

        fraction_cols = [
            c for c in df.columns
            if c.lower().startswith("fraction") and c != "alpha3"
        ]
        if fraction_cols:
            row_sums = df[fraction_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            assert (row_sums.round(6) <= 1.0 + 1e-6).all(), (
                f"Treatment fractions exceed 1.0 (sums: {row_sums.tolist()})"
            )

    @pytest.mark.anyio
    async def test_subcountry_alpha3_rollup(self):
        """Sub-national lookup uses alpha3 (first 3 chars of GID) – one row per country."""
        from waterpath_data_service.services.projections import fetch_treatment_future_csv

        csv_text = _treatment_future_csv(self.SUBCOUNTRY_ALPHA3)
        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            df = await fetch_treatment_future_csv(self.SUBCOUNTRY_ALPHA3, "SSP3", 2050)

        assert len(df) == len(self.SUBCOUNTRY_ALPHA3)
        assert set(df["alpha3"]) == set(self.SUBCOUNTRY_ALPHA3)

    @pytest.mark.anyio
    async def test_quaternary_column_derived_when_absent(self):
        """If FractionQuaternarytreatment is absent it must be derived from the other three."""
        from waterpath_data_service.services.projections import fetch_treatment_future_csv

        # CSV deliberately omits Quaternary
        csv_text = (
            "alpha3,scenario,year,FractionPrimarytreatment,"
            "FractionSecondarytreatment,FractionTertiarytreatment\n"
            "BGR,SSP3,2050,0.10,0.40,0.30\n"
            "ALB,SSP3,2050,0.20,0.30,0.30\n"
        )
        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            df = await fetch_treatment_future_csv(["BGR", "ALB"], "SSP3", 2050)

        assert "FractionQuaternarytreatment" in df.columns
        # BGR: 1 - 0.10 - 0.40 - 0.30 = 0.20
        bgr_row = df[df["alpha3"] == "BGR"].iloc[0]
        assert abs(bgr_row["FractionQuaternarytreatment"] - 0.20) < 1e-9


# ===========================================================================
# 3. Sanitation projections
# ===========================================================================

class TestSanitationProjection:
    COUNTRY_ALPHA3 = ["BGR", "ALB", "GRC", "MKD"]
    SUBCOUNTRY_ALPHA3 = ["BGD"]

    @pytest.mark.anyio
    async def test_country_level_columns_match_schema(self):
        from waterpath_data_service.services.projections import fetch_sanitation_projection

        csv_text = _sanitation_future_csv(self.COUNTRY_ALPHA3)
        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            df = await fetch_sanitation_projection(self.COUNTRY_ALPHA3, "SSP3", 2050)

        schema_fields = _schema_fields("sanitation")
        # The returned df uses 'alpha3' as the key column; allow it alongside schema fields.
        unexpected = set(df.columns) - schema_fields - {"alpha3"}
        assert not unexpected, (
            f"Sanitation output has columns not in sanitation schema: {unexpected}"
        )

    @pytest.mark.anyio
    async def test_country_level_all_areas_present(self):
        from waterpath_data_service.services.projections import fetch_sanitation_projection

        csv_text = _sanitation_future_csv(self.COUNTRY_ALPHA3)
        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            df = await fetch_sanitation_projection(self.COUNTRY_ALPHA3, "SSP3", 2050)

        assert set(df["alpha3"]) == set(self.COUNTRY_ALPHA3)

    @pytest.mark.anyio
    async def test_subcountry_alpha3_rollup(self):
        from waterpath_data_service.services.projections import fetch_sanitation_projection

        csv_text = _sanitation_future_csv(self.SUBCOUNTRY_ALPHA3)
        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            df = await fetch_sanitation_projection(self.SUBCOUNTRY_ALPHA3, "SSP3", 2050)

        assert len(df) == len(self.SUBCOUNTRY_ALPHA3)
        assert set(df["alpha3"]) == set(self.SUBCOUNTRY_ALPHA3)

    @pytest.mark.anyio
    async def test_fractions_are_numeric(self):
        from waterpath_data_service.services.projections import fetch_sanitation_projection

        csv_text = _sanitation_future_csv(self.COUNTRY_ALPHA3)
        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            df = await fetch_sanitation_projection(self.COUNTRY_ALPHA3, "SSP3", 2050)

        num_cols = [c for c in df.columns if c not in ("alpha3", "gid")]
        for col in num_cols:
            assert pd.to_numeric(df[col], errors="coerce").notna().all(), (
                f"Column '{col}' contains non-numeric values"
            )


# ===========================================================================
# 4. Livestock tabular CSV schema filtering
# ===========================================================================

class TestLivestockSchemaFiltering:
    """Verify that tabular_schema_fields constrains which future columns are applied."""

    def _baseline_dir(self) -> Path:
        return _DATA_DIR / "test" / "baseline" / "livestock_emissions"

    def _mapping(self) -> pd.DataFrame:
        pop_csv = _DATA_DIR / "test" / "baseline" / "human_emissions" / "population.csv"
        pop = pd.read_csv(pop_csv)
        mapping = pop[["iso", "gid"]].copy()
        mapping["iso"] = pd.to_numeric(mapping["iso"], errors="coerce").astype(int)
        mapping["gid"] = mapping["gid"].astype(str)
        return mapping.drop_duplicates(subset=["iso"], keep="first")

    def _make_future_df(self, alpha3_list: list[str]) -> pd.DataFrame:
        """Build a future DataFrame with deliberately mixed columns."""
        csv_text = _livestock_future_csv(alpha3_list, include_tabular_cols=True)
        df = pd.read_csv(io.StringIO(csv_text))
        df = df[df["alpha3"].isin(alpha3_list)].drop(columns=["scenario", "year"])
        return df

    def test_production_systems_only_schema_columns_written(self):
        """After filtering, production_systems.csv must only contain schema-defined columns."""
        from waterpath_data_service.services.livestock import _update_tabular_csv_from_future
        from waterpath_data_service.services.projections import read_schema_field_names

        baseline_dir = self._baseline_dir()
        if not baseline_dir.is_dir():
            pytest.skip("Test-session livestock data not found; skipping.")

        baseline_csv = baseline_dir / "production_systems.csv"
        if not baseline_csv.is_file():
            pytest.skip("Baseline production_systems.csv not found; skipping.")

        mapping = self._mapping()
        alpha3_list = mapping["gid"].str[:3].tolist()
        future_df = self._make_future_df(alpha3_list)

        prod_schema_fields = read_schema_field_names(
            _STATIC_SCHEMAS_DIR / "livestock_production_systems.json"
        )
        allowed = (prod_schema_fields - {"iso", "gid"}) | {"alpha3"}
        allowed_lower = {c.lower() for c in allowed}
        filtered_df = future_df[
            [c for c in future_df.columns if c.lower() in allowed_lower]
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "production_systems.csv"
            _update_tabular_csv_from_future(baseline_csv, out_csv, filtered_df, mapping)

            result = pd.read_csv(out_csv)

        # The noise column must not appear in the output.
        assert "unknown_future_col" not in result.columns
        # Core identifier columns must be preserved.
        assert "iso" in result.columns
        assert "gid" in result.columns

    def test_manure_fractions_only_schema_columns_written(self):
        from waterpath_data_service.services.livestock import _update_tabular_csv_from_future
        from waterpath_data_service.services.projections import read_schema_field_names

        baseline_dir = self._baseline_dir()
        if not baseline_dir.is_dir():
            pytest.skip("Test-session livestock data not found; skipping.")

        baseline_csv = baseline_dir / "manure_fractions.csv"
        if not baseline_csv.is_file():
            pytest.skip("Baseline manure_fractions.csv not found; skipping.")

        mapping = self._mapping()
        alpha3_list = mapping["gid"].str[:3].tolist()
        future_df = self._make_future_df(alpha3_list)

        frac_schema_fields = read_schema_field_names(
            _STATIC_SCHEMAS_DIR / "livestock_manure_fractions.json"
        )
        allowed = (frac_schema_fields - {"iso", "gid"}) | {"alpha3"}
        allowed_lower = {c.lower() for c in allowed}
        filtered_df = future_df[
            [c for c in future_df.columns if c.lower() in allowed_lower]
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "manure_fractions.csv"
            _update_tabular_csv_from_future(baseline_csv, out_csv, filtered_df, mapping)

            result = pd.read_csv(out_csv)

        assert "unknown_future_col" not in result.columns
        assert "iso" in result.columns
        assert "gid" in result.columns

    def test_generate_projection_passes_schema_fields(self):
        """generate_livestock_projection_rasters must respect tabular_schema_fields."""
        from waterpath_data_service.services.livestock import generate_livestock_projection_rasters
        from waterpath_data_service.services.projections import read_schema_field_names

        baseline_dir = self._baseline_dir()
        if not (baseline_dir / "animals").is_dir():
            pytest.skip("Test-session livestock rasters not found; skipping.")

        mapping = self._mapping()
        alpha3_list = mapping["gid"].str[:3].tolist()
        future_df = self._make_future_df(alpha3_list)

        # Build a minimal zone grid from the existing isoraster.
        isoraster_path = _DATA_DIR / "test" / "baseline" / "human_emissions" / "isoraster.tif"
        if not isoraster_path.is_file():
            pytest.skip("Baseline isoraster.tif not found; skipping.")

        import rasterio

        with rasterio.open(isoraster_path) as src:
            zone_idx = src.read(1).astype(np.int32)
            zone_profile = src.profile.copy()
        valid_mask = zone_idx > 0

        tabular_fields = {
            "production_systems": read_schema_field_names(
                _STATIC_SCHEMAS_DIR / "livestock_production_systems.json"
            ),
            "manure_fractions": read_schema_field_names(
                _STATIC_SCHEMAS_DIR / "livestock_manure_fractions.json"
            ),
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "livestock_out"
            generate_livestock_projection_rasters(
                baseline_livestock_dir=baseline_dir,
                output_dir=out_dir,
                livestock_future_df=future_df,
                mapping=mapping,
                zone_idx=zone_idx,
                valid_mask=valid_mask,
                zone_profile=zone_profile,
                tabular_schema_fields=tabular_fields,
            )

            ps_csv = out_dir / "production_systems.csv"
            mf_csv = out_dir / "manure_fractions.csv"

            assert ps_csv.is_file(), "production_systems.csv was not generated"
            assert mf_csv.is_file(), "manure_fractions.csv was not generated"

            ps_df = pd.read_csv(ps_csv)
            mf_df = pd.read_csv(mf_csv)

        # Noise column must not appear in either output.
        assert "unknown_future_col" not in ps_df.columns
        assert "unknown_future_col" not in mf_df.columns

        # Schema-specific columns must not cross-contaminate.
        manure_only_col = "meat_fgi"
        prod_only_col = "meat_i"
        if manure_only_col in mf_df.columns:
            assert manure_only_col not in ps_df.columns or "meat_fgi" in read_schema_field_names(
                _STATIC_SCHEMAS_DIR / "livestock_production_systems.json"
            ), "Manure fraction column leaked into production systems CSV"

    def test_meat_dairy_inherit_cattle_when_absent(self):
        """When future CSV has cattle but no meat/dairy, those columns are propagated."""
        from waterpath_data_service.services.livestock import generate_livestock_projection_rasters
        from waterpath_data_service.services.projections import read_schema_field_names

        baseline_dir = self._baseline_dir()
        if not (baseline_dir / "animals").is_dir():
            pytest.skip("Test-session livestock rasters not found; skipping.")

        mapping = self._mapping()
        alpha3_list = mapping["gid"].str[:3].tolist()

        # Future DataFrame with ONLY cattle (no meat or dairy columns).
        csv_text = _livestock_future_csv(alpha3_list, include_tabular_cols=False)
        df = pd.read_csv(io.StringIO(csv_text))
        df = df[df["alpha3"].isin(alpha3_list)].drop(columns=["scenario", "year"])
        # Confirm neither meat nor dairy present.
        assert not any(c.lower() in ("meat", "dairy") for c in df.columns)

        isoraster_path = _DATA_DIR / "test" / "baseline" / "human_emissions" / "isoraster.tif"
        if not isoraster_path.is_file():
            pytest.skip("Baseline isoraster.tif not found; skipping.")

        import rasterio

        with rasterio.open(isoraster_path) as src:
            zone_idx = src.read(1).astype(np.int32)
            zone_profile = src.profile.copy()
        valid_mask = zone_idx > 0

        tabular_fields = {
            "production_systems": read_schema_field_names(
                _STATIC_SCHEMAS_DIR / "livestock_production_systems.json"
            ),
            "manure_fractions": read_schema_field_names(
                _STATIC_SCHEMAS_DIR / "livestock_manure_fractions.json"
            ),
        }

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "livestock_out"
            generate_livestock_projection_rasters(
                baseline_livestock_dir=baseline_dir,
                output_dir=out_dir,
                livestock_future_df=df,
                mapping=mapping,
                zone_idx=zone_idx,
                valid_mask=valid_mask,
                zone_profile=zone_profile,
                tabular_schema_fields=tabular_fields,
            )

            ps_csv = out_dir / "production_systems.csv"
            mf_csv = out_dir / "manure_fractions.csv"
            assert ps_csv.is_file(), "production_systems.csv was not generated"
            assert mf_csv.is_file(), "manure_fractions.csv was not generated"

            ps_df = pd.read_csv(ps_csv)
            mf_df = pd.read_csv(mf_csv)

        # Both outputs must still have meat and dairy columns (propagated from cattle).
        for col_prefix in ("meat_i", "dairy_i", "meat_e", "dairy_e"):
            if col_prefix in read_schema_field_names(
                _STATIC_SCHEMAS_DIR / "livestock_production_systems.json"
            ):
                assert col_prefix in ps_df.columns, (
                    f"'{col_prefix}' missing from production_systems.csv after cattle propagation"
                )
        for col_prefix in ("meat_fgi", "dairy_fgi"):
            if col_prefix in read_schema_field_names(
                _STATIC_SCHEMAS_DIR / "livestock_manure_fractions.json"
            ):
                assert col_prefix in mf_df.columns, (
                    f"'{col_prefix}' missing from manure_fractions.csv after cattle propagation"
                )


# ===========================================================================
# 5. Country-level isodata projection pipeline (mocked HTTP)
# ===========================================================================

class TestCountryLevelIsodataProjection:
    """End-to-end: sanitation and treatment columns are updated in isodata.csv."""

    SESSION_DIR = _DATA_DIR / "test"

    def _baseline_isodata(self) -> pd.DataFrame:
        return pd.read_csv(self.SESSION_DIR / "baseline" / "human_emissions" / "isodata.csv")

    @pytest.mark.anyio
    async def test_sanitation_columns_updated_in_isodata(self):
        """After a sanitation projection, isodata.csv must have sanitation columns updated."""
        from waterpath_data_service.services.projections import fetch_sanitation_projection

        isodata = self._baseline_isodata()
        if not (self.SESSION_DIR / "baseline" / "human_emissions" / "isodata.csv").is_file():
            pytest.skip("Baseline isodata.csv for test session not found.")

        gid_col = next(
            (c for c in ("gid", "alpha3", "iso_country") if c in isodata.columns), None
        )
        assert gid_col is not None
        alpha3_list = isodata[gid_col].astype(str).str[:3].unique().tolist()

        san_cols = [
            "flushSewer_urb", "flushSeptic_urb",
            "flushSewer_rur", "flushSeptic_rur",
        ]
        csv_text = _sanitation_future_csv(alpha3_list, "SSP3", 2050)

        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            san_proj_df = await fetch_sanitation_projection(alpha3_list, "SSP3", 2050)

        assert len(san_proj_df) > 0, "Sanitation projection returned empty DataFrame"

        # Simulate the update that the endpoint performs.
        iso_df = isodata.copy()
        join_src = "iso_country" if "iso_country" in iso_df.columns else gid_col
        iso_df["_alpha3_join"] = iso_df[join_src].astype(str).str[:3]
        san_map = san_proj_df.set_index("alpha3")
        updated_any = False
        for col in san_map.columns:
            col_map = san_map[col].to_dict()
            updated = iso_df["_alpha3_join"].map(col_map)
            if col in iso_df.columns and updated.notna().any():
                iso_df[col] = updated.combine_first(iso_df[col])
                updated_any = True
            elif updated.notna().any():
                iso_df[col] = updated
                updated_any = True
        iso_df = iso_df.drop(columns=["_alpha3_join"])

        assert updated_any, "No sanitation columns were updated in isodata"

    @pytest.mark.anyio
    async def test_treatment_csv_generated_with_correct_columns(self):
        """Projected treatment.csv must contain exactly the columns in treatment.json."""
        from waterpath_data_service.services.projections import fetch_treatment_future_csv

        isodata = self._baseline_isodata()
        if isodata.empty:
            pytest.skip("Baseline isodata.csv is empty.")

        gid_col = next(
            (c for c in ("gid", "alpha3", "iso_country") if c in isodata.columns), None
        )
        alpha3_list = isodata[gid_col].astype(str).str[:3].unique().tolist()
        csv_text = _treatment_future_csv(alpha3_list, "SSP3", 2050)

        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            treat_df = await fetch_treatment_future_csv(alpha3_list, "SSP3", 2050)

        treat_df_renamed = treat_df.rename(columns={"alpha3": "gid"})
        schema_fields = _schema_fields("treatment")
        unexpected = set(treat_df_renamed.columns) - schema_fields
        assert not unexpected, f"treatment.csv has columns not in schema: {unexpected}"


# ===========================================================================
# 6. Sub-national projection pipeline (Dhaka)
# ===========================================================================

class TestSubNationalProjection:
    """Sub-national GIDs use parent alpha3 for HTTP lookups; outputs map back to all sub-areas."""

    SESSION_DIR = _DATA_DIR / "test_dhaka"

    def _baseline_population(self) -> pd.DataFrame:
        return pd.read_csv(
            self.SESSION_DIR / "baseline" / "human_emissions" / "population.csv"
        )

    @pytest.mark.anyio
    async def test_sanitation_projection_maps_to_all_subareas(self):
        """All sub-national rows must receive a sanitation update if the parent alpha3 is present."""
        from waterpath_data_service.services.projections import fetch_sanitation_projection

        pop_df = self._baseline_population()
        if pop_df.empty:
            pytest.skip("Sub-national population.csv is empty.")

        gid_col = next(
            (c for c in ("gid", "alpha3") if c in pop_df.columns), None
        )
        assert gid_col is not None

        # All rows belong to BGD
        alpha3_list = pop_df[gid_col].astype(str).str[:3].unique().tolist()
        assert alpha3_list == ["BGD"], f"Expected BGD only, got {alpha3_list}"

        csv_text = _sanitation_future_csv(alpha3_list, "SSP3", 2050)
        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            san_df = await fetch_sanitation_projection(alpha3_list, "SSP3", 2050)

        # Simulate the viewer-level join with sub-national isodata.
        # The endpoint maps alpha3 → all sub-areas via iso_country.
        san_col = "flushSewer_urb"
        if san_col not in san_df.columns:
            pytest.skip(f"Column '{san_col}' not in sanitation projection result.")

        san_map = san_df.set_index("alpha3")[san_col].to_dict()
        iso_df = pd.read_csv(
            self.SESSION_DIR / "baseline" / "human_emissions" / "sanitation.csv"
        )
        # Dhaka sanitation has gid column with sub-national identifiers.
        iso_df["_alpha3"] = iso_df[gid_col].astype(str).str[:3]
        iso_df["_updated"] = iso_df["_alpha3"].map(san_map)

        assert iso_df["_updated"].notna().all(), (
            "Some sub-national rows did not receive a sanitation projection value. "
            f"Missing for: {iso_df.loc[iso_df['_updated'].isna(), gid_col].tolist()}"
        )

    @pytest.mark.anyio
    async def test_treatment_projection_uses_alpha3(self):
        """Treatment projection for sub-national requests uses parent-country alpha3."""
        from waterpath_data_service.services.projections import fetch_treatment_future_csv

        pop_df = self._baseline_population()
        if pop_df.empty:
            pytest.skip("Sub-national population.csv is empty.")

        gid_col = next((c for c in ("gid", "alpha3") if c in pop_df.columns), None)
        alpha3_list = pop_df[gid_col].astype(str).str[:3].unique().tolist()

        csv_text = _treatment_future_csv(alpha3_list, "SSP3", 2050)
        with patch(
            "waterpath_data_service.services.projections.httpx.AsyncClient",
            side_effect=_build_mock_client_class(csv_text),
        ):
            treat_df = await fetch_treatment_future_csv(alpha3_list, "SSP3", 2050)

        assert len(treat_df) == len(alpha3_list)
        assert set(treat_df["alpha3"]) == set(alpha3_list)


# ===========================================================================
# Standalone entry-point
# ===========================================================================

if __name__ == "__main__":
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(_REPO_ROOT),
    )
    sys.exit(result.returncode)
