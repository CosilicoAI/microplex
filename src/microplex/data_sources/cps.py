"""
CPS ASEC (Annual Social and Economic Supplement) data loading.

The CPS ASEC is the primary source for income and poverty statistics in the US.
Released annually in March, it contains detailed income, employment, and
demographic information for ~100K households.

Data source: https://www.census.gov/data/datasets/time-series/demo/cps/cps-asec.html
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl

from microplex.core import EntityType
from microplex.core.sources import (
    EntityObservation,
    EntityRelationship,
    ObservationFrame,
    RelationshipCardinality,
    Shareability,
    SourceArchetype,
    SourceDescriptor,
    SourceQuery,
    TimeStructure,
    apply_source_query,
)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "microplex"

# CPS ASEC data URLs by year
CPS_URLS = {
    2025: "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asecpub25csv.zip",
    2024: "https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub24csv.zip",
    2023: "https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip",
    2022: "https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip",
    2021: "https://www2.census.gov/programs-surveys/cps/datasets/2021/march/asecpub21csv.zip",
}

# Key variable mappings (Census variable name -> our name)
PERSON_VARIABLES = {
    # Demographics
    "A_AGE": "age",
    "A_SEX": "sex",
    "PRDTRACE": "race",
    "PEHSPNON": "hispanic",
    "A_HGA": "education",
    # Employment
    "A_CLSWKR": "class_of_worker",
    "A_WKSTAT": "work_status",
    "A_HRS1": "hours_worked",
    # Income (annual)
    "WSAL_VAL": "wage_income",
    "SEMP_VAL": "self_employment_income",
    "INT_VAL": "interest_income",
    "DIV_VAL": "dividend_income",
    "RNT_VAL": "rental_income",
    "SS_VAL": "social_security",
    "PNSN_VAL": "taxable_pension_income",
    "SSI_VAL": "ssi",
    "UC_VAL": "unemployment_compensation",
    "PTOTVAL": "total_person_income",
    # Benefits
    "PAW_VAL": "public_assistance",
    "MCARE": "has_medicare",
    "MCAID": "has_medicaid",
    # Identifiers
    "PH_SEQ": "household_id",
    "PF_SEQ": "family_id",
    "A_LINENO": "person_number",
    "A_FAMREL": "family_relationship",
    "A_MARITL": "marital_status",
    # Weights
    "A_FNLWGT": "weight",
    "MARSUPWT": "march_supplement_weight",
}

HOUSEHOLD_VARIABLES = {
    "H_SEQ": "household_id",
    "GESTFIPS": "state_fips",
    "GTCBSA": "cbsa",
    "HRHTYPE": "household_type",
    "H_NUMPER": "household_size",
    "HHINC": "household_income_bracket",
    "HTOTVAL": "household_total_income",
    "HSUP_WGT": "household_weight",
}

_CPS_TAX_UNIT_AGGREGATES = {
    "wage_income": "employment_income",
    "self_employment_income": "self_employment_income",
    "interest_income": "taxable_interest_income",
    "dividend_income": "ordinary_dividend_income",
    "rental_income": "rental_income",
    "social_security": "gross_social_security",
    "taxable_pension_income": "taxable_pension_income",
    "unemployment_compensation": "unemployment_compensation",
}

_CPS_TAX_UNIT_BASE_VARIABLES = (
    "household_id",
    "age",
    "is_female",
    "is_male",
    "is_household_head",
    "cps_race",
    "is_married",
    "filing_status",
    "earned_income",
    "ctc_qualifying_children",
    "own_children_in_household",
    "count_under_18",
    "agi_proxy",
    "interest_dividend_income",
    "social_security",
    "pension_income",
    "social_security_pension_income",
)

_CPS_TAX_UNIT_HOUSEHOLD_VARIABLES = (
    "state_fips",
    "cbsa",
    "household_size",
)


@dataclass
class CPSDataset:
    """Container for CPS ASEC data."""

    persons: pl.DataFrame
    households: pl.DataFrame
    year: int
    source: str

    @property
    def n_persons(self) -> int:
        return len(self.persons)

    @property
    def n_households(self) -> int:
        return len(self.households)

    def summary(self) -> dict:
        """Return summary statistics."""
        return {
            "year": self.year,
            "n_persons": self.n_persons,
            "n_households": self.n_households,
            "states": self.households["state_fips"].n_unique(),
            "total_weight": float(self.persons["weight"].sum()),
        }


def _variable_names(
    table: pd.DataFrame,
    *,
    key_column: str,
    weight_column: str | None = None,
    period_column: str | None = None,
) -> tuple[str, ...]:
    excluded = {key_column}
    if weight_column is not None:
        excluded.add(weight_column)
    if period_column is not None:
        excluded.add(period_column)
    return tuple(column for column in table.columns if column not in excluded)


def _to_pandas(frame: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    return frame.to_pandas()


def _ensure_column(table: pd.DataFrame, column: str, values) -> pd.DataFrame:
    if column in table.columns:
        return table
    result = table.copy()
    result.insert(0, column, values)
    return result


def _sort_by_key(table: pd.DataFrame, key_column: str) -> pd.DataFrame:
    return table.sort_values(key_column).reset_index(drop=True)


def _numeric_series_or_zero(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def _sum_components_or_existing(
    frame: pd.DataFrame,
    *,
    output_column: str,
    component_columns: tuple[str, ...],
) -> pd.Series:
    if any(column in frame.columns for column in component_columns):
        total = pd.Series(0.0, index=frame.index)
        for column in component_columns:
            total = total + _numeric_series_or_zero(frame, column)
        return total
    return _numeric_series_or_zero(frame, output_column)


def _persons_with_household_id(persons: pd.DataFrame) -> pd.DataFrame:
    if "household_id" in persons.columns:
        return persons
    if "PH_SEQ" not in persons.columns:
        return persons
    result = persons.copy()
    result["household_id"] = result["PH_SEQ"]
    return result


def _harmonize_tax_units(
    tax_units: pd.DataFrame,
    households: pd.DataFrame,
    persons: pd.DataFrame,
) -> pd.DataFrame:
    result = tax_units.copy()
    if "household_id" not in result.columns:
        result["household_id"] = result["tax_unit_id"]

    persons = _persons_with_household_id(persons)
    available_aggregates = {
        source: target
        for source, target in _CPS_TAX_UNIT_AGGREGATES.items()
        if source in persons.columns
    }
    if available_aggregates:
        if "household_id" not in persons.columns:
            raise ValueError(
                "CPS person aggregate columns require household_id or PH_SEQ"
            )
        aggregated = persons.groupby("household_id", as_index=False).agg(
            {source: "sum" for source in available_aggregates}
        )
        aggregated = aggregated.rename(columns=available_aggregates)
        result = result.drop(
            columns=list(available_aggregates.values()),
            errors="ignore",
        ).merge(aggregated, on="household_id", how="left")
        for column in available_aggregates.values():
            result[column] = result[column].fillna(0)

    for source, target in (
        ("SS_VAL", "gross_social_security"),
        ("PNSN_VAL", "taxable_pension_income"),
    ):
        if target in result.columns or source not in persons.columns:
            continue
        if "household_id" not in persons.columns:
            raise ValueError(f"CPS raw aggregate {source!r} requires household_id or PH_SEQ")
        aggregated = persons.groupby("household_id", as_index=False)[source].sum()
        aggregated = aggregated.rename(columns={source: target})
        result = result.merge(aggregated, on="household_id", how="left")
        result[target] = result[target].fillna(0)

    result["interest_dividend_income"] = _sum_components_or_existing(
        result,
        output_column="interest_dividend_income",
        component_columns=("taxable_interest_income", "ordinary_dividend_income"),
    )
    result["social_security"] = _sum_components_or_existing(
        result,
        output_column="social_security",
        component_columns=("gross_social_security",),
    )
    result["pension_income"] = _sum_components_or_existing(
        result,
        output_column="pension_income",
        component_columns=("taxable_pension_income",),
    )
    result["social_security_pension_income"] = _sum_components_or_existing(
        result,
        output_column="social_security_pension_income",
        component_columns=("gross_social_security", "taxable_pension_income"),
    )
    if "is_female" in result.columns:
        result["is_male"] = ~result["is_female"].fillna(False).astype(bool)
    else:
        result["is_male"] = False
    result["is_household_head"] = True

    household_columns = [
        column
        for column in (
            "household_id",
            "household_weight",
            *_CPS_TAX_UNIT_HOUSEHOLD_VARIABLES,
        )
        if column in households.columns
    ]
    household_values = households[household_columns].drop_duplicates("household_id")
    result = result.drop(
        columns=[
            "household_weight",
            *[
                column
                for column in _CPS_TAX_UNIT_HOUSEHOLD_VARIABLES
                if column in result.columns
            ],
        ],
        errors="ignore",
    ).merge(household_values, on="household_id", how="left")
    if result["household_weight"].isna().any():
        missing = result.loc[result["household_weight"].isna(), "household_id"].tolist()
        raise ValueError(
            f"CPS tax-unit table has household ids with no household_weight: {missing}"
        )
    result["weight"] = result["household_weight"]
    return result


@dataclass
class CPSAsecSourceProvider:
    """Provider-backed CPS ASEC source with household/person/tax-unit tables."""

    asec_year: int = 2025
    calendar_year: int = 2024
    cache_dir: Path | None = None
    download: bool = True
    loader: Callable[..., CPSDataset] | None = None
    source_name: str = "cps_asec"

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name=self.source_name,
            shareability=Shareability.PUBLIC,
            time_structure=TimeStructure.CROSS_SECTION,
            archetype=SourceArchetype.HOUSEHOLD_INCOME,
            population="US CPS ASEC households and persons",
            observations=(
                EntityObservation(
                    entity=EntityType.HOUSEHOLD,
                    key_column="household_id",
                    variable_names=tuple(
                        name
                        for name in dict.fromkeys(HOUSEHOLD_VARIABLES.values())
                        if name not in {"household_id", "household_weight"}
                    ),
                    weight_column="household_weight",
                    period_column="year",
                ),
                EntityObservation(
                    entity=EntityType.PERSON,
                    key_column="person_id",
                    variable_names=tuple(
                        name
                        for name in (
                            *dict.fromkeys(PERSON_VARIABLES.values()),
                            "is_adult",
                            "is_child",
                            "is_senior",
                        )
                        if name not in {"person_id", "weight"}
                    ),
                    weight_column="weight",
                    period_column="year",
                ),
                EntityObservation(
                    entity=EntityType.TAX_UNIT,
                    key_column="tax_unit_id",
                    variable_names=tuple(
                        dict.fromkeys(
                            (
                                *_CPS_TAX_UNIT_BASE_VARIABLES,
                                *_CPS_TAX_UNIT_HOUSEHOLD_VARIABLES,
                                *_CPS_TAX_UNIT_AGGREGATES.values(),
                            )
                        )
                    ),
                    weight_column="household_weight",
                    period_column="year",
                ),
            ),
        )

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        from microplex.data_sources.cps_transform import transform_cps_to_policyengine

        loader = self.loader or load_cps_asec
        cps = loader(
            year=self.asec_year,
            cache_dir=self.cache_dir,
            download=self.download,
        )
        households = _to_pandas(cps.households)
        persons = _to_pandas(cps.persons)
        transformed = transform_cps_to_policyengine(cps)
        tax_units = _to_pandas(transformed.tax_units)
        persons = _persons_with_household_id(persons)

        households = _ensure_column(
            households,
            "household_id",
            pd.RangeIndex(len(households), name="household_id"),
        )
        persons = _ensure_column(
            persons,
            "person_id",
            pd.RangeIndex(len(persons), name="person_id"),
        )
        tax_units = _ensure_column(
            tax_units,
            "tax_unit_id",
            pd.RangeIndex(len(tax_units), name="tax_unit_id"),
        )
        tax_units = _harmonize_tax_units(tax_units, households, persons)

        for table in (households, persons, tax_units):
            table["year"] = self.calendar_year
        households = _sort_by_key(households, "household_id")
        persons = _sort_by_key(persons, "person_id")
        tax_units = _sort_by_key(tax_units, "tax_unit_id")

        frame = ObservationFrame(
            source=self._descriptor_for(households, persons, tax_units),
            tables={
                EntityType.HOUSEHOLD: households,
                EntityType.PERSON: persons,
                EntityType.TAX_UNIT: tax_units,
            },
            relationships=(
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.PERSON,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_MANY,
                ),
                EntityRelationship(
                    parent_entity=EntityType.HOUSEHOLD,
                    child_entity=EntityType.TAX_UNIT,
                    parent_key="household_id",
                    child_key="household_id",
                    cardinality=RelationshipCardinality.ONE_TO_ONE,
                ),
            ),
        )
        frame.validate()
        return apply_source_query(frame, query)

    def _descriptor_for(
        self,
        households: pd.DataFrame,
        persons: pd.DataFrame,
        tax_units: pd.DataFrame,
    ) -> SourceDescriptor:
        return SourceDescriptor(
            name=self.source_name,
            shareability=Shareability.PUBLIC,
            time_structure=TimeStructure.CROSS_SECTION,
            archetype=SourceArchetype.HOUSEHOLD_INCOME,
            population="US CPS ASEC households and persons",
            observations=(
                EntityObservation(
                    entity=EntityType.HOUSEHOLD,
                    key_column="household_id",
                    variable_names=_variable_names(
                        households,
                        key_column="household_id",
                        weight_column="household_weight"
                        if "household_weight" in households.columns
                        else None,
                        period_column="year",
                    ),
                    weight_column="household_weight"
                    if "household_weight" in households.columns
                    else None,
                    period_column="year",
                ),
                EntityObservation(
                    entity=EntityType.PERSON,
                    key_column="person_id",
                    variable_names=_variable_names(
                        persons,
                        key_column="person_id",
                        weight_column="weight" if "weight" in persons.columns else None,
                        period_column="year",
                    ),
                    weight_column="weight" if "weight" in persons.columns else None,
                    period_column="year",
                ),
                EntityObservation(
                    entity=EntityType.TAX_UNIT,
                    key_column="tax_unit_id",
                    variable_names=_variable_names(
                        tax_units,
                        key_column="tax_unit_id",
                        weight_column="household_weight",
                        period_column="year",
                    ),
                    weight_column="household_weight",
                    period_column="year",
                ),
            ),
        )


def download_cps_asec(
    year: int,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Download CPS ASEC data for a given year.

    Args:
        year: Year of CPS ASEC (e.g., 2023)
        cache_dir: Directory to cache downloads
        force: Re-download even if cached

    Returns:
        Path to downloaded/cached zip file
    """
    import httpx

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    cache_dir.mkdir(parents=True, exist_ok=True)

    if year not in CPS_URLS:
        available = ", ".join(str(y) for y in sorted(CPS_URLS.keys()))
        raise ValueError(f"CPS ASEC for {year} not available. Available: {available}")

    url = CPS_URLS[year]
    filename = f"cps_asec_{year}.zip"
    cache_path = cache_dir / filename

    if cache_path.exists() and not force:
        print(f"Using cached CPS ASEC {year} from {cache_path}")
        return cache_path

    print(f"Downloading CPS ASEC {year} from {url}...")

    with httpx.Client(follow_redirects=True, timeout=300) as client:
        response = client.get(url)
        response.raise_for_status()

        with open(cache_path, "wb") as f:
            f.write(response.content)

    print(f"Downloaded {len(response.content) / 1_000_000:.1f} MB to {cache_path}")
    return cache_path


def load_cps_asec(
    year: int = 2023,
    cache_dir: Path | None = None,
    download: bool = True,
) -> CPSDataset:
    """
    Load CPS ASEC data for a given year.

    Args:
        year: Year of CPS ASEC (e.g., 2023)
        cache_dir: Directory for cached data
        download: Whether to download if not cached

    Returns:
        CPSDataset with persons and households DataFrames
    """
    import zipfile

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    # Check for processed parquet first
    processed_path = cache_dir / f"cps_asec_{year}_processed.parquet"
    if processed_path.exists():
        print(f"Loading processed CPS ASEC {year} from {processed_path}")
        persons = pl.read_parquet(processed_path)
        household_processed_path = (
            cache_dir / f"cps_asec_{year}_households_processed.parquet"
        )
        if household_processed_path.exists():
            households = pl.read_parquet(household_processed_path)
        else:
            raise FileNotFoundError(
                f"Processed CPS person cache exists at {processed_path}, but "
                f"household cache is missing at {household_processed_path}. "
                "Delete the stale person-only cache so a raw reload can "
                "preserve HSUP_WGT household weights."
            )
        return CPSDataset(
            persons=persons,
            households=households,
            year=year,
            source=str(processed_path),
        )

    # Download if needed
    zip_path = cache_dir / f"cps_asec_{year}.zip"
    if not zip_path.exists():
        if not download:
            raise FileNotFoundError(
                f"CPS ASEC {year} not found at {zip_path}. "
                "Set download=True to fetch from Census."
            )
        zip_path = download_cps_asec(year, cache_dir)

    # Extract and parse
    print(f"Parsing CPS ASEC {year}...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the person file (pppub*.csv)
        person_file = None
        household_file = None

        for name in zf.namelist():
            lower = name.lower()
            if "pppub" in lower and lower.endswith(".csv"):
                person_file = name
            elif "hhpub" in lower and lower.endswith(".csv"):
                household_file = name

        if person_file is None:
            raise ValueError(f"Could not find person file in {zip_path}")

        # Schema overrides for columns with large IDs that overflow int64
        schema_overrides = {
            "PERIDNUM": pl.Utf8,  # Person ID - too large for int64
            "H_IDNUM": pl.Utf8,  # Household ID - too large for int64
            "OCCURNUM": pl.Utf8,  # Occurrence number
            "QSTNUM": pl.Utf8,  # Questionnaire number
        }

        # Read person data
        with zf.open(person_file) as f:
            persons_raw = pl.read_csv(
                f,
                infer_schema_length=10000,
                schema_overrides=schema_overrides,
            )

        # Read household data if available
        if household_file:
            with zf.open(household_file) as f:
                households_raw = pl.read_csv(
                    f,
                    infer_schema_length=10000,
                    schema_overrides=schema_overrides,
                )
        else:
            households_raw = None

    # Process person data
    persons = _process_persons(persons_raw, year)

    # Process or derive household data
    if households_raw is not None:
        households = _process_households(households_raw, year)
    else:
        households = _derive_households(persons)

    # Cache processed data
    persons.write_parquet(processed_path)
    households.write_parquet(
        cache_dir / f"cps_asec_{year}_households_processed.parquet"
    )
    print(f"Cached processed data to {processed_path}")

    return CPSDataset(
        persons=persons,
        households=households,
        year=year,
        source=str(zip_path),
    )


def _process_persons(df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Process raw person file into clean format."""
    # Select and rename available columns
    available = set(df.columns)
    selected = {}

    for census_name, our_name in PERSON_VARIABLES.items():
        if census_name in available:
            selected[census_name] = our_name

    if not selected:
        raise ValueError("No recognized variables found in person file")

    result = df.select(
        [
            pl.col(census_name).alias(our_name)
            for census_name, our_name in selected.items()
        ]
    )

    # Scale weights: CPS ASEC weights have 2 implied decimal places
    # See CPS documentation: A_FNLWGT is expressed in units of 1/100
    # Divide by 100 to get actual population representation
    if "weight" in result.columns:
        result = result.with_columns((pl.col("weight") / 100).alias("weight"))
    if "march_supplement_weight" in result.columns:
        result = result.with_columns(
            (pl.col("march_supplement_weight") / 100).alias("march_supplement_weight")
        )

    # Convert income values (negative values indicate no income or missing)
    income_cols = [
        "wage_income",
        "self_employment_income",
        "interest_income",
        "dividend_income",
        "rental_income",
        "social_security",
        "ssi",
        "taxable_pension_income",
        "unemployment_compensation",
        "public_assistance",
        "total_person_income",
    ]

    for col in income_cols:
        if col in result.columns:
            result = result.with_columns(
                pl.when(pl.col(col) < 0).then(0).otherwise(pl.col(col)).alias(col)
            )

    # Add derived columns
    if "age" in result.columns:
        result = result.with_columns(
            [
                (pl.col("age") >= 18).alias("is_adult"),
                (pl.col("age") < 18).alias("is_child"),
                (pl.col("age") >= 65).alias("is_senior"),
            ]
        )

    # Add year
    result = result.with_columns(pl.lit(year).alias("year"))

    return result


def _process_households(df: pl.DataFrame, year: int) -> pl.DataFrame:
    """Process raw household file into clean format."""
    available = set(df.columns)
    selected = {}

    for census_name, our_name in HOUSEHOLD_VARIABLES.items():
        if census_name in available:
            selected[census_name] = our_name

    if not selected:
        raise ValueError("No recognized variables found in household file")

    result = df.select(
        [
            pl.col(census_name).alias(our_name)
            for census_name, our_name in selected.items()
        ]
    )

    # Scale weights: CPS ASEC weights have 2 implied decimal places
    if "household_weight" in result.columns:
        result = result.with_columns(
            (pl.col("household_weight") / 100).alias("household_weight")
        )

    result = result.with_columns(pl.lit(year).alias("year"))

    return result


def _derive_households(persons: pl.DataFrame) -> pl.DataFrame:
    """Derive household-level data from person records."""
    if "household_id" not in persons.columns:
        raise ValueError("Cannot derive households without household_id")

    households = persons.group_by("household_id").agg(
        [
            pl.len().alias("household_size"),
            pl.col("weight").first().alias("household_weight"),
            pl.col("state_fips").first()
            if "state_fips" in persons.columns
            else pl.lit(None).alias("state_fips"),
            pl.col("total_person_income").sum().alias("household_total_income")
            if "total_person_income" in persons.columns
            else pl.lit(0).alias("household_total_income"),
            pl.col("is_child").sum().alias("num_children")
            if "is_child" in persons.columns
            else pl.lit(0).alias("num_children"),
            pl.col("is_adult").sum().alias("num_adults")
            if "is_adult" in persons.columns
            else pl.lit(0).alias("num_adults"),
        ]
    )

    if "year" in persons.columns:
        year_val = persons.select("year").unique().to_series()[0]
        households = households.with_columns(pl.lit(year_val).alias("year"))

    return households


def get_available_years() -> list[int]:
    """Return list of available CPS ASEC years."""
    return sorted(CPS_URLS.keys())
