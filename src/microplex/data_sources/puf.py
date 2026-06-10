"""IRS Public Use File (PUF) loader and processing.

Downloads PUF from HuggingFace, uprates 2015 → target year,
and maps to common variable schema for multi-survey fusion.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microplex.core import EntityType
from microplex.core.sources import (
    EntityObservation,
    ObservationFrame,
    Shareability,
    SourceArchetype,
    SourceDescriptor,
    SourceQuery,
    TimeStructure,
    apply_source_query,
)

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# PUF variable code to common name mapping
# See IRS documentation for variable definitions
PUF_VARIABLE_MAP = {
    # Demographics
    "MARS": "filing_status_code",  # 1=single, 2=joint, 3=separate, 4=HoH, 5=widow
    "XTOT": "exemptions_count",
    "EIC": "eitc_children",  # Number of EIC qualifying children
    "n24": "ctc_children",  # Number of CTC qualifying children

    # Wage and salary income
    "E00200": "employment_income",  # Wages, salaries, tips

    # Self-employment
    "E00900": "self_employment_income",  # Business income/loss (Schedule C)
    "E02100": "farm_income",  # Farm income/loss (Schedule F)

    # Investment income
    "E00300": "taxable_interest_income",
    "E00400": "tax_exempt_interest_income",
    "E00600": "ordinary_dividend_income",
    "E00650": "qualified_dividend_income",
    "P22250": "short_term_capital_gains",  # Net ST gain/loss
    "P23250": "long_term_capital_gains",  # Net LT gain/loss
    "E01100": "capital_gains_distributions",

    # Pass-through income
    "E26270": "partnership_s_corp_income",  # Partnership/S-corp income
    "E25850": "rental_income_positive",  # Rental income (positive only)
    "E25860": "rental_income_negative",  # Rental loss (negative only)

    # Retirement income
    "E01400": "ira_distributions",
    "E01500": "total_pension_income",
    "E01700": "taxable_pension_income",
    "E02400": "gross_social_security",
    "E02500": "taxable_social_security",

    # Other income
    "E02300": "unemployment_compensation",
    "E00800": "alimony_received",

    # Itemized deduction inputs (expenses, not deductions)
    "E17500": "medical_expense_agi_floor",  # Medical after 7.5% AGI floor
    "E18400": "state_income_tax_paid",
    "E18500": "real_estate_tax_paid",
    "E19200": "mortgage_interest_paid",
    "E19800": "charitable_cash",
    "E20100": "charitable_noncash",

    # Other
    "E03150": "ira_deduction",
    "E03210": "student_loan_interest",

    # v2 parity fields (PE-named donors; rules mirror usdata puf.py:637-706)
    "E03500": "alimony_expense",
    "E20500": "casualty_loss",
    "E03240": "domestic_production_ald",
    "E03220": "educator_expense",
    "E26390": "_estate_income_gross",
    "E26400": "_estate_income_loss",
    "E03290": "health_savings_account_ald",
    "E24518": "long_term_capital_gains_on_collectibles",
    "E20400": "unreimbursed_business_employee_expenses",
    "E03230": "qualified_tuition_expenses",
    "E27200": "farm_rent_income",
    "E03300": "self_employed_pension_contribution_ald",
    "E24515": "unrecaptured_section_1250_gain",
    "E01200": "puf_miscellaneous_income",
    "E00700": "salt_refund_income",
    "E58990": "investment_income_elected_form_4952",

    # Weights
    "S006": "weight",  # In hundredths (divide by 100)
}

# SOI growth factors for uprating 2015 → 2024
# Based on IRS SOI aggregate growth rates
# These should be updated with actual SOI data
UPRATING_FACTORS = {
    # v2 parity fields — mirror nearest category factors above.
    "alimony_expense": 0.50,
    "casualty_loss": 1.35,
    "domestic_production_ald": 1.35,
    "educator_expense": 1.35,
    "estate_income": 1.50,
    "health_savings_account_ald": 1.50,
    "long_term_capital_gains_on_collectibles": 2.20,
    "unreimbursed_business_employee_expenses": 1.35,
    "qualified_tuition_expenses": 1.35,
    "farm_rent_income": 1.20,
    "self_employed_pension_contribution_ald": 1.35,
    "unrecaptured_section_1250_gain": 2.20,
    "puf_miscellaneous_income": 1.35,
    "salt_refund_income": 1.35,
    "investment_income_elected_form_4952": 1.80,
    "employment_income": 1.45,  # ~4.5% annual wage growth
    "self_employment_income": 1.35,
    "farm_income": 1.20,
    "taxable_interest_income": 2.50,  # Interest rates rose significantly
    "tax_exempt_interest_income": 1.80,
    "ordinary_dividend_income": 1.60,
    "qualified_dividend_income": 1.60,
    "short_term_capital_gains": 1.80,
    "long_term_capital_gains": 2.20,  # Stock market growth
    "capital_gains_distributions": 1.80,
    "partnership_s_corp_income": 1.50,
    "rental_income_positive": 1.40,
    "rental_income_negative": 1.40,
    "ira_distributions": 1.60,
    "total_pension_income": 1.40,
    "taxable_pension_income": 1.40,
    "gross_social_security": 1.45,
    "taxable_social_security": 1.45,
    "unemployment_compensation": 0.30,  # Down from COVID peak
    "alimony_received": 0.50,  # Declining due to tax law change
    "medical_expense_agi_floor": 1.50,
    "state_income_tax_paid": 1.40,
    "real_estate_tax_paid": 1.35,
    "mortgage_interest_paid": 1.30,
    "charitable_cash": 1.40,
    "charitable_noncash": 1.40,
    "student_loan_interest": 1.20,
}


def download_puf(cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Download PUF from HuggingFace.

    Returns paths to downloaded PUF and demographics CSV files.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "microplex"
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_puf_path = cache_dir / "puf_2015.csv"
    local_demo_path = cache_dir / "demographics_2015.csv"
    if local_puf_path.exists() and local_demo_path.exists():
        return local_puf_path, local_demo_path

    if not HF_AVAILABLE:
        raise FileNotFoundError(_restricted_puf_missing_message(cache_dir))

    try:
        # Download PUF 2015
        puf_path = hf_hub_download(
            repo_id="policyengine/irs-soi-puf",
            filename="puf_2015.csv",
            repo_type="model",
            local_dir=cache_dir,
        )

        # Download demographics file
        demo_path = hf_hub_download(
            repo_id="policyengine/irs-soi-puf",
            filename="demographics_2015.csv",
            repo_type="model",
            local_dir=cache_dir,
        )
    except Exception as exc:  # pragma: no cover - depends on external auth
        raise FileNotFoundError(_restricted_puf_missing_message(cache_dir)) from exc

    return Path(puf_path), Path(demo_path)


def _restricted_puf_missing_message(cache_dir: Path) -> str:
    return (
        "Could not locate restricted IRS PUF files. Place puf_2015.csv "
        f"and demographics_2015.csv in {cache_dir}, or pass explicit "
        "puf_path and demographics_path to load_puf/PUFSourceProvider."
    )


def load_puf_raw(
    puf_path: Path,
    demographics_path: Path | None = None,
) -> pd.DataFrame:
    """Load raw PUF data from CSV."""
    print(f"Loading PUF from {puf_path}...")
    puf = pd.read_csv(puf_path)

    # Filter out aggregate records (MARS=0)
    puf = puf[puf["MARS"] != 0].copy()

    print(f"  Raw records: {len(puf):,}")

    # Load and merge demographics if available
    if demographics_path and demographics_path.exists():
        print(f"Loading demographics from {demographics_path}...")
        demo = pd.read_csv(demographics_path)

        # Demographics file has RECID to match
        if "RECID" in puf.columns and "RECID" in demo.columns:
            puf = puf.merge(demo, on="RECID", how="left", suffixes=("", "_demo"))
            print(f"  After demographics merge: {len(puf):,}")

    return puf


def map_puf_variables(puf: pd.DataFrame) -> pd.DataFrame:
    """Map PUF variable codes to common names."""
    result = pd.DataFrame(index=puf.index)

    for puf_code, common_name in PUF_VARIABLE_MAP.items():
        if puf_code in puf.columns:
            result[common_name] = puf[puf_code].fillna(0)
        else:
            result[common_name] = 0

    # Fix weight (PUF stores in hundredths)
    if "weight" in result.columns:
        result["weight"] = result["weight"] / 100

    # Combine rental income (positive and negative)
    result["rental_income"] = (
        result.get("rental_income_positive", 0).fillna(0) +
        result.get("rental_income_negative", 0).fillna(0)
    )
    if "_estate_income_gross" in result.columns:
        result["estate_income"] = (
            result["_estate_income_gross"].fillna(0)
            - result["_estate_income_loss"].fillna(0)
        )
        result = result.drop(columns=["_estate_income_gross", "_estate_income_loss"])


    # Map filing status code to string
    filing_status_map = {
        1: "SINGLE",
        2: "JOINT",
        3: "SEPARATE",
        4: "HEAD_OF_HOUSEHOLD",
        5: "WIDOW",
    }
    result["filing_status"] = result["filing_status_code"].map(filing_status_map).fillna("UNKNOWN")

    # Add age from demographics if available
    if "age" in puf.columns:
        result["age"] = puf["age"]
    elif "AGE_HEAD" in puf.columns:
        result["age"] = puf["AGE_HEAD"]
    else:
        # Impute age based on income patterns
        result["age"] = _impute_age(result)

    # Add sex from demographics if available
    if "is_male" in puf.columns:
        result["is_male"] = puf["is_male"]
    elif "GENDER" in puf.columns:
        result["is_male"] = (puf["GENDER"] == 1).astype(float)
    else:
        # Unknown - will be learned from CPS
        result["is_male"] = np.nan

    # Mark survey source
    result["_survey"] = "puf"

    return result


def _impute_age(df: pd.DataFrame) -> pd.Series:
    """Simple age imputation based on income patterns.

    This is a rough heuristic. The masked MAF will learn
    better age distributions from CPS.
    """
    # Base age on Social Security receipt and pension income
    age = pd.Series(40, index=df.index)  # Default

    # Social Security recipients tend to be older
    has_ss = df.get("gross_social_security", 0) > 0
    age = age.where(~has_ss, 68)

    # Pension recipients also older
    has_pension = df.get("taxable_pension_income", 0) > 0
    age = age.where(~has_pension | has_ss, 62)

    # IRA distributions suggest retirement age
    has_ira = df.get("ira_distributions", 0) > 0
    age = age.where(~has_ira | has_ss | has_pension, 60)

    # High earners tend to be prime working age
    high_wage = df.get("employment_income", 0) > 200_000
    age = age.where(~high_wage, 45)

    # Add some noise
    noise = np.random.normal(0, 5, len(age))
    age = (age + noise).clip(18, 95).astype(int)

    return age


def uprate_puf(df: pd.DataFrame, from_year: int = 2015, to_year: int = 2024) -> pd.DataFrame:
    """Uprate PUF income variables from one year to another.

    Uses SOI-based growth factors.
    """
    if from_year == to_year:
        return df

    # Simple scaling - in production, use year-specific factors
    year_factor = (to_year - from_year) / (2024 - 2015)

    result = df.copy()

    for var, factor in UPRATING_FACTORS.items():
        if var in result.columns:
            # Interpolate factor based on years
            scaled_factor = 1 + (factor - 1) * year_factor
            result[var] = result[var] * scaled_factor

    print(f"Uprated PUF from {from_year} to {to_year}")

    return result


def expand_to_persons(df: pd.DataFrame) -> pd.DataFrame:
    """Expand tax unit records to person-level records.

    Each tax unit becomes 1-2 persons (filer + spouse if joint).
    This enables stacking with CPS person-level data.
    """
    records = []

    for idx, row in df.iterrows():
        filing_status = row.get("filing_status", "SINGLE")

        # Create head record
        head = row.copy()
        head["is_head"] = 1
        head["is_spouse"] = 0
        head["is_dependent"] = 0
        head["person_id"] = f"{idx}_head"
        head["tax_unit_id"] = idx
        records.append(head)

        # Create spouse record if joint filing
        if filing_status == "JOINT":
            spouse = row.copy()
            spouse["is_head"] = 0
            spouse["is_spouse"] = 1
            spouse["is_dependent"] = 0
            spouse["person_id"] = f"{idx}_spouse"
            spouse["tax_unit_id"] = idx

            # Split some income between head and spouse
            # In reality, we'd want to model this better
            for income_var in ["employment_income", "self_employment_income"]:
                if income_var in spouse.index and spouse[income_var] > 0:
                    # Assume 60/40 split (head gets more on average)
                    spouse[income_var] = row[income_var] * 0.4
                    head[income_var] = row[income_var] * 0.6

            # Spouse weight is same as head (we'll deduplicate in calibration)
            records.append(spouse)

    result = pd.DataFrame(records).reset_index(drop=True)
    print(f"Expanded {len(df):,} tax units to {len(result):,} persons")

    return result


def load_puf(
    target_year: int = 2024,
    expand_persons: bool = True,
    cache_dir: Path | None = None,
    puf_path: Path | None = None,
    demographics_path: Path | None = None,
) -> pd.DataFrame:
    """Load and process PUF for multi-survey fusion.

    Args:
        target_year: Year to uprate to
        expand_persons: If True, expand tax units to person records
        cache_dir: Directory to cache downloaded files
        puf_path: Optional explicit local PUF CSV path. The IRS PUF is
            restricted-access, so local handoff paths are first-class.
        demographics_path: Optional explicit local demographics CSV path.

    Returns:
        DataFrame with common variable names, ready for stacking with CPS
    """
    # Download if needed, or use explicitly supplied restricted-access files.
    if puf_path is None:
        puf_path, demo_path = download_puf(cache_dir)
    else:
        puf_path = Path(puf_path)
        if not puf_path.exists():
            raise FileNotFoundError(f"PUF file not found at {puf_path}")
        if demographics_path is None:
            candidate = puf_path.with_name("demographics_2015.csv")
            demo_path = candidate if candidate.exists() else None
        else:
            demo_path = Path(demographics_path)
    if demographics_path is not None:
        demo_path = Path(demographics_path)
        if not demo_path.exists():
            raise FileNotFoundError(f"PUF demographics file not found at {demo_path}")

    # Load raw data
    raw = load_puf_raw(puf_path, demo_path)

    # Map to common variables
    df = map_puf_variables(raw)

    # Uprate to target year
    df = uprate_puf(df, from_year=2015, to_year=target_year)

    # Expand to persons if requested
    if expand_persons:
        df = expand_to_persons(df)

    print(f"\nPUF loaded: {len(df):,} records")
    print(f"  Weight sum: {df['weight'].sum():,.0f}")

    return df


def _puf_variable_names(
    table: pd.DataFrame,
    *,
    key_column: str = "tax_unit_id",
    weight_column: str = "weight",
    period_column: str = "year",
) -> tuple[str, ...]:
    excluded = {key_column, weight_column, period_column}
    return tuple(column for column in table.columns if column not in excluded)


def _expected_puf_variable_names() -> tuple[str, ...]:
    expected = (
        *PUF_VARIABLE_MAP.values(),
        "rental_income",
        "filing_status",
        "age",
        "is_male",
        "_survey",
    )
    return tuple(
        name
        for name in dict.fromkeys(expected)
        if name not in {"tax_unit_id", "weight", "year"}
    )


@dataclass
class PUFSourceProvider:
    """Provider-backed IRS SOI PUF source with a tax-unit table."""

    target_year: int = 2024
    cache_dir: Path | None = None
    puf_path: Path | None = None
    demographics_path: Path | None = None
    loader: Callable[..., pd.DataFrame] = load_puf
    source_name: str = "puf"

    @property
    def descriptor(self) -> SourceDescriptor:
        return SourceDescriptor(
            name=self.source_name,
            shareability=Shareability.RESTRICTED,
            time_structure=TimeStructure.CROSS_SECTION,
            archetype=SourceArchetype.TAX_MICRODATA,
            population="US tax units",
            observations=(
                EntityObservation(
                    entity=EntityType.TAX_UNIT,
                    key_column="tax_unit_id",
                    variable_names=_expected_puf_variable_names(),
                    weight_column="weight",
                    period_column="year",
                ),
            ),
        )

    def load_frame(self, query: SourceQuery | None = None) -> ObservationFrame:
        loader_kwargs = {
            "target_year": self.target_year,
            "expand_persons": False,
            "cache_dir": self.cache_dir,
        }
        if self.puf_path is not None:
            loader_kwargs["puf_path"] = self.puf_path
        if self.demographics_path is not None:
            loader_kwargs["demographics_path"] = self.demographics_path
        table = self.loader(**loader_kwargs).copy()
        if "tax_unit_id" not in table.columns:
            table.insert(0, "tax_unit_id", pd.RangeIndex(len(table), name="tax_unit_id"))
        if "weight" not in table.columns:
            raise ValueError("PUF tax-unit table must contain weight")
        table["year"] = self.target_year

        frame = ObservationFrame(
            source=SourceDescriptor(
                name=self.source_name,
                shareability=Shareability.RESTRICTED,
                time_structure=TimeStructure.CROSS_SECTION,
                archetype=SourceArchetype.TAX_MICRODATA,
                population="US tax units",
                observations=(
                    EntityObservation(
                        entity=EntityType.TAX_UNIT,
                        key_column="tax_unit_id",
                        variable_names=_puf_variable_names(table),
                        weight_column="weight",
                        period_column="year",
                    ),
                ),
            ),
            tables={EntityType.TAX_UNIT: table},
        )
        frame.validate()
        return apply_source_query(frame, query)


# Variables that PUF has but CPS doesn't (will be NaN in CPS)
PUF_EXCLUSIVE_VARS = [
    "short_term_capital_gains",
    "long_term_capital_gains",
    "capital_gains_distributions",
    "partnership_s_corp_income",
    "qualified_dividend_income",
    "tax_exempt_interest_income",
    "charitable_cash",
    "charitable_noncash",
    "mortgage_interest_paid",
    "state_income_tax_paid",
    "real_estate_tax_paid",
    "student_loan_interest",
    "ira_deduction",
]

# Variables that both surveys have (may differ in quality)
SHARED_VARS = [
    "employment_income",
    "self_employment_income",
    "taxable_interest_income",
    "ordinary_dividend_income",
    "rental_income",
    "gross_social_security",
    "taxable_pension_income",
    "unemployment_compensation",
    "age",
    "filing_status",
]


if __name__ == "__main__":
    # Test loading
    df = load_puf(target_year=2024)
    print("\nSample of loaded PUF:")
    print(df.head())

    print("\nIncome variable sums:")
    income_vars = [
        "employment_income", "self_employment_income",
        "long_term_capital_gains", "partnership_s_corp_income",
        "gross_social_security", "taxable_pension_income",
    ]
    for var in income_vars:
        if var in df.columns:
            total = (df[var] * df["weight"]).sum() / 1e9
            print(f"  {var}: ${total:.1f}B")
