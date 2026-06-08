"""
Transform CPS data to policyengine-us variables.

Applies all CPS -> policyengine-us mappings and constructs tax units.
"""

from dataclasses import dataclass, field

import polars as pl

from microplex.data_sources.cps import CPSDataset
from microplex.data_sources.cps_mappings import (
    CoverageLevel,
    get_all_mappings,
)


@dataclass
class TransformedDataset:
    """CPS data transformed to policyengine-us variables."""

    persons: pl.DataFrame
    tax_units: pl.DataFrame
    households: pl.DataFrame
    year: int
    source: str = "CPS ASEC"

    coverage_report: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Summary statistics."""
        return {
            "n_persons": len(self.persons),
            "n_tax_units": len(self.tax_units),
            "n_households": len(self.households),
            "year": self.year,
            "total_weight": float(self.tax_units["weight"].sum()),
        }


def transform_cps_to_policyengine(cps: CPSDataset) -> TransformedDataset:
    """
    Transform CPS data to policyengine-us variables.

    Steps:
    1. Apply person-level mappings (age, earned_income, is_blind, etc.)
    2. Construct tax units from households
    3. Apply tax-unit-level mappings (filing_status, agi_proxy, etc.)
    4. Generate coverage report

    Args:
        cps: Raw CPS dataset

    Returns:
        TransformedDataset with persons, tax_units, and coverage metadata
    """
    # Step 1: Person-level transforms
    persons = _transform_persons(cps.persons)

    # Step 2: Construct tax units
    tax_units = _construct_tax_units(persons, cps.households)

    # Step 3: Generate coverage report
    coverage_report = _generate_coverage_report()

    return TransformedDataset(
        persons=persons,
        tax_units=tax_units,
        households=cps.households,
        year=cps.year,
        source=cps.source,
        coverage_report=coverage_report,
    )


def _transform_persons(persons: pl.DataFrame) -> pl.DataFrame:
    """Apply person-level variable mappings."""
    result = persons.clone()

    # Age
    if "A_AGE" in result.columns:
        result = result.with_columns(pl.col("A_AGE").alias("age"))

    # Earned income (wages + self-employment)
    wage_col = "WSAL_VAL" if "WSAL_VAL" in result.columns else "wage_income"
    semp_col = "SEMP_VAL" if "SEMP_VAL" in result.columns else "self_employment_income"

    if wage_col in result.columns or semp_col in result.columns:
        wage_expr = (
            pl.col(wage_col).fill_null(0) if wage_col in result.columns else pl.lit(0)
        )
        semp_expr = (
            pl.col(semp_col).fill_null(0) if semp_col in result.columns else pl.lit(0)
        )

        # Earned income for EITC is non-negative
        result = result.with_columns(
            (wage_expr + semp_expr).clip(lower_bound=0).alias("earned_income")
        )

    # Is blind
    if "PEDISEYE" in result.columns:
        result = result.with_columns((pl.col("PEDISEYE") == 1).alias("is_blind"))
    else:
        result = result.with_columns(pl.lit(False).alias("is_blind"))

    # Is dependent (child under 19 with child relationship)
    if "A_EXPRRP" in result.columns and "age" in result.columns:
        result = result.with_columns(
            (
                (pl.col("A_EXPRRP").is_in([4, 8]))  # Child or grandchild
                & (pl.col("age") < 19)
            ).alias("is_dependent")
        )
    else:
        result = result.with_columns(pl.lit(False).alias("is_dependent"))

    return result


def _own_child_relationship_expr(persons: pl.DataFrame) -> pl.Expr:
    if "A_EXPRRP" in persons.columns:
        return pl.col("A_EXPRRP").cast(pl.Int64, strict=False) == 4
    if "family_relationship" in persons.columns:
        return pl.col("family_relationship").cast(pl.Int64, strict=False) == 3
    if "is_child" in persons.columns:
        return pl.col("is_child").fill_null(False)
    return pl.col("age") < 18


def _construct_tax_units(
    persons: pl.DataFrame, households: pl.DataFrame
) -> pl.DataFrame:
    """
    Construct tax units from persons.

    Simplified tax unit construction:
    - Each household becomes one tax unit
    - Reference person is the filer
    - Filing status based on marital status
    - Aggregate income to tax unit level

    In reality, a household can have multiple tax units (e.g., adult children
    who file separately). This is a simplification for v1.
    """
    # Get household ID column
    hh_id_col = "PH_SEQ" if "PH_SEQ" in persons.columns else "household_id"

    # Identify reference persons (filers)
    # A_LINENO = 1 is typically the reference person
    if "A_LINENO" in persons.columns:
        lineno_col = "A_LINENO"
    elif "person_number" in persons.columns:
        lineno_col = "person_number"
    else:
        lineno_col = None

    # Aggregate earned income to household level
    earned_income_agg = persons.group_by(hh_id_col).agg(
        pl.col("earned_income").sum().alias("earned_income")
    )

    # Count CTC qualifying children per household
    if "age" in persons.columns:
        own_child_expr = _own_child_relationship_expr(persons)
        persons_with_ctc = persons.with_columns(
            (own_child_expr & (pl.col("age") < 17)).alias("_is_ctc_child")
        )

        ctc_counts = persons_with_ctc.group_by(hh_id_col).agg(
            pl.col("_is_ctc_child").sum().alias("ctc_qualifying_children")
        )
        persons_with_children = persons.with_columns(
            (own_child_expr & (pl.col("age") < 18)).alias("_own_child")
        )
        own_children_counts = persons_with_children.group_by(hh_id_col).agg(
            pl.col("_own_child").sum().alias("own_children_in_household")
        )
        under_18_counts = persons.group_by(hh_id_col).agg(
            (pl.col("age") < 18).sum().alias("count_under_18")
        )
    else:
        ctc_counts = persons.group_by(hh_id_col).agg(
            pl.lit(0).alias("ctc_qualifying_children")
        )
        own_children_counts = persons.group_by(hh_id_col).agg(
            pl.lit(0).alias("own_children_in_household")
        )
        under_18_counts = persons.group_by(hh_id_col).agg(
            pl.lit(0).alias("count_under_18")
        )

    # Get reference person attributes for each household
    if lineno_col:
        ref_persons = persons.filter(
            pl.col(lineno_col).cast(pl.Int64, strict=False) == 1
        )
    else:
        # Take first person per household as reference
        ref_persons = persons.group_by(hh_id_col).first()

    # Determine filing status from marital status
    # Check for both raw CPS column (A_MARITL) and processed column (marital_status)
    if "A_MARITL" in ref_persons.columns:
        ref_persons = ref_persons.with_columns(
            pl.when(pl.col("A_MARITL") == 1)  # Married, spouse present
            .then(pl.lit("married_joint"))
            .otherwise(pl.lit("single"))
            .alias("filing_status")
        )
    elif "marital_status" in ref_persons.columns:
        # Processed CPS uses marital_status with value 1 = married spouse present
        ref_persons = ref_persons.with_columns(
            pl.when(pl.col("marital_status") == 1)  # Married, spouse present
            .then(pl.lit("married_joint"))
            .otherwise(pl.lit("single"))
            .alias("filing_status")
        )
    else:
        ref_persons = ref_persons.with_columns(pl.lit("single").alias("filing_status"))
    ref_persons = ref_persons.with_columns(
        (pl.col("filing_status") == "married_joint").alias("is_married")
    )
    if "A_SEX" in ref_persons.columns:
        ref_persons = ref_persons.with_columns(
            (pl.col("A_SEX") == 2).alias("is_female")
        )
    elif "sex" in ref_persons.columns:
        ref_persons = ref_persons.with_columns((pl.col("sex") == 2).alias("is_female"))
    else:
        ref_persons = ref_persons.with_columns(pl.lit(False).alias("is_female"))

    if "PRDTRACE" in ref_persons.columns:
        ref_persons = ref_persons.with_columns(pl.col("PRDTRACE").alias("cps_race"))
    elif "race" in ref_persons.columns:
        ref_persons = ref_persons.with_columns(pl.col("race").alias("cps_race"))
    else:
        ref_persons = ref_persons.with_columns(pl.lit(0).alias("cps_race"))

    # Get weight
    weight_col = "A_FNLWGT" if "A_FNLWGT" in ref_persons.columns else "weight"
    if weight_col not in ref_persons.columns:
        weight_col = "weight"

    # Select columns for tax unit
    tu_cols = [hh_id_col, "filing_status"]
    for column in ("age", "is_female", "cps_race", "is_married"):
        if column in ref_persons.columns:
            tu_cols.append(column)
    if weight_col in ref_persons.columns:
        tu_cols.append(weight_col)

    tax_units = ref_persons.select([c for c in tu_cols if c in ref_persons.columns])

    # Rename weight column
    if weight_col in tax_units.columns and weight_col != "weight":
        tax_units = tax_units.rename({weight_col: "weight"})

    # Join aggregated values
    tax_units = tax_units.join(earned_income_agg, on=hh_id_col, how="left")
    tax_units = tax_units.join(ctc_counts, on=hh_id_col, how="left")
    tax_units = tax_units.join(own_children_counts, on=hh_id_col, how="left")
    tax_units = tax_units.join(under_18_counts, on=hh_id_col, how="left")

    # Compute AGI proxy at tax unit level
    # For now, just use earned income as base (will add investment income)
    tax_units = tax_units.with_columns(
        pl.col("earned_income").fill_null(0).alias("agi_proxy")
    )

    # Add investment income if available in persons
    int_col = "INT_VAL" if "INT_VAL" in persons.columns else "interest_income"
    div_col = "DIV_VAL" if "DIV_VAL" in persons.columns else "dividend_income"

    if int_col in persons.columns or div_col in persons.columns:
        invest_agg_exprs = []
        if int_col in persons.columns:
            invest_agg_exprs.append(pl.col(int_col).fill_null(0).sum().alias("_int"))
        if div_col in persons.columns:
            invest_agg_exprs.append(pl.col(div_col).fill_null(0).sum().alias("_div"))

        if invest_agg_exprs:
            invest_income = persons.group_by(hh_id_col).agg(invest_agg_exprs)

            tax_units = tax_units.join(invest_income, on=hh_id_col, how="left")

            # Add to AGI proxy
            int_expr = (
                pl.col("_int").fill_null(0)
                if "_int" in tax_units.columns
                else pl.lit(0)
            )
            div_expr = (
                pl.col("_div").fill_null(0)
                if "_div" in tax_units.columns
                else pl.lit(0)
            )

            tax_units = tax_units.with_columns(
                [
                    (pl.col("agi_proxy") + int_expr + div_expr).alias("agi_proxy"),
                    (int_expr + div_expr).alias("interest_dividend_income"),
                ]
            )

            # Drop temp columns
            tax_units = tax_units.drop(
                [c for c in ["_int", "_div"] if c in tax_units.columns]
            )

    ss_col = "SS_VAL" if "SS_VAL" in persons.columns else "social_security"
    pension_col = (
        "PNSN_VAL" if "PNSN_VAL" in persons.columns else "taxable_pension_income"
    )
    if ss_col in persons.columns or pension_col in persons.columns:
        ss_pension_agg_exprs = []
        if ss_col in persons.columns:
            ss_pension_agg_exprs.append(pl.col(ss_col).fill_null(0).sum().alias("_ss"))
        if pension_col in persons.columns:
            ss_pension_agg_exprs.append(
                pl.col(pension_col).fill_null(0).sum().alias("_pension")
            )
        ss_pension_income = persons.group_by(hh_id_col).agg(ss_pension_agg_exprs)
        tax_units = tax_units.join(ss_pension_income, on=hh_id_col, how="left")
        ss_expr = (
            pl.col("_ss").fill_null(0) if "_ss" in tax_units.columns else pl.lit(0)
        )
        pension_expr = (
            pl.col("_pension").fill_null(0)
            if "_pension" in tax_units.columns
            else pl.lit(0)
        )
        tax_units = tax_units.with_columns(
            (ss_expr + pension_expr).alias("social_security_pension_income")
        )
        tax_units = tax_units.drop(
            [c for c in ["_ss", "_pension"] if c in tax_units.columns]
        )

    # Fill nulls
    tax_units = tax_units.with_columns(
        [
            pl.col("earned_income").fill_null(0),
            pl.col("ctc_qualifying_children").fill_null(0),
            pl.col("own_children_in_household").fill_null(0),
            pl.col("count_under_18").fill_null(0),
            pl.col("agi_proxy").fill_null(0),
            (
                pl.col("interest_dividend_income").fill_null(0)
                if "interest_dividend_income" in tax_units.columns
                else pl.lit(0).alias("interest_dividend_income")
            ),
            (
                pl.col("social_security_pension_income").fill_null(0)
                if "social_security_pension_income" in tax_units.columns
                else pl.lit(0).alias("social_security_pension_income")
            ),
        ]
    )

    # Rename household ID to tax_unit_id
    tax_units = tax_units.rename({hh_id_col: "tax_unit_id"})

    return tax_units


def _generate_coverage_report() -> dict:
    """Generate report on variable coverage."""
    mappings = get_all_mappings()

    full = []
    partial = []
    derived = []
    none = []
    gaps = []

    for m in mappings:
        if m.coverage == CoverageLevel.FULL:
            full.append(m.policyengine_us_variable)
        elif m.coverage == CoverageLevel.PARTIAL:
            partial.append(m.policyengine_us_variable)
        elif m.coverage == CoverageLevel.DERIVED:
            derived.append(m.policyengine_us_variable)
        else:
            none.append(m.policyengine_us_variable)

        # Collect gaps
        for gap in m.gaps:
            gaps.append(
                {
                    "variable": m.policyengine_us_variable,
                    "component": gap.component,
                    "statute_ref": gap.statute_ref,
                    "impact": gap.impact,
                    "notes": gap.notes,
                }
            )

    return {
        "full": full,
        "partial": partial,
        "derived": derived,
        "none": none,
        "gaps": gaps,
        "summary": {
            "n_full": len(full),
            "n_partial": len(partial),
            "n_derived": len(derived),
            "n_none": len(none),
            "n_gaps": len(gaps),
        },
    }
