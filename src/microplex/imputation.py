"""Run the declarative imputation graph: fit a microimpute model per step and
write the synthesized columns onto the target half.

This is the heart of the spec-driven engine (see
``docs/spec-driven-rebuild.md`` §2 stage 4 and the eCPS reference
``policyengine_us_data/calibration/puf_impute.py``). microimpute already does
the model — regime-gated, QRF-based, sequentially-chained imputation. The
runner only orchestrates the *declared graph*:

- For each :class:`~microplex.spec.ImputationStep`, fit microimpute's canonical
  regime-aware ``Imputer`` on the donor frame over the step's variable list,
  conditioned on ``condition_on`` (default: the target half's demographic
  columns). Fits are unweighted unless the step declares ``weights``.
- microimpute chains internally: numeric ``imputed_variables`` are conditioned
  sequentially on the originals plus the previously-imputed targets, in list
  order. So the runner's only job for chaining is to *order* the variable list;
  :func:`spine_first_order` provides the generic ordering.
- Write the predicted columns onto the target half, respecting passthrough:
  columns the half already has are not overwritten unless the step sets
  ``synthesize: True``.

Country-agnostic: what "demographics" means and which variables are "income
spine" keywords are injected by the caller, never hard-coded.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from microplex.spec import (
    DEMOGRAPHICS_TOKEN,
    ImputationOrder,
    ImputationPhase,
    ImputationStep,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SPINE_FIRST_KEYWORDS",
    "spine_first_order",
    "ImputationStepResult",
    "ImputationRunner",
]

#: Default substring keywords that mark a variable as part of the income
#: "spine" — the receipt-type / total-income variables that dependent items
#: should chain on. Ordering puts any variable whose name contains one of these
#: (case-insensitive) before the rest. Generic and overridable; the spec/pack
#: can pass its own list. These are deliberately broad, country-neutral income
#: terms, not US-variable names.
SPINE_FIRST_KEYWORDS: tuple[str, ...] = (
    "employment_income",
    "wage",
    "salary",
    "earning",
    "self_employment",
    "total_income",
    "gross_income",
    "agi",
    "income",
    "pension",
    "social_security",
    "interest",
    "dividend",
    "capital_gain",
    "business",
    "rent",
)


def spine_first_order(
    variables: Sequence[str],
    *,
    keywords: Sequence[str] = SPINE_FIRST_KEYWORDS,
) -> list[str]:
    """Order variables so income/receipt-type ("spine") vars come first.

    A simple, documented two-tier heuristic:

    - **Tier 0 (spine):** any variable whose name contains one of ``keywords``
      (case-insensitive substring match). These are the income-bearing /
      receipt-type variables (wages, total income, etc.).
    - **Tier 1 (dependent):** everything else.

    The relative order *within* each tier is preserved from ``variables`` (a
    stable partition), so a pack can still influence sub-ordering by how it
    lists vars. Because microimpute chains numeric targets in list order, this
    makes dependent items condition on the income spine that precedes them.

    Args:
        variables: The variable names to order.
        keywords: Substrings that mark a variable as spine (tier 0). Defaults to
            :data:`SPINE_FIRST_KEYWORDS`. Pass a pack-specific list to override.

    Returns:
        A new list with spine variables first, then the rest; stable within
        each tier.
    """
    lowered_keywords = [k.lower() for k in keywords]

    def is_spine(name: str) -> bool:
        lname = name.lower()
        return any(kw in lname for kw in lowered_keywords)

    spine = [v for v in variables if is_spine(v)]
    rest = [v for v in variables if not is_spine(v)]
    return spine + rest


def order_variables(
    variables: Sequence[str],
    order: ImputationOrder,
    *,
    keywords: Sequence[str] = SPINE_FIRST_KEYWORDS,
) -> list[str]:
    """Apply an :class:`~microplex.spec.ImputationOrder` to a variable list."""
    if order is ImputationOrder.SPINE_FIRST:
        return spine_first_order(variables, keywords=keywords)
    return list(variables)


@dataclass
class ImputationStepResult:
    """Outcome of running one imputation step.

    Attributes:
        onto: The half name (or ``"both"``) the step targeted.
        donor: The donor source name the step drew from.
        imputed: Variables actually imputed (after dropping passthrough-skipped
            and donor-missing ones), in chain order.
        skipped_passthrough: Requested vars that already existed on the target
            and were preserved (step did not set ``synthesize``).
        skipped_missing_in_donor: Requested vars not present in the donor frame.
        predictors: The declared chain mapping (var -> chained predictor list),
            for inspection/testing. Empty if nothing numeric was imputed.
        regimes: The fitted sign-regime mapping from ``microimpute.Imputer``
            (``{var: regime}``). Empty when nothing numeric was imputed.
    """

    onto: str
    donor: str
    imputed: list[str] = field(default_factory=list)
    skipped_passthrough: list[str] = field(default_factory=list)
    skipped_missing_in_donor: list[str] = field(default_factory=list)
    predictors: dict[str, list[str]] = field(default_factory=dict)
    regimes: dict[str, str] = field(default_factory=dict)


class ImputationRunner:
    """Orchestrate canonical microimpute over a declared imputation graph.

    Args:
        column_groups: Mapping from group token (e.g. ``"demographics"``) to its
            concrete columns, used to resolve ``condition_on`` group tokens and
            the default condition set.
        demographic_columns: The default conditioning columns when a step has no
            ``condition_on`` (the half's demographics). May also be supplied via
            ``column_groups["demographics"]``; this argument is a convenience and
            takes precedence when set.
        weight_column: Retained for backwards-compatible runner construction.
            Donor fits are unweighted unless an imputation step declares
            ``weights``.
        spine_keywords: Keyword list for :func:`spine_first_order`.
        seed: Seed forwarded to ``microimpute.Imputer``.

    Notes:
        The runner is country-agnostic. It never special-cases a variable name;
        all semantics (demographics, spine keywords) are injected.
    """

    def __init__(
        self,
        *,
        column_groups: Mapping[str, Sequence[str]] | None = None,
        demographic_columns: Sequence[str] | None = None,
        weight_column: str | None = "household_weight",
        spine_keywords: Sequence[str] = SPINE_FIRST_KEYWORDS,
        seed: int = 0,
    ) -> None:
        self.column_groups = {
            token: list(cols) for token, cols in (column_groups or {}).items()
        }
        if demographic_columns is not None:
            self.column_groups[DEMOGRAPHICS_TOKEN] = list(demographic_columns)
        self.weight_column = weight_column
        self.spine_keywords = tuple(spine_keywords)
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def demographic_columns(self) -> list[str]:
        """The default conditioning columns (the ``demographics`` group)."""
        return list(self.column_groups.get(DEMOGRAPHICS_TOKEN, []))

    def resolve_condition_on(self, step: ImputationStep) -> list[str]:
        """Resolve a step's ``condition_on`` into concrete predictor columns.

        Defaults to the demographic columns when the step declares none.
        Expands group tokens (e.g. ``demographics``) and de-duplicates while
        preserving order. Literal entries pass through unchanged.
        """
        raw: Sequence[str]
        if step.condition_on is None:
            raw = [DEMOGRAPHICS_TOKEN]
        else:
            raw = step.condition_on

        resolved: list[str] = []
        for entry in raw:
            if entry in self.column_groups:
                resolved.extend(self.column_groups[entry])
            elif entry == DEMOGRAPHICS_TOKEN:
                raise ValueError(
                    f"step onto='{step.onto}' condition_on references the "
                    f"'{DEMOGRAPHICS_TOKEN}' group but no demographic columns "
                    "were configured on the runner."
                )
            else:
                resolved.append(entry)

        seen: set[str] = set()
        unique: list[str] = []
        for col in resolved:
            if col not in seen:
                seen.add(col)
                unique.append(col)
        if not unique:
            raise ValueError(
                f"step onto='{step.onto}' resolved to an empty condition_on set."
            )
        return unique

    def run_step(
        self,
        step: ImputationStep,
        *,
        donor: pd.DataFrame,
        target: pd.DataFrame,
    ) -> tuple[pd.DataFrame, ImputationStepResult]:
        """Run one imputation step against one target half.

        Args:
            step: The step to run.
            donor: The donor source frame (training data). Must contain the
                resolved predictors and at least some of the step's vars.
            target: The target half frame to write imputed columns onto. Must
                contain the resolved predictors.

        Returns:
            A ``(new_target, result)`` tuple. ``new_target`` is a copy of
            ``target`` with the imputed columns written; ``result`` records what
            happened (imputed / skipped / predictors).

        Raises:
            ValueError: if predictors are missing from donor or target.
        """
        predictors = self.resolve_condition_on(step)

        missing_donor_preds = [c for c in predictors if c not in donor.columns]
        if missing_donor_preds:
            raise ValueError(
                f"step onto='{step.onto}' from='{step.from_}': donor is missing "
                f"predictor columns {missing_donor_preds}."
            )
        missing_target_preds = [c for c in predictors if c not in target.columns]
        if missing_target_preds:
            raise ValueError(
                f"step onto='{step.onto}': target half is missing predictor "
                f"columns {missing_target_preds}."
            )

        # Decide which vars to impute: drop donor-missing and (unless
        # synthesize) those the target already has (passthrough).
        requested = order_variables(step.vars, step.order, keywords=self.spine_keywords)
        result = ImputationStepResult(onto=step.onto, donor=step.from_)

        to_impute: list[str] = []
        for var in requested:
            if var not in donor.columns:
                result.skipped_missing_in_donor.append(var)
                continue
            already_present = var in target.columns and target[var].notna().any()
            if already_present and not step.synthesize:
                result.skipped_passthrough.append(var)
                continue
            # Guard: a predictor cannot also be an imputed variable.
            if var in predictors:
                result.skipped_passthrough.append(var)
                continue
            to_impute.append(var)

        new_target = target.copy()
        if not to_impute:
            logger.info(
                "step onto=%s from=%s: nothing to impute (passthrough=%s, "
                "missing_in_donor=%s)",
                step.onto,
                step.from_,
                result.skipped_passthrough,
                result.skipped_missing_in_donor,
            )
            return new_target, result

        imputer = self._make_imputer()
        weight_col = self._resolve_step_weight_column(step, donor)

        # Include an explicitly declared weight column in the training frame so
        # microimpute can resolve it by name. Omitted step.weights means the fit
        # is intentionally unweighted, even if the donor has survey weights.
        train_cols = [*predictors, *to_impute]
        if weight_col is not None and weight_col not in train_cols:
            train_cols.append(weight_col)
        train = donor[train_cols].copy()
        fitted = imputer.fit(
            X_train=train,
            predictors=list(predictors),
            imputed_variables=list(to_impute),
            weight_col=weight_col,
        )
        predictions = fitted.predict(target[list(predictors)].copy())

        for var in to_impute:
            new_target[var] = predictions[var].to_numpy()

        result.imputed = list(to_impute)
        result.predictors = self._declared_predictors(predictors, to_impute)
        result.regimes = self._extract_regimes(fitted, to_impute)
        logger.info(
            "step onto=%s from=%s: imputed %d vars%s",
            step.onto,
            step.from_,
            len(to_impute),
            " (weighted)" if weight_col else "",
        )
        return new_target, result

    def _resolve_step_weight_column(
        self,
        step: ImputationStep,
        donor: pd.DataFrame,
    ) -> str | None:
        if step.weights is None:
            return None
        weight_col = step.weights
        if weight_col not in donor.columns:
            raise ValueError(
                f"step onto='{step.onto}' from='{step.from_}': donor is missing "
                f"weights column '{weight_col}'."
            )
        return weight_col

    def run(
        self,
        steps: Sequence[ImputationStep],
        *,
        halves: Mapping[str, pd.DataFrame],
        donors: Mapping[str, pd.DataFrame],
    ) -> tuple[dict[str, pd.DataFrame], list[ImputationStepResult]]:
        """Run the full imputation graph over the spine halves.

        Steps with ``onto == "both"`` are applied to every half. Steps mutate a
        working copy of each half in sequence, so later steps can condition on
        columns written by earlier ones (when those columns are named in
        ``condition_on``).

        Args:
            steps: The ordered imputation steps.
            halves: Mapping of half name to its frame (from the spine builder).
            donors: Mapping of donor source name to its frame.

        Returns:
            A ``(new_halves, results)`` tuple. ``new_halves`` maps each half
            name to its updated frame; ``results`` is the per-(step, half)
            outcome list in execution order.

        Raises:
            KeyError: if a step references a donor not in ``donors`` or a half
                not in ``halves``.
        """
        working = {name: frame.copy() for name, frame in halves.items()}
        results: list[ImputationStepResult] = []

        for step in steps:
            if step.at is not ImputationPhase.HALVES:
                raise ValueError(
                    "ImputationRunner.run only accepts at='halves' steps; "
                    "run at='base' steps before building the spine."
                )
            if step.from_ not in donors:
                raise KeyError(
                    f"imputation step references unknown donor '{step.from_}'."
                )
            targets = list(working.keys()) if step.targets_both else [step.onto]
            for half_name in targets:
                if half_name not in working:
                    raise KeyError(
                        f"imputation step references unknown half '{half_name}'."
                    )
                updated, result = self.run_step(
                    step,
                    donor=donors[step.from_],
                    target=working[half_name],
                )
                working[half_name] = updated
                # Record the concrete half name even for 'both' steps.
                result.onto = half_name
                results.append(result)

        return working, results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_imputer(self):
        """Construct a fresh canonical ``microimpute.Imputer`` for one step.

        ``microimpute.Imputer`` is the only imputer: regime-gated, QRF-based,
        and always chains numeric targets in declared order. There is no
        fallback and no alternate backend.
        """
        # Lazy import so microimpute is only required when actually running.
        from microimpute import Imputer

        return Imputer(seed=self.seed, log_level="WARNING")

    @staticmethod
    def _extract_regimes(fitted, imputed_variables: Sequence[str]) -> dict[str, str]:
        """Extract the fitted sign regimes from ``microimpute.Imputer``."""
        wanted = set(imputed_variables)
        return {
            var: str(regime)
            for var, regime in dict(fitted.regimes_).items()
            if var in wanted
        }

    @staticmethod
    def _declared_predictors(
        predictors: Sequence[str],
        imputed_variables: Sequence[str],
    ) -> dict[str, list[str]]:
        """Return the declared chain map for fitted targets.

        microimpute internally encodes categorical predictors, so fitted model
        feature names are implementation details. The release-facing contract is
        simpler: each later variable conditions on the step predictors plus all
        previously imputed variables in declared chain order.
        """
        base = list(predictors)
        chain: dict[str, list[str]] = {}
        prior: list[str] = []
        for var in imputed_variables:
            chain[var] = [*base, *prior]
            prior.append(var)
        return chain
