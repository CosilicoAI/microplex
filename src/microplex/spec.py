"""The Microplex spec DSL: a declarative document describing how a country
pack builds calibrated microdata.

This module is the single source of truth for "what a pack declares" (see
``docs/spec-driven-rebuild.md`` §1). A pack ships a YAML document; the engine
(:mod:`microplex.run`) consumes the validated :class:`MicroplexSpec` and runs
generic stages over it. There is no logic-Python in the pack: the spec names
sources, declares how the spine is split, what is imputed from what (and in
what order), which deterministic transforms run, and which targets calibrate
the result.

The schema is intentionally permissive about *values* (variable names, dataset
ids, block names) — those are pack/country specifics — but strict about
*structure* and *cross-references* so a malformed spec fails loudly with a
clear message rather than silently producing a wrong dataset.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

__all__ = [
    "SourceRole",
    "SpineMethod",
    "ImputationPhase",
    "ImputationOrder",
    "TransformKind",
    "CalibrationMethod",
    "VariableOperationKind",
    "SpecMeta",
    "SourceSpec",
    "CloneSpec",
    "HalfSpec",
    "SpineSpec",
    "ImputationStep",
    "SplitTransform",
    "DeriveTransform",
    "TransformSpec",
    "VariableCodeReference",
    "VariableOperationSpec",
    "VariableSystemProvenance",
    "VariableSpec",
    "ArchTargetSpec",
    "TargetsSpec",
    "CalibrateSpec",
    "MicroplexSpec",
    "SpecError",
    "load_spec",
    "load_spec_dict",
    "BASE_TOKEN",
    "BOTH_TOKEN",
    "DEMOGRAPHICS_TOKEN",
    "KEEP_ALL_TOKEN",
]


class SpecError(ValueError):
    """Raised when a spec document fails to load or validate.

    Wraps :class:`pydantic.ValidationError` (and YAML errors) so callers can
    catch a single, engine-owned exception type and get a human-readable
    message that names the offending field path.
    """


# ---------------------------------------------------------------------------
# Enumerations (the small set of declarative "knobs")
# ---------------------------------------------------------------------------


class SourceRole(StrEnum):
    """The role a source plays in the build.

    - ``spine``: the survey substrate that is split into halves. Exactly one
      source must be the spine.
    - ``donor``: a source other halves draw imputed variables from.
    """

    SPINE = "spine"
    DONOR = "donor"


class SpineMethod(StrEnum):
    """Spine construction strategy.

    - ``support_spine``: seeded 50/50 support partition. Every base row appears
      once, in either a passthrough half or a synthetic half.
    - ``clone``: legacy spelling retained for existing specs.
    """

    SUPPORT_SPINE = "support_spine"
    CLONE = "clone"


class ImputationOrder(StrEnum):
    """Chain ordering strategy for an imputation step.

    - ``spine_first``: a generic ordering that puts income-bearing /
      receipt-type variables (wages, total income, etc.) before the dependent
      items, so the chain conditions dependents on the income spine. See
      :func:`microplex.imputation.spine_first_order`.
    - ``as_declared``: keep the variable list exactly as written.
    """

    SPINE_FIRST = "spine_first"
    AS_DECLARED = "as_declared"


class ImputationPhase(StrEnum):
    """Where an imputation step runs in the spine pipeline.

    - ``base``: run before :class:`microplex.spine.SpineBuilder`; the output
      becomes the base frame that is split into halves.
    - ``halves``: run after the split on the declared half (or ``both`` halves).
    """

    BASE = "base"
    HALVES = "halves"


class TransformKind(StrEnum):
    """The kind of deterministic transform a rule declares."""

    SPLIT = "split"
    DERIVE = "derive"


class CalibrationMethod(StrEnum):
    """Reweighting method the calibrator should use."""

    APG = "apg"
    IPF = "ipf"
    L0 = "l0"


class VariableOperationKind(StrEnum):
    """Declarative operation that produces one variable in the spec runtime.

    This is the migration boundary between a country content package and the
    generic Microplex engine. Country packs may keep temporary provenance code
    references while porting, but their intended ``mp_spec`` behavior should
    converge on one of these operation kinds instead of executable country
    Python.
    """

    PASSTHROUGH = "passthrough"
    IMPUTE = "impute"
    DERIVE = "derive"
    SPLIT = "split"
    ASSIGN_GEO = "assign_geo"
    CLONE_GEO = "clone_geo"
    MATERIALIZE_POLICYENGINE = "materialize_policyengine"
    RERANDOMIZE_TAKEUP = "rerandomize_takeup"
    ENCODE_GEOID = "encode_geoid"
    STRUCTURAL_EXPORT = "structural_export"
    DEFAULT = "default"
    OPEN_DECISION = "open_decision"


# ---------------------------------------------------------------------------
# Spec sub-models
# ---------------------------------------------------------------------------

# A sentinel for "the half's demographic columns" / "keep everything". The DSL
# uses bare YAML strings like ``keep: all`` and ``strip_to: [demographics, ...]``.
# We model those as literal strings the engine resolves against the loaded
# frame, because the concrete column set is a pack/country specific (the engine
# is told what "demographics" means via the frame's columns at run time).
DEMOGRAPHICS_TOKEN = "demographics"
KEEP_ALL_TOKEN = "all"


class _StrictModel(BaseModel):
    """Base model that forbids unknown fields (typos fail loudly)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SpecMeta(_StrictModel):
    """Top-level identity for the spec document."""

    country: str = Field(
        ..., min_length=1, description="ISO-ish country code, e.g. 'us'."
    )
    model_year: int = Field(..., description="Calendar / model year the spec targets.")
    policyengine_model: str | None = Field(
        default=None,
        description="The PolicyEngine model package, e.g. 'policyengine-us'.",
    )

    @field_validator("country")
    @classmethod
    def _lower_country(cls, value: str) -> str:
        return value.strip().lower()


class SourceSpec(_StrictModel):
    """One named source: which dataset it resolves to and its role."""

    dataset: str = Field(
        ..., min_length=1, description="Dataset id the SourceRegistry resolves."
    )
    role: SourceRole = Field(
        ..., description="Whether this source is the spine or a donor."
    )


class CloneSpec(_StrictModel):
    """Options for seeded support-spine construction."""

    seed: int = Field(
        default=0,
        description="Deterministic seed for the passthrough/synthetic row split.",
    )


class HalfSpec(_StrictModel):
    """One half of the split spine.

    Exactly one of ``keep`` / ``strip_to`` must be set:

    - ``keep: all`` — passthrough: the half retains every base column (real
      survey values). Other tokens are reserved for future use.
    - ``strip_to: [demographics, tax_unit_id, ...]`` — the half keeps only the
      listed column groups/columns; everything else is dropped so it can be
      synthesized from scratch. ``demographics`` is a group token the engine
      resolves against the frame; any other entry is a literal column name.
    """

    name: str = Field(
        ..., min_length=1, description="Label for this half (becomes a column value)."
    )
    keep: str | None = Field(
        default=None,
        description="Passthrough token; only 'all' is currently supported.",
    )
    strip_to: list[str] | None = Field(
        default=None,
        description="Column groups/names to retain; the rest are stripped for synthesis.",
    )

    @model_validator(mode="after")
    def _exactly_one_mode(self) -> HalfSpec:
        if (self.keep is None) == (self.strip_to is None):
            raise ValueError(
                f"half '{self.name}': set exactly one of 'keep' or 'strip_to' "
                "(got " + ("neither" if self.keep is None else "both") + ")."
            )
        if self.keep is not None and self.keep != KEEP_ALL_TOKEN:
            raise ValueError(
                f"half '{self.name}': keep='{self.keep}' is not supported; "
                f"only keep='{KEEP_ALL_TOKEN}' is currently valid."
            )
        if self.strip_to is not None:
            if len(self.strip_to) == 0:
                raise ValueError(
                    f"half '{self.name}': strip_to must list at least one "
                    "column group or name."
                )
            if len(set(self.strip_to)) != len(self.strip_to):
                raise ValueError(f"half '{self.name}': strip_to has duplicate entries.")
        return self

    @property
    def is_passthrough(self) -> bool:
        """Whether this half keeps the base frame whole (no synthesis)."""
        return self.keep is not None


class SpineSpec(_StrictModel):
    """The spine: a base source split into exactly two halves.

    This is a deterministic 50/50 support partition: one half keeps real survey
    values, the other is stripped to demographics/ids and synthesized through
    the imputation graph.
    """

    base: str = Field(
        ...,
        min_length=1,
        description="Name of the source to split (its role must be 'spine').",
    )
    method: SpineMethod = Field(
        default=SpineMethod.SUPPORT_SPINE,
        description="Spine construction method.",
    )
    support: CloneSpec | None = Field(
        default=None,
        description="Options for the support_spine method.",
    )
    clone: CloneSpec = Field(
        default_factory=CloneSpec,
        description="Legacy options name for specs still using method: clone.",
    )
    halves: list[HalfSpec] = Field(..., min_length=2, max_length=2)

    @model_validator(mode="after")
    def _validate_halves(self) -> SpineSpec:
        if self.method not in {SpineMethod.SUPPORT_SPINE, SpineMethod.CLONE}:
            raise ValueError(
                f"spine.method '{self.method.value}' is not supported; "
                "use 'support_spine'."
            )
        explicit_clone = "clone" in self.model_fields_set
        if self.method is SpineMethod.SUPPORT_SPINE and explicit_clone:
            raise ValueError(
                "spine method: support_spine must use the 'support' options "
                "block, not legacy 'clone'."
            )
        if self.method is SpineMethod.CLONE and self.support is not None:
            raise ValueError(
                "spine method: clone must use the legacy 'clone' options block, "
                "not 'support'."
            )
        names = [half.name for half in self.halves]
        if len(set(names)) != len(names):
            raise ValueError(f"spine halves must have distinct names; got {names}.")
        passthrough = [half.name for half in self.halves if half.is_passthrough]
        if len(passthrough) != 1:
            raise ValueError(
                "spine must declare exactly one passthrough (keep: all) half and "
                f"one stripped (strip_to) half; passthrough halves: {passthrough}."
            )
        return self

    @property
    def partition_seed(self) -> int:
        """The deterministic support-partition seed."""
        return (self.support or self.clone).seed

    @property
    def half_names(self) -> tuple[str, ...]:
        """The two half names, in declared order."""
        return tuple(half.name for half in self.halves)

    @property
    def passthrough_half(self) -> HalfSpec:
        """The half that keeps real values."""
        return next(half for half in self.halves if half.is_passthrough)

    @property
    def synthetic_half(self) -> HalfSpec:
        """The half that is stripped and synthesized."""
        return next(half for half in self.halves if not half.is_passthrough)


# The special "onto" targets an imputation step may name in addition to a half.
BASE_TOKEN = "base"
BOTH_TOKEN = "both"


class ImputationStep(_StrictModel):
    """One step in the declarative imputation graph.

    Synthesize ``vars`` onto the ``onto`` target by fitting microimpute's
    canonical regime-aware donor backend on the ``from`` donor, conditioned on
    ``condition_on`` (default: demographics) plus the already-imputed chain.
    ``at: base`` steps run before the spine split and must target ``base`` or the
    declared spine source; ``at: halves`` steps run after the split and must
    target a half or ``both``. ``order`` controls the chain ordering.
    """

    onto: str = Field(..., min_length=1, description="Target half name, or 'both'.")
    from_: str = Field(
        ..., min_length=1, alias="from", description="Donor source name."
    )
    vars: list[str] = Field(
        ..., min_length=1, description="Variable names (or block names) to synthesize."
    )
    condition_on: list[str] | None = Field(
        default=None,
        description=(
            "Predictor columns/groups to condition on. Defaults to the half's "
            "demographic columns. May include the 'demographics' group token."
        ),
    )
    at: ImputationPhase = Field(
        default=ImputationPhase.HALVES,
        description="Pipeline phase: 'base' before the split or 'halves' after it.",
    )
    order: ImputationOrder = Field(
        default=ImputationOrder.SPINE_FIRST,
        description="Chain ordering strategy.",
    )
    weights: str | None = Field(
        default=None,
        description=(
            "Optional donor column to use as sample weights when fitting this "
            "step. Omitted or blank means an unweighted donor fit."
        ),
    )
    synthesize: bool = Field(
        default=False,
        description=(
            "If True, overwrite columns the target half already has. If False "
            "(default), existing columns pass through and are not re-imputed."
        ),
    )

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @field_validator("vars")
    @classmethod
    def _unique_vars(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("imputation step 'vars' contains duplicates.")
        return value

    @field_validator("weights", mode="before")
    @classmethod
    def _blank_weights_are_unweighted(cls, value: object) -> object:
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @property
    def targets_both(self) -> bool:
        """Whether this step applies to both halves."""
        return self.onto == BOTH_TOKEN


class SplitTransform(_StrictModel):
    """A transform that splits one source column into named pieces.

    ``into`` maps each output column name to a *fraction* (a float in [0, 1])
    or an *expression* string evaluated against the frame. When all values are
    fractions, the engine asserts they sum to ~1.0 and partitions the source
    deterministically.
    """

    source: str = Field(..., min_length=1, description="Source column to split.")
    into: dict[str, float | str] = Field(
        ..., min_length=1, description="Output column -> fraction or expression."
    )

    @model_validator(mode="after")
    def _validate_into(self) -> SplitTransform:
        if self.source in self.into:
            raise ValueError(
                f"split transform: output '{self.source}' collides with the "
                "source column name."
            )
        fractions = [v for v in self.into.values() if isinstance(v, (int, float))]
        if fractions and len(fractions) == len(self.into):
            total = float(sum(fractions))
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"split transform on '{self.source}': fractions must sum to "
                    f"1.0 (got {total:.6f})."
                )
            if any(f < 0 for f in fractions):
                raise ValueError(
                    f"split transform on '{self.source}': fractions must be "
                    "non-negative."
                )
        return self

    @property
    def is_fractional(self) -> bool:
        """Whether every output is a numeric fraction (vs. an expression)."""
        return all(isinstance(v, (int, float)) for v in self.into.values())


class DeriveTransform(_StrictModel):
    """A transform that derives a new column from an expression.

    ``expr`` is a pandas-eval expression over existing frame columns (e.g.
    ``"wages + self_employment_income"``). The engine evaluates it and writes
    the result to ``target``.
    """

    target: str = Field(..., min_length=1, description="Output column name.")
    expr: str = Field(
        ..., min_length=1, description="Expression over existing columns."
    )


class TransformSpec(_StrictModel):
    """A single transform rule: exactly one of ``split`` / ``derive``."""

    split: SplitTransform | None = None
    derive: DeriveTransform | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> TransformSpec:
        set_count = sum(x is not None for x in (self.split, self.derive))
        if set_count != 1:
            raise ValueError(
                "each transform must set exactly one of 'split' or 'derive' "
                f"(got {set_count})."
            )
        return self

    @property
    def kind(self) -> TransformKind:
        """Which kind of transform this rule is."""
        return TransformKind.SPLIT if self.split is not None else TransformKind.DERIVE


class VariableCodeReference(_StrictModel):
    """A source-code pointer that explains how one system builds a variable."""

    path: str = Field(
        ...,
        min_length=1,
        description="Repository-relative path, or package path, to the relevant code.",
    )
    lines: str | None = Field(
        default=None,
        min_length=1,
        description="Human-readable line reference, e.g. '36,246-248'.",
    )
    symbol: str | None = Field(
        default=None,
        min_length=1,
        description="Function, constant, or class name containing the behavior.",
    )
    summary: str | None = Field(
        default=None,
        min_length=1,
        description="Short explanation of the referenced code path.",
    )


class VariableOperationSpec(_StrictModel):
    """Declarative operation that the Microplex engine should eventually run.

    The schema is intentionally generic: ``kind`` is the stable operator name,
    while the optional fields carry the country-specific operands until the
    corresponding operator has a narrower typed model. This keeps content
    packages declarative without pretending every US migration rule is already
    executable.
    """

    kind: VariableOperationKind = Field(
        ...,
        description="Generic Microplex operation kind that produces the variable.",
    )
    source: str | None = Field(
        default=None,
        min_length=1,
        description="Source dataset/half/frame name, e.g. 'cps_asec' or 'puf'.",
    )
    source_column: str | None = Field(
        default=None,
        min_length=1,
        description="Input column when the operation copies or transforms one column.",
    )
    imputation_step: str | None = Field(
        default=None,
        min_length=1,
        description="Named imputation step or donor surface this variable belongs to.",
    )
    transform: str | None = Field(
        default=None,
        min_length=1,
        description="Named transform/operator detail, e.g. an eCPS-compatible split.",
    )
    expression: str | None = Field(
        default=None,
        min_length=1,
        description="Declarative expression for derive-style operations.",
    )
    encoding: str | None = Field(
        default=None,
        min_length=1,
        description="Export encoding detail, e.g. fixed-width S15 geoid bytes.",
    )
    handler: str | None = Field(
        default=None,
        min_length=1,
        description="Registered handler name for policy/takeup/materialization operators.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Variable dependencies the operation requires.",
    )
    temporary: bool = Field(
        default=True,
        description="Whether the operation is still a migration placeholder.",
    )
    notes: str | None = Field(
        default=None,
        min_length=1,
        description="Known caveats or remaining work for this operation declaration.",
    )

    @field_validator("depends_on")
    @classmethod
    def _unique_dependencies(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("variable operation depends_on contains duplicates.")
        return value


class VariableSystemProvenance(_StrictModel):
    """How one implementation produces a spec variable.

    ``method``/``code`` are intentionally descriptive audit metadata.
    ``operation`` is the executable spec-driven direction of travel for
    ``mp_spec`` rows: country packs should use it to name the generic Microplex
    operator that will eventually replace the referenced Python code.
    """

    method: str = Field(
        ..., min_length=1, description="Short method label, e.g. 'PUF QRF'."
    )
    operation: VariableOperationSpec | None = Field(
        default=None,
        description="Optional declarative operation that implements this behavior.",
    )
    code: list[VariableCodeReference] = Field(
        default_factory=list,
        description="Source-code references backing the method label.",
    )
    notes: str | None = Field(
        default=None,
        min_length=1,
        description="Known caveats or divergences for this system.",
    )


class VariableSpec(_StrictModel):
    """Temporary per-variable audit metadata.

    ``variables:`` is not used by the runtime engine yet. It exists to make
    country-pack specs self-auditing during migrations: every declared variable
    can carry the eCPS/incumbent code path, the legacy-MP code path, and the
    intended spec behavior side-by-side.
    """

    entity: str | None = Field(
        default=None,
        min_length=1,
        description="Optional PolicyEngine entity label for the variable.",
    )
    role: str | None = Field(
        default=None,
        min_length=1,
        description="Optional role/category, e.g. 'puf_imputed' or 'derived'.",
    )
    ecps: VariableSystemProvenance | None = Field(
        default=None,
        description="How the incumbent eCPS / production pipeline builds it.",
    )
    mp_legacy: VariableSystemProvenance | None = Field(
        default=None,
        description="How the legacy imperative Microplex pipeline builds it.",
    )
    mp_spec: VariableSystemProvenance | None = Field(
        default=None,
        description="How this declarative spec intends to build it.",
    )
    temporary: bool = Field(
        default=True,
        description="Marks this as a migration audit scaffold, not runtime logic.",
    )

    @model_validator(mode="after")
    def _has_some_provenance(self) -> VariableSpec:
        if self.ecps is None and self.mp_legacy is None and self.mp_spec is None:
            raise ValueError(
                "variable spec must declare at least one of ecps, mp_legacy, or mp_spec."
            )
        return self


class ArchTargetSpec(_StrictModel):
    """Names the Arch target set to fetch and roll up."""

    country: str = Field(..., min_length=1)
    model_year: int
    target_profile: str | None = Field(
        default=None,
        min_length=1,
        description="Target surface/profile used for scoring and metadata.",
    )
    calibration_target_profile: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional narrower/source-backed profile used by calibration. "
            "Defaults to target_profile when omitted."
        ),
    )

    @field_validator("country")
    @classmethod
    def _lower_country(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def resolved_calibration_target_profile(self) -> str | None:
        """Calibration profile after applying the default-to-target rule."""
        return self.calibration_target_profile or self.target_profile


class TargetsSpec(_StrictModel):
    """Calibration target sources. Currently only the Arch target set."""

    arch: ArchTargetSpec


class CalibrateSpec(_StrictModel):
    """How to reweight the built frame to the targets."""

    loss: str = Field(
        ..., min_length=1, description="Named loss, e.g. 'pe_native_bucketed_huber_v1'."
    )
    method: CalibrationMethod = Field(..., description="Reweighting method.")
    target_records: int | None = Field(
        default=None,
        gt=0,
        description="Optional L0 prune target record count.",
    )


# ---------------------------------------------------------------------------
# Top-level spec
# ---------------------------------------------------------------------------


class MicroplexSpec(_StrictModel):
    """A complete Microplex build specification.

    Validates cross-references between sections: the spine base must be a
    declared source with role ``spine``; imputation steps must reference
    declared halves (or ``both``) and donor sources; etc.
    """

    meta: SpecMeta
    sources: dict[str, SourceSpec] = Field(..., min_length=1)
    spine: SpineSpec
    imputation: list[ImputationStep] = Field(default_factory=list)
    transforms: list[TransformSpec] = Field(default_factory=list)
    variables: dict[str, VariableSpec] = Field(default_factory=dict)
    targets: TargetsSpec | None = None
    calibrate: CalibrateSpec | None = None

    @model_validator(mode="after")
    def _validate_cross_references(self) -> MicroplexSpec:
        # Exactly one spine source.
        spine_sources = [
            name for name, src in self.sources.items() if src.role is SourceRole.SPINE
        ]
        if len(spine_sources) != 1:
            raise ValueError(
                "exactly one source must have role 'spine'; spine sources: "
                f"{spine_sources}."
            )

        # spine.base must be that spine source.
        if self.spine.base not in self.sources:
            raise ValueError(
                f"spine.base '{self.spine.base}' is not a declared source."
            )
        if self.sources[self.spine.base].role is not SourceRole.SPINE:
            raise ValueError(
                f"spine.base '{self.spine.base}' must have role 'spine' (has "
                f"'{self.sources[self.spine.base].role.value}')."
            )
        if spine_sources[0] != self.spine.base:
            raise ValueError(
                f"the spine source '{spine_sources[0]}' does not match "
                f"spine.base '{self.spine.base}'."
            )

        valid_half_onto = set(self.spine.half_names) | {BOTH_TOKEN}
        valid_base_onto = {BASE_TOKEN, self.spine.base}
        for index, step in enumerate(self.imputation):
            if step.at is ImputationPhase.BASE:
                if step.onto not in valid_base_onto:
                    raise ValueError(
                        f"imputation[{index}] at 'base' must target 'base' or "
                        f"the spine source '{self.spine.base}'; got "
                        f"onto='{step.onto}'."
                    )
            elif step.onto not in valid_half_onto:
                raise ValueError(
                    f"imputation[{index}].onto '{step.onto}' is not a declared "
                    f"half or 'both'; valid: {sorted(valid_half_onto)}."
                )
            if step.from_ not in self.sources:
                raise ValueError(
                    f"imputation[{index}].from '{step.from_}' is not a declared source."
                )

        return self

    @property
    def spine_source(self) -> str:
        """The name of the single spine source."""
        return self.spine.base

    @property
    def donor_sources(self) -> tuple[str, ...]:
        """Names of all donor-role sources, in declaration order."""
        return tuple(
            name for name, src in self.sources.items() if src.role is SourceRole.DONOR
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _format_validation_error(error: ValidationError) -> str:
    """Turn a pydantic ValidationError into a compact, field-pathed message."""
    lines = []
    for err in error.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "spec validation failed:\n" + "\n".join(lines)


def load_spec_dict(data: Mapping[str, Any]) -> MicroplexSpec:
    """Validate an already-parsed mapping into a :class:`MicroplexSpec`.

    Raises:
        SpecError: if validation fails, with a field-pathed message.
    """
    try:
        return MicroplexSpec.model_validate(dict(data))
    except ValidationError as exc:
        raise SpecError(_format_validation_error(exc)) from exc


def load_spec(path: str | Path) -> MicroplexSpec:
    """Load and validate a Microplex spec from a YAML file.

    Args:
        path: Path to the YAML spec document.

    Returns:
        A validated :class:`MicroplexSpec`.

    Raises:
        SpecError: if the file is missing, not valid YAML, not a mapping, or
            fails schema validation.
    """
    spec_path = Path(path)
    if not spec_path.exists():
        raise SpecError(f"spec file not found: {spec_path}")
    try:
        raw = yaml.safe_load(spec_path.read_text())
    except yaml.YAMLError as exc:
        raise SpecError(f"spec file is not valid YAML ({spec_path}): {exc}") from exc
    if not isinstance(raw, Mapping):
        raise SpecError(
            f"spec file must contain a YAML mapping at the top level "
            f"({spec_path}); got {type(raw).__name__}."
        )
    return load_spec_dict(raw)
