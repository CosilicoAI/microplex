"""Optional PolicyEngine-US runtime materialization helpers.

The US content pack stays declarative. This module provides opt-in Python
adapters that callers can register with :func:`microplex.run.run_spec` and the
simulator-aware target compiler when a build environment has a PolicyEngine-US
runtime available.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from microplex.core import EntityType
from microplex.runtime_operations import RuntimeVariableOperationHandler
from microplex.spec import MicroplexSpec, VariableOperationKind, VariableSpec
from microplex.targets import (
    MaterializedSimulationTargetCompiler,
    TargetSimulationModifier,
    TargetSpec,
)

POLICYENGINE_US_RUNTIME_HANDLER = "policyengine_us"
POLICYENGINE_US_TAKEUP_HANDLER = "policyengine_us_takeup"

TAKEUP_VARIABLE_ENTITY: dict[str, EntityType] = {
    "takes_up_aca_if_eligible": EntityType.TAX_UNIT,
    "takes_up_dc_ptc": EntityType.TAX_UNIT,
    "takes_up_eitc": EntityType.TAX_UNIT,
    "would_file_taxes_voluntarily": EntityType.TAX_UNIT,
    "takes_up_early_head_start_if_eligible": EntityType.PERSON,
    "takes_up_head_start_if_eligible": EntityType.PERSON,
    "takes_up_medicaid_if_eligible": EntityType.PERSON,
    "takes_up_medicare_if_eligible": EntityType.PERSON,
    "takes_up_ssi_if_eligible": EntityType.PERSON,
    "would_claim_wic": EntityType.PERSON,
    "takes_up_housing_assistance_if_eligible": EntityType.SPM_UNIT,
    "takes_up_snap_if_eligible": EntityType.SPM_UNIT,
    "takes_up_tanf_if_eligible": EntityType.SPM_UNIT,
}

TAKEUP_VARIABLE_PROGRAM: dict[str, str] = {
    "takes_up_aca_if_eligible": "aca",
    "takes_up_dc_ptc": "dc_ptc",
    "takes_up_eitc": "eitc",
    "would_file_taxes_voluntarily": "voluntary_filing",
    "takes_up_early_head_start_if_eligible": "early_head_start",
    "takes_up_head_start_if_eligible": "head_start",
    "takes_up_medicaid_if_eligible": "medicaid",
    "takes_up_medicare_if_eligible": "medicare",
    "takes_up_ssi_if_eligible": "ssi",
    "would_claim_wic": "wic",
    "takes_up_housing_assistance_if_eligible": "housing_assistance",
    "takes_up_snap_if_eligible": "snap",
    "takes_up_tanf_if_eligible": "tanf",
}

TAKEUP_VARIABLE_RATE_PARAMETER: dict[str, str] = {
    **TAKEUP_VARIABLE_PROGRAM,
    "would_claim_wic": "wic_takeup",
}

TAKEUP_VARIABLES_BY_PROGRAM: dict[str, tuple[str, ...]] = {}
for _variable, _program in TAKEUP_VARIABLE_PROGRAM.items():
    TAKEUP_VARIABLES_BY_PROGRAM.setdefault(_program, tuple())
    TAKEUP_VARIABLES_BY_PROGRAM[_program] = (
        *TAKEUP_VARIABLES_BY_PROGRAM[_program],
        _variable,
    )


@runtime_checkable
class PolicyEngineUSVariableMaterializer(Protocol):
    """Materialize PolicyEngine-US variables onto entity frames."""

    def materialize_policyengine_variables(
        self,
        entity_frames: Mapping[EntityType, pd.DataFrame],
        *,
        variables_by_entity: Mapping[EntityType, Sequence[str]],
        period: int | str,
    ) -> Mapping[EntityType, pd.DataFrame]:
        """Return entity frames with the requested PE-US variables present."""


@runtime_checkable
class PolicyEngineUSTakeupRerandomizer(Protocol):
    """Regenerate PolicyEngine-US stochastic take-up input columns."""

    def rerandomize_takeup(
        self,
        entity_frames: Mapping[EntityType, pd.DataFrame],
        *,
        variables_by_entity: Mapping[EntityType, Sequence[str]],
        period: int | str,
        modifiers: Sequence[TargetSimulationModifier],
    ) -> Mapping[EntityType, pd.DataFrame]:
        """Return entity frames with the requested take-up columns present."""


@runtime_checkable
class PolicyEngineUSTakeupRateSource(Protocol):
    """Resolve per-row PolicyEngine-US take-up rates."""

    def rate_for_takeup_variable(
        self,
        variable: str,
        frame: pd.DataFrame,
        *,
        period: int | str,
    ) -> float | np.ndarray:
        """Return a scalar rate or row-aligned rates for ``variable``."""


@dataclass
class PolicyEngineUSRuntimeAdapter(RuntimeVariableOperationHandler):
    """Bridge Microplex runtime hooks to PolicyEngine-US materializers.

    The adapter is deliberately dependency-injected. Base Microplex can test and
    ship the orchestration without depending on ``policyengine-us``; production
    builds pass a concrete materializer/rerandomizer when they have those
    optional packages available.
    """

    materializer: PolicyEngineUSVariableMaterializer | None = None
    takeup_rerandomizer: PolicyEngineUSTakeupRerandomizer | None = None
    period: int | str | None = None

    def runtime_variable_operation_handlers(
        self,
    ) -> dict[str, RuntimeVariableOperationHandler]:
        """Return handlers keyed by the names used in the US spec."""
        return {
            POLICYENGINE_US_RUNTIME_HANDLER: self,
            POLICYENGINE_US_TAKEUP_HANDLER: self,
        }

    def simulation_compiler(self) -> MaterializedSimulationTargetCompiler:
        """Return a grouped target compiler backed by this adapter."""
        return MaterializedSimulationTargetCompiler(self)

    def apply_variable_operations(
        self,
        frame: pd.DataFrame,
        *,
        variables: Mapping[str, VariableSpec],
        spec: MicroplexSpec,
        operation_kind: VariableOperationKind,
    ) -> pd.DataFrame:
        """Materialize spec-declared runtime variables onto a flat run frame."""
        period = self._period_for_spec(spec)
        variables_by_entity = _variables_by_spec_entity(variables)
        if operation_kind is VariableOperationKind.RERANDOMIZE_TAKEUP:
            frames = self._rerandomize_takeup(
                _single_frame_entities(frame, variables_by_entity),
                variables_by_entity=variables_by_entity,
                period=period,
                modifiers=(),
            )
        elif operation_kind is VariableOperationKind.MATERIALIZE_POLICYENGINE:
            frames = self._materialize_policyengine(
                _single_frame_entities(frame, variables_by_entity),
                variables_by_entity=variables_by_entity,
                period=period,
            )
        else:
            raise ValueError(
                f"PolicyEngine-US adapter does not handle {operation_kind.value!r}."
            )
        return _merge_runtime_columns(frame, frames, variables_by_entity)

    def materialize_simulation_features(
        self,
        *,
        targets: Sequence[TargetSpec],
        entity_frames: Mapping[EntityType, pd.DataFrame],
        modifiers: Sequence[TargetSimulationModifier],
    ) -> Mapping[EntityType, pd.DataFrame]:
        """Materialize target features that depend on PolicyEngine-US."""
        if not targets:
            return {}

        period = self.period or targets[0].period
        frames = _copy_entity_frames(entity_frames)
        for modifier in modifiers:
            if modifier.name == "rerandomize_takeup":
                variables_by_entity = _takeup_variables_from_modifier(
                    modifier,
                    targets,
                )
                frames = self._rerandomize_takeup(
                    frames,
                    variables_by_entity=variables_by_entity,
                    period=period,
                    modifiers=(modifier,),
                )
            elif modifier.name == "materialize_policyengine":
                _assert_policyengine_us_modifier(modifier)
                variables_by_entity = _missing_required_features(targets, frames)
                frames = self._materialize_policyengine(
                    frames,
                    variables_by_entity=variables_by_entity,
                    period=period,
                )
            else:
                raise ValueError(
                    f"Unsupported PolicyEngine-US simulation modifier: {modifier.name!r}"
                )
        return frames

    def _period_for_spec(self, spec: MicroplexSpec) -> int | str:
        return self.period or spec.meta.model_year

    def _materialize_policyengine(
        self,
        entity_frames: Mapping[EntityType, pd.DataFrame],
        *,
        variables_by_entity: Mapping[EntityType, Sequence[str]],
        period: int | str,
    ) -> dict[EntityType, pd.DataFrame]:
        variables_by_entity = _nonempty_variables(variables_by_entity)
        if not variables_by_entity:
            return _copy_entity_frames(entity_frames)
        if self.materializer is None:
            raise ValueError(
                "PolicyEngine-US materialization requires a "
                "PolicyEngineUSVariableMaterializer."
            )
        updated = self.materializer.materialize_policyengine_variables(
            _copy_entity_frames(entity_frames),
            variables_by_entity=variables_by_entity,
            period=period,
        )
        return _validated_entity_frame_update(entity_frames, updated)

    def _rerandomize_takeup(
        self,
        entity_frames: Mapping[EntityType, pd.DataFrame],
        *,
        variables_by_entity: Mapping[EntityType, Sequence[str]],
        period: int | str,
        modifiers: Sequence[TargetSimulationModifier],
    ) -> dict[EntityType, pd.DataFrame]:
        variables_by_entity = _nonempty_variables(variables_by_entity)
        if not variables_by_entity:
            return _copy_entity_frames(entity_frames)
        if self.takeup_rerandomizer is None:
            raise ValueError(
                "PolicyEngine-US take-up rerandomization requires a "
                "PolicyEngineUSTakeupRerandomizer."
            )
        updated = self.takeup_rerandomizer.rerandomize_takeup(
            _copy_entity_frames(entity_frames),
            variables_by_entity=variables_by_entity,
            period=period,
            modifiers=tuple(modifiers),
        )
        return _validated_entity_frame_update(entity_frames, updated)


@dataclass(frozen=True)
class SeededPolicyEngineUSTakeupRerandomizer:
    """Simple deterministic take-up rerandomizer with caller-supplied rates."""

    rates: Mapping[str, float] | PolicyEngineUSTakeupRateSource
    seed: int = 0

    def rerandomize_takeup(
        self,
        entity_frames: Mapping[EntityType, pd.DataFrame],
        *,
        variables_by_entity: Mapping[EntityType, Sequence[str]],
        period: int | str,
        modifiers: Sequence[TargetSimulationModifier],
    ) -> Mapping[EntityType, pd.DataFrame]:
        del modifiers
        frames = _copy_entity_frames(entity_frames)
        for entity, variables in variables_by_entity.items():
            if entity not in frames:
                raise ValueError(
                    "Cannot rerandomize PolicyEngine-US take-up variable(s) "
                    f"{tuple(variables)} without an {entity.value!r} frame."
                )
            frame = frames[entity].copy()
            for variable in variables:
                rate = _rate_for_takeup_variable(
                    variable,
                    self.rates,
                    frame=frame,
                    period=period,
                )
                rng = _stable_rng(self.seed, variable)
                frame[variable] = rng.random(len(frame)) < rate
            frames[entity] = frame
        return frames


@dataclass(frozen=True)
class PolicyEngineUSDataTakeupRateSource:
    """Rate source matching legacy ``policyengine-us-data`` take-up inputs."""

    loader: Any | None = None

    def rate_for_takeup_variable(
        self,
        variable: str,
        frame: pd.DataFrame,
        *,
        period: int | str,
    ) -> float | np.ndarray:
        if variable == "takes_up_eitc":
            rates = self._load("eitc", period)
            return _rates_by_numeric_child_count(
                variable,
                frame,
                rates,
                column="eitc_child_count",
                default=0.85,
                max_key=3,
            )
        if variable == "takes_up_medicaid_if_eligible":
            rates = self._load("medicaid", period)
            return _rates_by_string_column(
                variable,
                frame,
                rates,
                column="state_code_str",
                default=0.93,
            )
        if variable == "would_claim_wic":
            rates = self._load("wic_takeup", period)
            return _rates_by_string_column(
                variable,
                frame,
                rates,
                column="wic_category_str",
                default=0.0,
            )
        if variable == "would_file_taxes_voluntarily":
            rate = _scalar_rate(
                variable,
                self._load("voluntary_filing", period),
            )
            if "takes_up_eitc" not in frame:
                raise ValueError(
                    "Legacy voluntary filing take-up requires "
                    "'takes_up_eitc' on the tax_unit frame."
                )
            takes_up_eitc = frame["takes_up_eitc"].astype(bool).to_numpy()
            return np.where(takes_up_eitc, 0.0, rate)

        parameter = TAKEUP_VARIABLE_RATE_PARAMETER.get(variable)
        if parameter is None:
            raise ValueError(f"Unknown PolicyEngine-US take-up variable {variable!r}.")
        unsupported = {
            "housing_assistance",
            "medicare",
            "wic",
        }
        if parameter in unsupported:
            raise ValueError(
                "No legacy policyengine-us-data take-up rate parameter is "
                f"declared for {variable!r}."
            )
        return _scalar_rate(variable, self._load(parameter, period))

    def _load(self, parameter: str, period: int | str) -> Any:
        year = _period_year(period)
        if self.loader is not None:
            return self.loader(parameter, year)
        try:
            from policyengine_us_data.parameters import load_take_up_rate
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PolicyEngine-US data take-up rates require policyengine-us-data."
            ) from exc
        return load_take_up_rate(parameter, year)


@dataclass(frozen=True)
class PolicyEngineUSMicrosimulationMaterializer:
    """Lazy optional backend that evaluates variables with ``policyengine-us``."""

    dataset_factory: Any | None = None
    simulation_factory: Any | None = None

    def materialize_policyengine_variables(
        self,
        entity_frames: Mapping[EntityType, pd.DataFrame],
        *,
        variables_by_entity: Mapping[EntityType, Sequence[str]],
        period: int | str,
    ) -> Mapping[EntityType, pd.DataFrame]:
        frames = _copy_entity_frames(entity_frames)
        if not _nonempty_variables(variables_by_entity):
            return frames
        dataset = self._build_dataset(frames, period)
        simulation = self._build_simulation(dataset)
        for entity, variables in _nonempty_variables(variables_by_entity).items():
            if entity not in frames:
                raise ValueError(
                    "Cannot materialize PolicyEngine-US variable(s) "
                    f"{tuple(variables)} without an {entity.value!r} frame."
                )
            frame = frames[entity].copy()
            for variable in variables:
                frame[variable] = _calculate_policyengine_variable(
                    simulation,
                    variable,
                    period=period,
                    entity=entity,
                    expected_rows=len(frame),
                )
            frames[entity] = frame
        return frames

    def _build_dataset(
        self,
        entity_frames: Mapping[EntityType, pd.DataFrame],
        period: int | str,
    ) -> Any:
        if self.dataset_factory is not None:
            return self.dataset_factory(entity_frames, period)

        missing = [
            entity.value
            for entity in (
                EntityType.PERSON,
                EntityType.HOUSEHOLD,
                EntityType.TAX_UNIT,
            )
            if entity not in entity_frames
        ]
        if missing:
            raise ValueError(
                "PolicyEngine-US materialization requires entity frames for "
                f"person, household, and tax_unit; missing {missing}."
            )
        try:
            from policyengine_us.data import USSingleYearDataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PolicyEngine-US materialization requires policyengine-us."
            ) from exc

        return USSingleYearDataset(
            person=entity_frames[EntityType.PERSON].copy(),
            household=entity_frames[EntityType.HOUSEHOLD].copy(),
            tax_unit=entity_frames[EntityType.TAX_UNIT].copy(),
            spm_unit=entity_frames.get(EntityType.SPM_UNIT, pd.DataFrame()).copy(),
            family=entity_frames.get(EntityType.FAMILY, pd.DataFrame()).copy(),
            marital_unit=pd.DataFrame(),
            time_period=int(period),
        )

    def _build_simulation(self, dataset: Any) -> Any:
        if self.simulation_factory is not None:
            return self.simulation_factory(dataset)
        try:
            from policyengine_us import Microsimulation
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PolicyEngine-US materialization requires policyengine-us."
            ) from exc
        return Microsimulation(dataset=dataset)


def _variables_by_spec_entity(
    variables: Mapping[str, VariableSpec],
) -> dict[EntityType, tuple[str, ...]]:
    grouped: dict[EntityType, list[str]] = {}
    for name, variable in variables.items():
        if variable.entity is None:
            raise ValueError(
                f"PolicyEngine-US runtime variable {name!r} is missing an entity."
            )
        entity = EntityType(variable.entity)
        grouped.setdefault(entity, []).append(name)
    return {entity: tuple(names) for entity, names in grouped.items()}


def _single_frame_entities(
    frame: pd.DataFrame,
    variables_by_entity: Mapping[EntityType, Sequence[str]],
) -> dict[EntityType, pd.DataFrame]:
    return {entity: frame.copy() for entity in variables_by_entity}


def _takeup_variables_from_modifier(
    modifier: TargetSimulationModifier,
    targets: Sequence[TargetSpec],
) -> dict[EntityType, tuple[str, ...]]:
    variables: list[str] = []
    parameters = modifier.parameters
    raw_variables = parameters.get("variables") or parameters.get("features")
    if raw_variables is not None:
        if isinstance(raw_variables, str):
            variables.append(raw_variables)
        else:
            variables.extend(str(variable) for variable in raw_variables)
    program = parameters.get("program")
    if program is not None:
        variables.extend(TAKEUP_VARIABLES_BY_PROGRAM.get(str(program), ()))

    if not variables:
        target_features = [
            feature
            for target in targets
            for feature in target.required_features
            if feature in TAKEUP_VARIABLE_ENTITY
        ]
        variables.extend(target_features)

    unknown = [
        variable for variable in variables if variable not in TAKEUP_VARIABLE_ENTITY
    ]
    if unknown:
        raise ValueError(
            f"Unknown PolicyEngine-US take-up variable(s): {sorted(set(unknown))}."
        )

    grouped: dict[EntityType, list[str]] = {}
    for variable in dict.fromkeys(variables):
        grouped.setdefault(TAKEUP_VARIABLE_ENTITY[variable], []).append(variable)
    return {entity: tuple(names) for entity, names in grouped.items()}


def _assert_policyengine_us_modifier(modifier: TargetSimulationModifier) -> None:
    model = modifier.parameters.get("model")
    if model not in (None, "policyengine-us"):
        raise ValueError(
            "PolicyEngine-US materialization modifier requires "
            f"model='policyengine-us', got {model!r}."
        )


def _missing_required_features(
    targets: Sequence[TargetSpec],
    entity_frames: Mapping[EntityType, pd.DataFrame],
) -> dict[EntityType, tuple[str, ...]]:
    grouped: dict[EntityType, list[str]] = {}
    for target in targets:
        frame = entity_frames.get(target.entity)
        present = set(frame.columns) if frame is not None else set()
        missing = [
            feature for feature in target.required_features if feature not in present
        ]
        if missing:
            grouped.setdefault(target.entity, []).extend(missing)
    return {
        entity: tuple(dict.fromkeys(variables)) for entity, variables in grouped.items()
    }


def _nonempty_variables(
    variables_by_entity: Mapping[EntityType, Sequence[str]],
) -> dict[EntityType, tuple[str, ...]]:
    return {
        entity: tuple(dict.fromkeys(str(variable) for variable in variables))
        for entity, variables in variables_by_entity.items()
        if variables
    }


def _copy_entity_frames(
    entity_frames: Mapping[EntityType, pd.DataFrame],
) -> dict[EntityType, pd.DataFrame]:
    return {
        entity if isinstance(entity, EntityType) else EntityType(entity): frame.copy()
        for entity, frame in entity_frames.items()
    }


def _validated_entity_frame_update(
    base_frames: Mapping[EntityType, pd.DataFrame],
    updated_frames: Mapping[EntityType, pd.DataFrame],
) -> dict[EntityType, pd.DataFrame]:
    if not isinstance(updated_frames, Mapping):
        raise TypeError(
            "PolicyEngine-US runtime backend returned "
            f"{type(updated_frames).__name__}; expected a mapping of entity frames."
        )
    merged = _copy_entity_frames(base_frames)
    for raw_entity, frame in updated_frames.items():
        entity = (
            raw_entity if isinstance(raw_entity, EntityType) else EntityType(raw_entity)
        )
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                "PolicyEngine-US runtime backend returned "
                f"{type(frame).__name__} for {entity.value!r}; "
                "expected pandas.DataFrame."
            )
        if entity in base_frames and len(frame) != len(base_frames[entity]):
            raise ValueError(
                "PolicyEngine-US runtime backend changed row count for "
                f"{entity.value!r} from {len(base_frames[entity])} to {len(frame)}."
            )
        merged[entity] = frame.copy()
    return merged


def _merge_runtime_columns(
    frame: pd.DataFrame,
    entity_frames: Mapping[EntityType, pd.DataFrame],
    variables_by_entity: Mapping[EntityType, Sequence[str]],
) -> pd.DataFrame:
    result = frame.copy()
    for entity, variables in variables_by_entity.items():
        entity_frame = entity_frames.get(entity)
        if entity_frame is None:
            raise ValueError(
                f"PolicyEngine-US runtime backend returned no {entity.value!r} frame."
            )
        if len(entity_frame) != len(result):
            raise ValueError(
                "PolicyEngine-US runtime backend returned "
                f"{len(entity_frame)} {entity.value!r} rows for a flat frame "
                f"with {len(result)} rows."
            )
        missing = [variable for variable in variables if variable not in entity_frame]
        if missing:
            raise ValueError(
                "PolicyEngine-US runtime backend did not materialize "
                f"{entity.value!r} variable(s): {missing}."
            )
        for variable in variables:
            result[variable] = entity_frame[variable].reset_index(drop=True)
    return result


def _rate_for_takeup_variable(
    variable: str,
    rates: Mapping[str, float] | PolicyEngineUSTakeupRateSource,
    *,
    frame: pd.DataFrame,
    period: int | str,
) -> float | np.ndarray:
    if isinstance(rates, PolicyEngineUSTakeupRateSource):
        return _validate_rate(
            variable,
            rates.rate_for_takeup_variable(variable, frame, period=period),
            expected_rows=len(frame),
        )
    program = TAKEUP_VARIABLE_PROGRAM.get(variable)
    raw_rate = rates.get(variable)
    if raw_rate is None and program is not None:
        raw_rate = rates.get(program)
    if raw_rate is None:
        raise ValueError(f"No take-up rate supplied for {variable!r}.")
    return _validate_rate(variable, raw_rate, expected_rows=len(frame))


def _validate_rate(
    variable: str,
    raw_rate: Any,
    *,
    expected_rows: int,
) -> float | np.ndarray:
    array = np.asarray(raw_rate, dtype=float)
    if array.ndim == 0:
        rate = float(array)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Take-up rate for {variable!r} must be in [0, 1].")
        return rate
    if len(array) != expected_rows:
        raise ValueError(
            f"Take-up rates for {variable!r} have {len(array)} rows; "
            f"expected {expected_rows}."
        )
    if not np.all((0.0 <= array) & (array <= 1.0)):
        raise ValueError(f"Take-up rates for {variable!r} must be in [0, 1].")
    return array


def _scalar_rate(variable: str, raw_rate: Any) -> float:
    rate = float(raw_rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"Take-up rate for {variable!r} must be in [0, 1].")
    return rate


def _rates_by_string_column(
    variable: str,
    frame: pd.DataFrame,
    rates: Mapping[Any, Any],
    *,
    column: str,
    default: float,
) -> np.ndarray:
    if column not in frame:
        raise ValueError(
            f"Legacy {variable!r} take-up rates require column {column!r}."
        )
    mapped_rates = {
        _string_key(key): _scalar_rate(variable, value) for key, value in rates.items()
    }
    default_rate = _scalar_rate(variable, default)
    return np.array(
        [
            mapped_rates.get(_string_key(value), default_rate)
            for value in frame[column].to_numpy()
        ],
        dtype=float,
    )


def _rates_by_numeric_child_count(
    variable: str,
    frame: pd.DataFrame,
    rates: Mapping[Any, Any],
    *,
    column: str,
    default: float,
    max_key: int,
) -> np.ndarray:
    if column not in frame:
        raise ValueError(
            f"Legacy {variable!r} take-up rates require column {column!r}."
        )
    numeric_rates = {
        int(key): _scalar_rate(variable, value) for key, value in rates.items()
    }
    default_rate = _scalar_rate(variable, default)
    values = []
    for raw_value in frame[column].to_numpy():
        key = min(int(raw_value), max_key)
        values.append(numeric_rates.get(key, default_rate))
    return np.asarray(values, dtype=float)


def _string_key(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _period_year(period: int | str) -> int:
    if isinstance(period, str):
        return int(period[:4])
    return int(period)


def _stable_rng(seed: int, variable: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{variable}".encode()).digest()
    variable_seed = int.from_bytes(digest[:8], "little") % (2**32)
    return np.random.default_rng(variable_seed)


def _calculate_policyengine_variable(
    simulation: Any,
    variable: str,
    *,
    period: int | str,
    entity: EntityType,
    expected_rows: int,
) -> np.ndarray:
    calculate = getattr(simulation, "calc", None) or getattr(
        simulation,
        "calculate",
        None,
    )
    if calculate is None:
        raise TypeError(
            "PolicyEngine-US simulation backend must expose calc() or calculate()."
        )
    try:
        values = calculate(
            variable,
            period=period,
            map_to=entity.value,
            use_weights=False,
        )
    except TypeError:
        values = calculate(variable, period=period, map_to=entity.value)
    array = np.asarray(values)
    if len(array) != expected_rows:
        raise ValueError(
            "PolicyEngine-US calculation returned "
            f"{len(array)} rows for {variable!r} on {entity.value!r}; "
            f"expected {expected_rows}."
        )
    return array


__all__ = [
    "POLICYENGINE_US_RUNTIME_HANDLER",
    "POLICYENGINE_US_TAKEUP_HANDLER",
    "PolicyEngineUSMicrosimulationMaterializer",
    "PolicyEngineUSRuntimeAdapter",
    "PolicyEngineUSDataTakeupRateSource",
    "PolicyEngineUSTakeupRateSource",
    "PolicyEngineUSTakeupRerandomizer",
    "PolicyEngineUSVariableMaterializer",
    "SeededPolicyEngineUSTakeupRerandomizer",
    "TAKEUP_VARIABLE_ENTITY",
    "TAKEUP_VARIABLE_PROGRAM",
    "TAKEUP_VARIABLE_RATE_PARAMETER",
    "TAKEUP_VARIABLES_BY_PROGRAM",
]
