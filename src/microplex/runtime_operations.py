"""Runtime handlers for declarative variable operations.

Country packs declare operation kinds in the spec. Core owns the orchestration:
select declarations, route them to registered handlers, and fail closed if a
handler does not produce the declared variables. Country-specific logic such as
PolicyEngine-US takeup or microsimulation remains behind the supplied handler.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from microplex.spec import (
    MicroplexSpec,
    VariableOperationKind,
    VariableSpec,
)

DEFAULT_RUNTIME_VARIABLE_OPERATION_KINDS: tuple[VariableOperationKind, ...] = (
    VariableOperationKind.MATERIALIZE_POLICYENGINE,
    VariableOperationKind.RERANDOMIZE_TAKEUP,
)


@dataclass(frozen=True)
class RuntimeVariableOperationResult:
    """One batch of variables materialized by a runtime handler."""

    kind: VariableOperationKind
    handler: str
    variables: tuple[str, ...]


@runtime_checkable
class RuntimeVariableOperationHandler(Protocol):
    """Handler protocol for runtime spec variable operations."""

    def apply_variable_operations(
        self,
        frame: pd.DataFrame,
        *,
        variables: Mapping[str, VariableSpec],
        spec: MicroplexSpec,
        operation_kind: VariableOperationKind,
    ) -> pd.DataFrame:
        """Return ``frame`` with all requested operation variables present."""


def apply_runtime_variable_operations(
    frame: pd.DataFrame,
    *,
    spec: MicroplexSpec,
    handlers: Mapping[str, RuntimeVariableOperationHandler],
    operation_kinds: Sequence[VariableOperationKind | str] | None = None,
) -> tuple[pd.DataFrame, tuple[RuntimeVariableOperationResult, ...]]:
    """Apply spec-declared runtime variable operations.

    The stage is opt-in: callers pass explicit handlers. When handlers are
    supplied, any selected operation declaration must resolve to a handler and
    the handler must add every declared variable column without changing row
    count.
    """
    selected_kinds = _operation_kind_set(operation_kinds)
    grouped = _group_runtime_variable_operations(spec, selected_kinds)
    if not grouped:
        return frame, ()

    working = frame.copy()
    results: list[RuntimeVariableOperationResult] = []
    for (handler_name, operation_kind), variables in grouped.items():
        handler = handlers.get(handler_name)
        if handler is None:
            raise ValueError(
                "No runtime variable operation handler registered for "
                f"{handler_name!r} ({operation_kind.value}) while materializing "
                f"{sorted(variables)}."
            )
        before_rows = len(working)
        updated = _call_variable_operation_handler(
            handler,
            working,
            variables=variables,
            spec=spec,
            operation_kind=operation_kind,
        )
        if not isinstance(updated, pd.DataFrame):
            raise TypeError(
                f"Runtime variable operation handler {handler_name!r} returned "
                f"{type(updated).__name__}; expected pandas.DataFrame."
            )
        if len(updated) != before_rows:
            raise ValueError(
                f"Runtime variable operation handler {handler_name!r} changed "
                f"row count from {before_rows} to {len(updated)}."
            )
        missing = [name for name in variables if name not in updated.columns]
        if missing:
            raise ValueError(
                f"Runtime variable operation handler {handler_name!r} did not "
                f"materialize declared variable(s): {missing}."
            )
        working = updated
        results.append(
            RuntimeVariableOperationResult(
                kind=operation_kind,
                handler=handler_name,
                variables=tuple(variables),
            )
        )

    return working, tuple(results)


def _operation_kind_set(
    operation_kinds: Sequence[VariableOperationKind | str] | None,
) -> frozenset[VariableOperationKind]:
    raw = (
        DEFAULT_RUNTIME_VARIABLE_OPERATION_KINDS
        if operation_kinds is None
        else operation_kinds
    )
    return frozenset(
        kind if isinstance(kind, VariableOperationKind) else VariableOperationKind(kind)
        for kind in raw
    )


def _group_runtime_variable_operations(
    spec: MicroplexSpec,
    operation_kinds: frozenset[VariableOperationKind],
) -> OrderedDict[tuple[str, VariableOperationKind], dict[str, VariableSpec]]:
    grouped: OrderedDict[tuple[str, VariableOperationKind], dict[str, VariableSpec]]
    grouped = OrderedDict()
    for variable_name, variable in spec.variables.items():
        operation = variable.mp_spec.operation if variable.mp_spec else None
        if operation is None or operation.kind not in operation_kinds:
            continue
        handler_name = operation.handler or operation.kind.value
        key = (handler_name, operation.kind)
        grouped.setdefault(key, {})[variable_name] = variable
    return grouped


def _call_variable_operation_handler(
    handler: RuntimeVariableOperationHandler,
    frame: pd.DataFrame,
    *,
    variables: Mapping[str, VariableSpec],
    spec: MicroplexSpec,
    operation_kind: VariableOperationKind,
) -> pd.DataFrame:
    apply_method = getattr(handler, "apply_variable_operations", None)
    if apply_method is not None:
        return apply_method(
            frame,
            variables=variables,
            spec=spec,
            operation_kind=operation_kind,
        )
    return handler(  # type: ignore[operator]
        frame,
        variables=variables,
        spec=spec,
        operation_kind=operation_kind,
    )
