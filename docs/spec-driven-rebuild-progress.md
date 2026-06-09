# Spec-driven Microplex rebuild — progress

Status as of this overnight session. Branch: `claude/spec-driven-engine`
(repo `PolicyEngine/microplex`). Authoritative design:
[`docs/spec-driven-rebuild.md`](./spec-driven-rebuild.md).

## What's built, tested, and pushed

The five foundation modules from the build brief are complete: each is
implemented, unit-tested on small **synthetic** frames (no real CPS/PUF data
build — that's a later phase), and pushed. The spine now uses a seeded 50/50
split with the eCPS synthetic-half correctness anchor. The follow-up runner
correction pins the default to canonical regime-aware microimpute.

| # | Module | Class / entry point | Tests | Status |
|---|--------|--------------------|-------|--------|
| 1 | `src/microplex/spec.py` | `MicroplexSpec`, `load_spec`, `load_spec_dict` | `tests/spec/test_spec.py` (34) | pushed |
| 2 | `src/microplex/spine.py` | `SpineBuilder`, `SpineBuildResult` | `tests/spec/test_spine.py` (18) | pushed |
| 3 | `src/microplex/imputation.py` | `ImputationRunner`, `spine_first_order` | `tests/spec/test_imputation.py` (25) | pushed |
| 4 | `src/microplex/spec_transforms.py` | `TransformEngine` | `tests/spec/test_spec_transforms.py` (13) | pushed |
| 5 | `src/microplex/run.py` | `run_spec`, `resolve_sources`, `RunResult` | `tests/spec/test_run.py` (10) | pushed |

**Spec tests after PR #18: 98 passing.** Fixture spec:
`tests/spec/fixtures/us_2024.yaml`.

### 1. `microplex.spec` — the DSL (blueprint §1)
Pydantic v2 schema + YAML loader for the full DSL: `meta`, `sources`
(name→{dataset, role}), `spine` (base, method: support_spine, support{seed},
halves[{name, keep|strip_to}]), `imputation` (steps {at?, onto, from, vars,
condition_on?, order?, synthesize?}, where `at` defaults to `halves` and
`at: base` runs before the spine split), `transforms` (split/derive),
`targets` ({arch:{country,model_year}}), `calibrate` ({loss, method,
target_records?}). Strict (`extra="forbid"`) with cross-reference validation:
exactly one spine source, `spine.base` resolves and is the spine source,
imputation `at: halves` `onto` references a declared half or `both`,
`at: base` `onto` references `base` or the spine source, `from` references a
declared source, exactly-one `keep`/`strip_to` per half, exactly one
passthrough half, fractional split sums to 1, etc. `load_spec(path)` and
`load_spec_dict(mapping)` raise a single `SpecError` with field-pathed
messages.

### 2. `microplex.spine` — `SpineBuilder` (blueprint §4)
The support-spine pattern generalized over a seeded 50/50 partition. Each
base row appears once: in either the passthrough (`keep: all`) half, which keeps
all columns, or the synthetic (`strip_to`) half, which keeps only its declared
columns plus ids/weights
(so its income tail is synthesized from scratch, not inherited — the
correctness anchor). Synthetic ids are offset and synthetic weights start at
zero. Appends a half-label column.
Country-agnostic: the `demographics` group token is resolved via a
caller-supplied `column_groups` mapping, never hard-coded.

### 3. `microplex.imputation` — `ImputationRunner` (blueprint §2 stage 4, the heart)
For each step, fits canonical regime-aware `microimpute.Imputer` on the donor
frame (unweighted unless the step declares `weights: <donor column>`),
conditioned on the resolved `condition_on` (default: the half's demographic
columns), and writes the
predicted columns onto the target half. **Chaining is microimpute's job** — the runner only orders the
variable list; `spine_first_order` is the generic, documented, overridable
heuristic (income/receipt-type keywords first, stable within tiers) so
dependents chain on the income spine. Respects passthrough: a column the half
already has is preserved unless the step sets `synthesize: true`. `run()`
applies the whole graph, expanding `both` to every half and threading
sequential steps through a working copy.

Chaining is verified via the runner's declared chain map, not encoded model
feature names: e.g. with `imputed_variables=[employment_income, capital_gains]`,
`predictors["capital_gains"]` contains `"employment_income"`.

### 4. `microplex.spec_transforms` — `TransformEngine` (blueprint §2 stage 5)
Applies declared `split`/`derive` rules deterministically. A fractional split
partitions a source column into named pieces that **sum back to the source**
(asserted within tolerance); a derive evaluates a pandas-eval expression. Later
transforms see earlier outputs; `apply()` never mutates input.

> **Naming note.** Named `spec_transforms` rather than `transforms` because
> `microplex.transforms` already exists and houses the unrelated
> numeric/array variable transformers (`LogTransform`, `Standardizer`,
> `ZeroInflatedTransform`, …) used for neural-network training. The blueprint's
> "TransformEngine" is a distinct, spec-driven concern, so it lives in its own
> module to avoid clobbering that public surface. If a later phase wants the
> flat `microplex.transforms.TransformEngine` name, the variable transformers
> should be moved/renamed first (e.g. to `microplex.variable_transforms`) as a
> deliberate, separately-reviewed change.

### 5. `microplex.run` — `run_spec` (blueprint §2, §6)
Sequences the wired stages: `resolve_sources` → `at: base` `ImputationRunner`
→ `SpineBuilder` → `at: halves` `ImputationRunner` → `TransformEngine`.
Returns a `RunResult` with the post-transform stacked frame, the pre-split
base frame after source-level imputations, the spine result, per-half frames,
per-step imputation outcomes, and, when a `TargetProvider` is supplied, the
spec-declared `TargetSet`. The end-to-end smoke test runs the full sequence on
tiny synthetic CPS/PUF/SCF frames and asserts the output column set, that a
base-imputed `net_worth` predictor survives the split into both halves, that the
synthetic half's income is synthesized while the kept half's real income is
preserved (passthrough), that a split sums back, and that the target provider
receives the declared Arch country/model-year/profile query.

`run_spec` can now run the generic national/entity-table calibration path when
the caller supplies either `calibration_entity` plus `calibration_id_column` or a
prepared `calibration_entity_bundle`: it builds or accepts an `EntityTableBundle`,
compiles the loaded `TargetSet` into the certified sparse target matrix path,
fits through `microcalibrate`, and returns the calibrated frame plus stable
diagnostics. It still refuses to calibrate without a spec-declared loaded target
surface, a stable non-null unique record id for one-table calibration, and an
explicit `calibrate` section.

### 5a. `microplex.source_registry` — provider-backed source resolution
`SourceRegistry` bridges spec content to real provider loaders: it resolves each
`sources[*].dataset` id through a registered `SourceProvider`, loads and
validates an `ObservationFrame`, and selects the declared entity table for the
frame-based runner. Sources with multiple entity tables must declare
`sources.<name>.entity` in the spec or register a `default_entity`; otherwise
resolution fails closed instead of guessing.

### 5b. `microplex.data_sources` — first ASEC+PUF source providers
The first real-data providers now materialize validated `ObservationFrame`s for
the US ASEC/CPS spine and PUF tax donor. `CPSAsecSourceProvider` exposes
household, person, and tax-unit tables and defaults the frame-based runner to
tax units; `PUFSourceProvider` exposes a tax-unit table with stable ids and
weights. `create_us_asec_puf_source_registry` registers only the first
critical source pair (`cps_asec_2025_calendar_2024` and `puf_2024`) so the
pipeline can exercise ASEC+PUF before adding SCF/SIPP/ACS.

### 5c. `microplex.data_sources.asec_puf_smoke` — first real-data support-spine run
`microplex-us-asec-puf-smoke` is a maintained smoke command for the first
construction stage: load the ASEC/CPS and PUF providers, validate their shared
tax-unit surface, and run the seeded 50/50 support-spine split. It accepts
explicit local PUF and demographics CSV paths because the IRS PUF is
restricted-access; `PUFSourceProvider` now treats those handoff files as a
first-class input path rather than depending only on a Hugging Face fetch.

Codex ran the command on 2026-06-08 with cached 2025 ASEC and local 2015 PUF
files, capped to 5,000 CPS rows and 10,000 PUF rows. The smoke produced a
5,000-row support spine split 2,500/2,500 between `cps_keep` and
`synthetic_puf`, confirmed no ASEC/PUF shared-variable gaps, and initialized
synthetic-half household weights to zero.

### 6. `microplex.stage_manifest` — strict stage artifact manifests
Build stages can now write a self-verifying JSON manifest with schema version,
stage id, declared seeds, parameters, metadata, and per-artifact relative path,
size, and SHA-256. Load/resume code can call
`assert_stage_manifest(..., root=...)` before reusing a checkpoint; missing
files, changed bytes, absolute paths, and `..` path escapes fail closed. This is
the generic core primitive that replaces ad hoc US-only manifest checks as the
fresh pipeline adds SourceRegistry, calibration, and export stages.

### 7. `microplex.calibration.solve_policy` — explicit calibration solve policy
Calibration requests now resolve through a small fail-closed policy object
before a solver runs. APG without `target_records` is dense microcalibrate;
APG with `target_records` means "APG + L0 prune"; `method: l0` requires an
explicit `target_records`; and IPF rejects `target_records`. The policy also
rejects empty target surfaces, prune counts larger than the available records,
and an optional minimum-records-per-target floor. This gives manifests and
country adapters one stable place to log solver/pruning semantics instead of
letting a build infer them ad hoc.

`MicrocalibrateAdapter.fit_sparse_target_matrix_with_policy(...)` now applies
that policy directly to a certified `SparseTargetMatrix`: it validates the
certificate, rejects invalid solver/pruning requests before fitting, flips
microcalibrate's L0 regularization only through the resolved policy, and returns
weights plus stable policy/certificate/validation diagnostics for stage
manifests.

## Notable correctness fix found while building

`ImputationRunner` initially passed only `[predictors + imputed_vars]` to
`Imputer.fit`. If a step explicitly declares `weights: <donor column>`,
microimpute routes **non-numeric** (categorical/boolean) targets to an auxiliary
imputer that reads `weight_col` off `X_train` *by name*, so the declared weight
column must be included in the training frame even though it is neither a
predictor nor an imputed variable. Omitted `weights` remains unweighted, by
design.

## Deliberately NOT wired (clear TODOs, not faked)

`run_spec` reports not-run stages via `PENDING_STAGES = ("targets",
"calibrate", "export")`. If a `TargetProvider` is supplied, the `targets`
stage is no longer pending: the runner builds a `TargetQuery` from
`targets.arch` (`country`, `model_year`, `target_profile`,
`calibration_target_profile`) and attaches the loaded `TargetSet` to the
result. If `calibration_entity` and `calibration_id_column` are supplied, or if
a prepared `calibration_entity_bundle` is supplied, `calibrate` runs through the
generic `EntityTableBundleMicrocalibrator`; if not, it remains pending. No
weights or calibrated datasets are fabricated. Remaining TODOs:

- **`calibrate` (concrete US wiring/clone-local variants):** the
  national/entity-table path and prepared multi-entity bundle path are wired.
  Real-data builders still need to pass the resolved US entity tables and
  runtime simulation compiler explicitly; more complex local-area clone
  surfaces should be added only as real data demands them.
- **`export` (Exporter):** write the PolicyEngine dataset.

## Stretch goal (Arch provider move) — assessed and deferred, not attempted

The brief's stretch was to begin moving the generic Arch target provider +
rollups out of `microplex-us/src/microplex_us/targets/arch.py` into
`microplex/targets/arch_provider.py`. After reading the source
(`/Users/maxghenis/.claude-worktrees/microplex-us-microimpute/src/microplex_us/targets/arch.py`,
**7,116 lines**, with a **4,145-line** test suite at
`tests/targets/test_arch.py`), I deferred this rather than thrash, per the
"faithful move or skip" instruction. Reasons:

1. **It is not cleanly generic yet.** The providers (`ArchSQLiteTargetProvider`,
   `ArchFactSQLiteTargetProvider`, `ArchConsumerFactJSONLTargetProvider`,
   `ArchCompositeSQLiteTargetProvider`) import `microplex_us.geography`,
   `microplex_us.microdata_roles`, and
   `microplex_us.policyengine.target_profiles`, and resolve records via
   `arch_target_record_to_canonical_spec` (US PolicyEngine profile resolution).
2. **Carry-forward / rollup / component-sum are parameterized by US constant
   tables, not generic config.** `_is_latest_carry_forward_candidate` hard-codes
   `source == "SSA"` and `ARCH_LATEST_CARRY_FORWARD_VARIABLES`;
   `_component_sum_records` uses `ARCH_COMPONENT_SUM_TARGETS`; the
   state→national rollup calls US-specific BEA/NIPA helpers and emits
   `concept="policyengine_us.…"`. They operate on the US-shaped
   `ArchTargetRecord` (jurisdiction, `SOIAgingFactors`, state FIPS).
3. **The existing tests are US-behavioral**, not generic: SOI aging, AGI
   brackets, EITC child counts, congressional districts, census STC state
   income tax, PolicyEngine target cells. They cannot validate a generic core
   module against "the same Arch fixtures" without dragging `microplex_us` into
   core — which violates the core's country-agnostic contract (the whole point
   of the rebuild).

A faithful move therefore requires real design work that belongs in a dedicated
phase: define a **generic, country-neutral target-record protocol**; lift
carry-forward / state→national rollup / component-sum onto it with every
US-specific constant (`ARCH_LATEST_CARRY_FORWARD_VARIABLES`,
`ARCH_COMPONENT_SUM_TARGETS`, source aliases, the SSA gate, BEA wage
synthesis) **injected** rather than hard-coded; decouple from
`policyengine_us` profile resolution; and write a fresh synthetic generic test
corpus. Rushing it overnight would produce either an unfaithful half-move or
US-leakage into core. Note: core *already* has a small neutral Arch helper
layer — `src/microplex/targets/arch.py` (171 lines, `ArchConsumerFact` +
JSONL loaders) and the generic `TabularRollupTargetProvider` in
`src/microplex/targets/rollups.py` — which is the right foundation to build the
generic provider/rollups on when that phase starts.

## How to run the tests

A persistent venv at `.venv` has `pytest`, `ruff`, and `microimpute` (editable
from the reconcile worktree) installed directly, so test runs don't churn
`uv.lock`:

```bash
# microimpute (the canonical Imputer) installed editable into .venv:
uv pip install --python .venv/bin/python -e \
  /Users/maxghenis/.claude-worktrees/microimpute-reconcile

# run the spec-engine suite:
.venv/bin/python -m pytest tests/spec/ -q
.venv/bin/ruff check src/microplex/{spec,spine,imputation,spec_transforms,run}.py tests/spec/
```

(The blueprint's suggested `uv run --with-editable …/microimpute-reconcile …`
also works but re-resolves and rewrites `uv.lock`, which is why the committed
runs use the direct-venv path and `uv.lock` is left untouched.)

## What remains (next phases, in blueprint build order)

1. **Wire export into `run_spec`** and keep the generic calibration path focused
   on the immediate national target surface first.
2. **The Arch provider move** as scoped above — generic record protocol +
   injected config + fresh generic tests.
3. **Continue provider-backed source loading** behind `SourceRegistry` for the
   US data lane: exercise the new ASEC/CPS+PUF pair on real data first, then add
   SCF/SIPP/ACS providers once that thin flow is running.
4. **`microplex-us/specs/us-2024.yaml`** — the pure-spec pack.
5. **The real data build + validation harness vs. frozen eCPS** (blueprint §5,
   §6 step 6) — the multi-hour CPS/PUF run, explicitly out of scope here.
