# Spec-driven Microplex: the rebuild blueprint

**Goal.** A country pack (`microplex-us`) that is a **pure declarative spec — zero logic-Python** — exactly the way `rulespec-us` is pure `.rac`/rulespec and the `rulespec` engine does the work. All imperative logic lives in **`microplex` (the engine)** and **`microimpute` (the imputation model)**. Reading the spec should make it *obvious* what happens: which sources, how the spine is built, what is imputed from what (and in what order), what deterministic transforms run, which targets calibrate it.

**Litmus test (the discipline).** Any logic-Python in `microplex-us` is a bug: it means the engine is missing a generic capability. Push it down to `microplex`/`microimpute` and leave a declaration behind. (Tests, packaging `__init__`, and the spec files themselves are the only things in the pack.)

**Current state we're replacing.** `microplex-us/src` is ~77k lines; `pipelines/` alone is ~43k (us.py is 8,759), and `targets/arch.py` is 7k. Almost none of it is US-specific — it's a generic engine that got built inside the country pack. Core is only ~27k. We invert that: core becomes the big engine, the pack becomes a spec.

---

## 1. The spec (the "mp DSL")

A single declarative document per pack/model-year (YAML to start; the schema is a Pydantic model in `microplex.spec`). Shape:

```yaml
# microplex-us/specs/us-2024.yaml  (illustrative)
meta:
  country: us
  model_year: 2024
  policyengine_model: policyengine-us

sources:
  cps_asec:   { dataset: cps_asec_2024, role: spine }      # survey substrate
  puf:        { dataset: puf_2024,      role: donor }      # tax-return donor
  scf:        { dataset: scf_2022,      role: donor }
  sipp:       { dataset: sipp_2023,     role: donor }
  acs:        { dataset: acs_2024,      role: donor }

spine:
  # The eCPS puf_clone pattern, generalized + correct (see §4).
  base: cps_asec
  method: clone
  clone: { seed: 0 }
  halves:
    - { name: cps_keep,       keep: all }                  # real CPS values (passthrough)
    - { name: synthetic_puf,  strip_to: [demographics, tax_unit_id] }  # synthesize the rest

imputation:
  # Declarative graph. Each step = synthesize a set of vars from a donor onto a
  # target half, conditioned on demographics + the already-imputed chain.
  # The engine runs microimpute's canonical regime-aware donor backend per step.
  - { onto: synthetic_puf, from: puf, vars: PUF_TAX_BLOCK,  order: spine_first }
  - { onto: cps_keep,      from: puf, vars: PUF_ONLY_BLOCK, condition_on: [demographics, cps_income] }
  - { onto: both,          from: scf, vars: [net_worth, ...] }
  - { onto: both,          from: sipp, vars: [...] }
  # passthrough is the default for anything a half already has; "synthesize"
  # is explicit. No per-variable Python.

transforms:
  # Deterministic derivations/splits. Declared, not coded.
  - split: { source: social_security, into: {retirement: ..., disability: ..., survivors: ..., ssi: ...} }

targets:
  arch: { country: us, model_year: 2024 }   # the Arch target set; engine fetches + rolls up

calibrate:
  loss: pe_native_bucketed_huber_v1
  method: apg                 # + optional l0 prune to a target record count
```

Notes:
- **Variable blocks** (`PUF_TAX_BLOCK`, etc.) are named lists declared in the spec (or pulled from the model's variable metadata), not Python.
- **`condition_on` / `order`** are the only "knobs" — they map to microimpute's `predictors` + chain order. `spine_first` = a generic ordering that puts wage/total-income/receipt-type variables before the dependent items.
- **passthrough vs synthesize** is a per-variable tag, declared. This is what's left of the 1,300-line `VariableSemanticSpec` — the rest dissolves because microimpute's canonical donor backend now does regime+chaining generically.

---

## 2. The engine (microplex core) — stages that consume the spec

`microplex.run(spec) -> dataset` pipelines these generic stages. Each is country-agnostic.

1. **`spec` (`microplex.spec`)** — Pydantic schema + loader/validator for the DSL above. The single source of truth for "what a pack declares."
2. **`SourceRegistry`** — resolve `sources` to loaded, harmonized frames (already largely in `microplex/data_sources` + the source providers).
3. **`SpineBuilder`** — implement §4 (clone base; one half kept; other half stripped to declared columns). Generic; no country logic.
4. **`ImputationRunner`** — for each `imputation` step, fit microimpute's canonical regime-aware donor backend on the donor (weighted) over the step's var block, conditioned on `condition_on` (+ chain), and apply to the target half. This is the heart; microimpute already does the model. The runner just orchestrates the declared graph + entity grain.
5. **`TransformEngine`** — apply declared `transforms` (splits/derivations) deterministically.
6. **`ArchTargetProvider`** (moved out of `microplex-us/targets/arch.py`) — fetch + roll up Arch target records into `TargetSet`. Generic; the *values* already live in Arch (the pack just names the set).
7. **`Calibrator`** — reweight to targets via the declared loss/method (already in core).
8. **`Exporter`** — write the PolicyEngine dataset (the per-country model integration is the one place a thin country adapter may remain, but it should be driven by the model's own metadata, not bespoke code).

---

## 3. Migration map (what moves where)

| today (microplex-us) | becomes |
| --- | --- |
| `pipelines/us.py` spine construction (~clone) | `microplex.SpineBuilder` (generic), driven by `spine:` spec |
| `pipelines/us.py` donor integration + `donor_imputers.py` | `microplex.ImputationRunner` + canonical microimpute (done), driven by `imputation:` spec |
| `variables.py` `VariableSemanticSpec` (1,300 lines) | dissolves → per-variable `{source, passthrough\|synthesize, order}` tags in the spec; regime/chaining is microimpute's job |
| `targets/arch.py` (7k, "Adapters from Arch records to core specs") | `microplex.targets.ArchProvider` (generic; the providers/rollups/carry-forward are not US-specific) |
| SS-split etc. | declared `transforms:` consumed by `microplex.TransformEngine` |
| `pipelines/pe_native_scores.py` + comparison harness | **stays Python**, in a separate validation repo for now (see §5) |

---

## 4. Correctness anchor: the spine (do NOT keep CPS wages on the synthetic half)

This is the bug that cost the current build. The synthetic-PUF half must carry the **PUF's** income distribution (the tail), not CPS's. So:

- `cps_keep` half: real CPS values (passthrough). Its PUF-only tax detail (cap gains, etc.) is imputed *conditioned on* its real CPS income.
- `synthetic_puf` half: **strip to demographics (+ tax_unit_id)**, then **synthesize the whole PUF tax-unit** (wages included) through microimpute's canonical regime-aware donor backend conditioned on demographics + the chain. Wide distributions on the first chained variables are correct — it's a barely-conditioned pure synthetic PUF. **Never keep/condition on CPS wages here.**

This is exactly what eCPS's `policyengine_us_data/calibration/puf_impute.py:puf_clone_dataset` does by hand ("doubles CPS; one half keeps CPS values, the other half gets PUF tax variables imputed via QRF" from demographic predictors). `SpineBuilder` + `ImputationRunner` implement that pattern generically; the spec declares it.

---

## 5. Validation (stays Python, separate, for now)

The eCPS-replacement comparison (frozen-baseline, clean target surface, the #233/#235 gates, per-variable diagnostics) stays Python and lives outside the pure-spec pack (a separate validation repo or `microplex.eval` for now). It measures any candidate vs frozen production eCPS: full/holdout loss, unweighted MSRE, per-target wins, protected income-family losses. **A rebuild is "done" when a spec-built candidate matches or beats frozen eCPS on the clean surface, with income families not materially worse** — the bar that the current build fails.

---

## 6. Build order

1. `microplex.spec` schema + loader (the DSL above) — foundation.
2. `SpineBuilder` (eCPS puf_clone pattern) + `ImputationRunner` (microimpute) — the heart; get a minimal sources→spine→impute path producing a frame.
3. `ArchProvider` move + `Calibrator` wiring + `Exporter`.
4. `TransformEngine`.
5. `microplex-us/specs/us-2024.yaml` — the pure-spec pack.
6. Build a candidate; run the validation harness vs frozen eCPS; iterate the spec (chain order, condition sets) until it matches/beats eCPS.

Each stage lands with tests in core. The pack stays spec-only.
