# Mission journal: spec-built microplex candidate that beats eCPS

**Mission (Max, 2026-06-09):** "i dont want you to stop until you've built a satisfactory
version of microplex that beats the ecps according to the gates we've defined."
Direction: microplex-us is DEPRECATED; every dataset = spec YAML in core `packs/`;
all Python in microplex core.

**Done bar (blueprint §5):** a spec-built candidate matches/beats frozen eCPS on the
clean target surface — full + holdout loss via matched-N, symmetric-refit, holdout
comparison — with protected income families (capital_gains, dividends, interest,
retirement_income) not materially worse. Guards: entity-convergence (#113 — no
microunit-driven "wins"), no raw weight math (#224), weighted-vs-unweighted donor
fits adjudicated by measurement (microplex#76, was microplex-us#263 — unweighted
default was DELIBERATE; support-building vs representation argument).

**Score to beat (old imperative pipeline, a3a1934, 2026-06-07, matched 15,875 hh,
3,701-target surface):** eCPS full 0.05585 / holdout 0.01266; MP candidate full
0.06055 / holdout 0.02778 (MP loses, ~2× on holdout; income families slightly worse).
Artifact: ~/CosilicoAI/microplex-us/artifacts/ecps_shaped_cps_puf_support_clone_release_candidate_canonical_regime_gated_a3a1934_20260607/post_h5_checks/sound_comparison_PRODUCTION_ecps_a3a1934_20260607/sound_ecps_replacement_comparison.json

## Environment
- Build worktree: ~/.claude-worktrees/microplex-spec-build (branch claude/spec-build-20260610
  from origin/claude/spec-driven-engine @ 1ee026f). Venv: .venv (py3.13),
  `uv pip install -e ".[dev,l0,calibrate]" policyengine-us` — microimpute comes from
  [tool.uv.sources] git pin @90be828 (concrete canonical Imputer; reconcile branch,
  PR microimpute#196 unmerged). NEVER plain pip here (ignores tool.uv.sources).
- Old pipeline worktree (scoring env): ~/CosilicoAI/microplex-us (branch g3-aotc) has
  src/microplex_us/pipelines/ecps_replacement_comparison.py + .venv.
- Frozen eCPS baseline: ~/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/enhanced_cps_2024.h5 (106MB)
  + microplex-us artifacts baselines/enhanced_cps_2024_hf_main.h5.
- ASEC parquet: ~/CosilicoAI/microplex/data/cps_asec_{persons,households}.parquet.
- Arch consumer facts (calibration targets): ~/CosilicoAI/microplex-us/artifacts/ecps_shaped_..._arch_calibrated_036fa067_20260604/arch_source_suites/*/consumer_facts.jsonl
  (+ aggregate arch_source_suites_consumer_facts.txt). run_spec takes
  arch_consumer_fact_paths=... directly.
- PUF: NOT yet located locally (codex smoke used "local 2015 PUF" + HF fallback in
  PUFSourceProvider). TODO: find path.
- 48GB Mac; full builds are multi-hour and have been RAM-killed before — checkpoint
  via stage manifests; cap rows for smokes (codex used 5k CPS/10k PUF).

## State / decisions log
- 2026-06-10: #262 closed (wrong direction — core packs/us canonical). #263 → microplex#76
  with deliberate-default nuance. PR #75 (CI on trunk) updated: CI now installs via uv so
  the microimpute git pin is honored — expect the 41 failures to clear; suite was also
  uncollectable before (__init__.py fix in same PR).
- PENDING_STAGES still ('targets','calibrate','export') — targets+calibrate ARE wired
  when bindings supplied (tuple partly stale); EXPORT is genuinely missing → task 4.
- Export contract: packs/us/manifests/ecps_export_contract.json; old exporter reference:
  microplex-us@f3af332 src/microplex_us/policyengine/us.py (do NOT recreate bespoke
  country code; drive from contract + entity bundle).

## Scoping (2026-06-10, verified)
- Harness CLI runs in old venv: `~/CosilicoAI/microplex-us/.venv/bin/python -m
  microplex_us.pipelines.ecps_replacement_comparison --candidate-dataset H5
  --baseline-dataset H5 --output-dir DIR [--matched-household-count N ...]`.
- Export format: USSingleYearDataset(person, household, tax_unit, spm_unit, family,
  marital_unit, time_period).save(path) — policyengine-us native. The engine's
  PolicyEngineUSMicrosimulationMaterializer already builds it in memory (policyengine_us.py:422).
- Contract: 252 required (12 id/weight cols across all 6 entity systems + 240 vars),
  22 forbidden, 5 optional. Content gate asserts spec.variables covers it.
- ENTITY CONSTRUCTION DOES NOT EXIST in new engine (PR #74 was bundle plumbing).
  Sanctioned path: microunit (PyPI 0.1.0, installed) for tax units; native ASEC ids
  for spm (SPM_ID)/family (PF_SEQ); spouse-pointer pairing for marital.
- ASEC zips cached: ~/.cache/microplex/cps_asec_{2024,2025}.zip; pppub25.csv has all
  pointer cols (A_SPOUSE, PEPAR1/2, A_EXPRRP, SPM_ID, PF_SEQ, A_HSCOL; 844 cols).
  Current provider column map DROPS pointer cols — needs extension.
- PUF: ~/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/puf_2024.h5.
- pe-native target caches: ~/.cache/microplex-us/pe-native-baseline* (216MB main).
- microimpute pinned via [tool.uv.sources] git@90be828 (concrete Imputer). pip ignores
  tool.uv.sources → that was the CI failure; PR #75 switched CI to uv.

## Agent deliverables (committed 7a060dc, 61 tests green)
- microplex.units.assign_us_unit_structure(person, year, tax_unit_mode) → UnitAssignmentResult
  (.person + person_{tax_unit,spm_unit,family,marital_unit}_id dense int64 + tax_unit_role_input;
  .tax_unit has filing_status_input). microunit REQUIRES raw: PH_SEQ, A_LINENO, A_AGE,
  A_MARITL, A_SPOUSE, PEPAR1, PEPAR2, A_EXPRRP (+ *_VAL incomes optional; harmonized
  fallback mapping built in). SPM: SPM_ID→household fallback. family: (household,PF_SEQ).
  marital: spouse-pointer pairs.
- microplex.export.export_policyengine_us_dataset(entity_frames, period, output_path,
  contract, defaults, allow_incomplete) → ExportGateResult. USSingleYearDataset contract:
  person_id + person_{group}_id ×5; {group}_id per table; household_weight on household;
  EMPTY TABLES VANISH on save (must have ≥1 row); column names globally unique across
  entities; EntityType has NO MARITAL_UNIT — pass key with .value=='marital_unit'.
  Verified Microsimulation round-trip.
- CPSAsecSourceProvider extra_person_columns/extra_household_columns (raw ASEC passthrough,
  cache-key digest __x<sha12>). packs/us/manifests/export_defaults.json (168 defaults,
  verbatim f3af332). Coverage: 252 required − defaults ⇒ 100 pipeline-owned columns.

## v1 candidate architecture (DECIDED 2026-06-10)
Spine PARTITIONS (50/50), does not clone ⇒ both halves keep real ASEC weights ⇒ no
zero-weight problem; the harness's matched-N symmetric refit does weight optimization.
v1 needs NO own calibration pass for scoring (pool quality is the contest).
Driver flow (scripts/build_us_candidate.py):
1. persons = CPSAsecSourceProvider(persons + extra raw cols incl. A_AGE).
2. units = assign_us_unit_structure(persons) (microunit-first, eCPS-identical construction;
   #113: report as entity-convergence, fine for the gate).
3. tax_units spine base = aggregate persons by person_tax_unit_id (demographics aggregates,
   harmonized income sums, filing_status_input, household_id, weight, state_fips).
4. sources = {"cps_asec": tax_units, "puf": registry puf_2024}; run_spec(spec built like
   build_asec_puf_support_spine_spec + PUF imputation steps lifted from packs yaml
   [puf_support_clone block onto synthetic_puf; puf-only block onto cps_keep]).
5. Person re-attach by tax-unit id (each unit in exactly one half); person-level values:
   cps_keep = ASEC person passthrough + head-allocated PUF-only vars; synthetic_puf =
   head-allocated imputed vars (v1 crudeness — iterate person allocation later).
6. household table from ASEC households (HSUP_WGT→household_weight) + block geography
   (data/block_probabilities*.parquet via smoke's _assign_block_geography pattern)
   → block_geoid/county_fips/congressional_district_geoid.
7. export(..., defaults manifest, allow_incomplete only in --smoke).
8. score: subprocess ~/CosilicoAI/microplex-us/.venv python -m
   microplex_us.pipelines.ecps_replacement_comparison --candidate-dataset OUT.h5
   --baseline-dataset ~/PolicyEngine/policyengine-us-data/policyengine_us_data/storage/enhanced_cps_2024.h5
   --output-dir <dir> (defaults: holdout 0.2 seed 20260529).
PUF source: registry puf_path=…/storage/puf_2024.h5 (verify format; HF fallback).

## Implementation plan (in flight)
- Agent A: src/microplex/units.py + tests — assign_us_unit_structure(person, year, ...)
  → person + person_*_id cols + per-unit tables (microunit tax; SPM/family native;
  marital from spouse pointers). Optional-import microunit like the PE adapter.
- Agent B: src/microplex/export.py + tests — ExportContract.from_path,
  export_policyengine_us_dataset(entity_frames, period, output_path, contract, defaults)
  → USSingleYearDataset.save + gate result (missing/forbidden/defaulted).
- Agent C: data_sources/cps.py extension (extra raw columns w/ cache-key digest) +
  port POLICYENGINE_US_EXPORT_DEFAULTS from microplex-us@f3af332 policyengine/us.py
  (~505-687) → packs/us/manifests/export_defaults.json.
- Me: run_spec wiring + scripts/build_us_candidate.py driver → capped smoke → full
  build → harness score → iterate. Spine column_groups must keep id/pointer cols on
  the synthetic half (strip_to demographics + ids).

## Milestones
- 2026-06-10: PR microplex#75 GREEN (3.12/3.13/3.14) + MERGEABLE after switching CI
  to uv (tool.uv.sources honored). Trunk has working CI; no microimpute release needed.
- Driver: scripts/build_us_candidate.py (smoke: 4k tax units / 8k PUF; full: --mode full --score).
  Iterating smoke: fix1 CPSDataset polars accessors; fix2 resolve PUF via main spec.

## Smoke COMPLETE (2026-06-10 ~01:30): full pipeline + scoring path validated
- Export gate: 252/252 required, 188 defaulted, 0 forbidden. Time-period sibling export
  (candidate_timeperiod.h5) for the harness.
- SCORING ENV (the hard-won recipe): harness = ~/CosilicoAI/microplex-us/.venv python -m
  microplex_us.pipelines.ecps_replacement_comparison; baseline MUST be
  ~/CosilicoAI/microplex-us/artifacts/baselines/enhanced_cps_2024_hf_main.h5 (storage
  copy lacks congressional_district_geoid); delegation MUST pin
  --policyengine-us-data-repo ~/.claude-worktrees/usdata-f7458313 (worktree @ f7458313,
  venv PE 1.715.3 == June-7 certificate env) AND --policyengine-us-data-python
  <that venv> — otherwise `uv run --project` RESYNCS the moving us-data repo (local main
  ahead-198) whose current code breaks (get_soi uprating KeyError) and whose PE 1.570.7
  breaks reform branches (de_tanf ParameterNotFoundError under tax-expenditure targets;
  --skip-tax-expenditure-targets is REFUSED by the sound comparison).
- Smoke's only failure mode left: "target names differ" because head(4000) cap = ME/MA
  only → state targets missing. Cap artifact; full national build resolves.
- Driver defaults updated (hf_main baseline + pinned repo/python flags).

## Build v3 fixes (2026-06-10 ~03:30) — the half-attachment bug
- First full score (v2) was INVALID: only cps_keep exported (32,406 hh, 78.5M weights)
  → candidate 8.00 vs baseline 1.76. Root cause: SpineBuilder RE-IDENTIFIES synthetic-half
  id columns (offset = max - min(0,min) + 1, spine.py:266) and ZEROES weight columns;
  halves frames have RESET indexes (do NOT use them to map to base rows).
- Driver v3: recover original ids via the offset rule (validated: recovered set must
  exactly partition base ids); re-key spm/family/marital per (orig,half); export
  households as (orig household, half) pieces with person-share-prorated original
  weights (total = full ASEC ~135M; synthetic half carries real mass — our budget
  decision vs the engine's zero-weight default; harness refit preserves input totals).
- Smoke v16: 4,000 units both halves, 7,227 persons, 3,360 household pieces, 0 zero
  weights, gate 257/257.

## Score history (sound comparison, matched 41,314 hh, ~3,704-target surface)
| build | config | full loss | holdout | notes |
|---|---|---|---|---|
| eCPS baseline | (refit) | 1.41 | 0.32 | the bar |
| v2 2026-06-10 | INVALID (cps_keep only exported) | 8.00 | 2.31 | half-attachment bug |
| v3 2026-06-10 | both halves, UNWEIGHTED donor fits | 130.1 | 33.6 | landmines at scale: LTCG $200.9T vs $257B, qual div $18.5T vs $154B — microplex#76 A-arm measured |
| v4 (running) | weighted PUF fits (B arm) | ? | ? | |
Old imperative pipeline best (a3a1934, 06-07): cand 0.0606 vs eCPS 0.0559 full / 0.0278 vs 0.0127 holdout
(different matched-N=15,875 + surface scale — not directly comparable to the 41,314-hh refits above).
