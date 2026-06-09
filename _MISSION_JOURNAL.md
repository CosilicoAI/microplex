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
