# populace-us v2: output parity + gates + full publication

Max's mandate (2026-06-10 ~23:38): **don't stop** until populace is (1)
policyengine-compatible, (2) output parity with eCPS ("exactly the same
outputs": no variable where eCPS is non-degenerate and populace is
degenerate/absent), (3) lower TRAINING and HOLDOUT loss in the sound
comparison, (4) published (HF), (5) integrated in `PolicyEngine/policyengine-bundles`
(bundle manifest like bundles/4.14.0 — pins core+model+data with sha256
release manifests; note 4.14.0 already ships a microplex eCPS release, follow
that contract) and in policyengine.py (src/policyengine — dataset/provenance
registry consumes `hf://owner/repo/path@revision`).

## State (2026-06-10 23:45)
- v1 artifact built+verified (90.58% within-10%, ratio-50 bound, smoke green:
  332.5M people / $14.73T HNI / $97B snap). Scoring run of v1 in
  ~/populace-score-work (plain bg, heartbeat) — read score_out/sound_ecps_replacement_comparison.json
  when present: candidate_refit.optimized_{train,full,holdout}_loss vs baseline_refit.*.
  v1 publication was HELD at review + superseded by the v2 mandate.
- populace repo: PRs #1-3 merged; PR #4 (populace-data shard, registry-driven
  loader, review fixes, LICENSE, provenance snapshot) green-pending; populace.dev/dashboard live.
- Adversarial review verdict on v1: claims fixed; gaps documented (card Known-gaps).
- Microplex automation OFF; kill any `ecps_replacement|microplex_us` python on sight (Max authorized).

## The 82 parity gaps (full list: V2_PARITY_GAPS.txt) by source family
1. **CPS-carried** — extend `load_cps_asec(extra_person_columns=...)` keep-list +
   `_derive_person_columns` in scripts/build_us_candidate.py (~line 504 / 141):
   veterans_benefits (VET_VAL), survivor_benefits (SUR_VAL1/2+SUR_SC),
   workers_compensation (WC_VAL), educational_assistance (ED_VAL?),
   financial_assistance (FIN_VAL), weeks_worked (WKSWORK), hours_worked_last_week
   (HRSWK/A_USLHRS), hourly_wage (PEERNH/A_HRSPAY), detailed_occupation_recode
   (A_DTOCC), treasury_tipped_occupation_code (derive from occupation),
   tenure_type (HOUSEHOLD record: H_TENURE → OWNED_*/RENTED; also fixes
   spm_unit_tenure_type downstream), spm_unit_pre_subsidy_childcare_expenses
   (SPM_CHILDCAREXPNS on SPM record?). Check each against the ASEC dictionary
   in the usdata worktree.
2. **PUF-carried** — extend `PUF_IMPUTE_VARS` + `DONOR_TO_PE` (~line 218):
   estate_income, miscellaneous_income, non_sch_d_capital_gains (e01100),
   investment_income_elected_form_4952 (e58990), educator_expense (e03220),
   qualified_tuition_expenses?, self_employed_pension_contributions (e03300),
   health_savings_account_ald (e03290), domestic_production_ald, partnership_se_income,
   farm_rent_income, salt_refund_income, casualty_loss, unrecaptured_section_1250_gain,
   long_term_capital_gains_on_collectibles, unreimbursed_business_employee_expenses,
   alimony_expense (e03500), qualified_bdc_income, sstb_* / w2_wages / unadjusted_basis
   (CHECK what the registry's PUF source actually carries: microplex
   data_sources/us_registry; eCPS may derive sstb via flags — mirror its rule).
3. **eCPS-as-donor imputation** (SIPP/SCF/ACS-class; use populace-fit weighted QRF,
   train on eCPS hf_main flat h5, predictors age/sex/income/filing/household size):
   net_worth + all scf_* + stock/bond/bank assets (household block),
   401k/IRA/Roth contributions (+_desired), other_health_insurance_premiums,
   employer_sponsored_insurance_premiums, tip_income, first/second_home_mortgage_*
   (balance/interest/origination_year), auto_loan_*, household_vehicles_*,
   self_employment_income_last_year, previous_year_income_available,
   weekly_hours_worked_before_lsr (real hours, not constant 40), fsla_overtime_premium.
   NOTE eCPS donor is legitimate for parity; score gate remains the admin surface.
4. **Bookkeeping**: *_is_puf_clone flags exist but all-zero in the populace
   timeperiod export while the pool flags ~50% — fix the export carry.

## Pipeline (each step gated)
A. Edit build_us_candidate.py: (1) loader columns + household/SPM raw columns,
   (2) _derive_person_columns additions, (3) PUF_IMPUTE_VARS/DONOR_TO_PE
   extensions, (4) remove covered names from V1_ZERO_DEFAULTS, (5) new Stage:
   eCPS-donor imputation for family-3 blocks (populace-fit; weighted bootstrap).
B. Rebuild pool: `cd ~/.claude-worktrees/microplex-spec-build && .venv python
   scripts/build_us_candidate.py --mode full --calendar-year 2024 --usdata-repo
   ~/.claude-worktrees/usdata-populace` (use the PRIVATE usdata copy; v1 build
   took ~1-2h; run via nohup+heartbeat, watch memory, NOT launchd).
C. Recalibrate with populace.calibrate (raw surface re-extract for the NEW pool:
   SCORING_HARNESS_SRC=~/CosilicoAI/microplex-us/src + usdata-populace; then
   epochs=3000 lr=0.15 mass=free max_weight_ratio=50 seed=0).
D. Export USSingleYearDataset + timeperiod; apply year/interest_deduction
   surgeries (build script already does); verify weights byte-identical.
E. GATES: parity script → 0 gaps; smoke (Microsimulation loads; people≈335M,
   snap sane, HNI should RISE toward ~$20T with the new income streams);
   sound comparison → candidate_refit.optimized_train_loss AND
   optimized_holdout_loss both < baseline_refit.*.
F. PUBLISH: HF dataset policyengine/populace-us (h5 + calibration npz + card
   with full validation incl. train/holdout/matched-N/per-target counts);
   merge populace PR #4; update card VALIDATION_TABLE; populace.dev dashboard
   regenerate (+ index evidence, remove "preliminary"); memory.
G. INTEGRATE: policyengine-bundles — new bundle manifest pointing at the HF
   release (follow bundles/4.14.0/countries/us.json schema incl. sha256s,
   install locks via their generator workflow if runnable); policyengine.py —
   register the dataset in its data/provenance registry (read
   src/policyengine/core/dataset.py + provenance/dataset_sources.py for the
   pattern; likely a DatasetSource entry with hf:// URI @revision). File PRs
   to both repos (branch from origin/main, towncrier fragments where required,
   NEVER merge without green CI + MERGEABLE).

## Cautions
- NO Thesis references anywhere public. populace stays PolicyEngine.
- If model ever downgrades to Opus: STOP.
- Commit WIP often (worktree git). Memory-pressure kills: keep ≤1 heavy child;
  the scoring driver pattern (~/populace-score-work/run_comparison.py) works.
- hf_main eCPS baseline stays FROZEN for the comparison.

## H. populace.dev observatory v2 (Max, 2026-06-10 23:47: "if waiting, make
## populace.dev more detailed - show full lineage of each variable, stage gates, etc (versioned)")
- Generator emits VERSIONED documents: populace.dev/data/releases/<release>.json
  (e.g. "us-2024-v1"), with a version switcher on /dashboard. Keep data/calibration.json
  as alias of latest.
- **Variable lineage table**: every stored variable in the artifact → entity,
  source family (cps-carried | cps-derived | puf-imputed | ecps-donor-imputed |
  zero-default-v1 | structural | calibration), fill (nonzero %), weighted total,
  eCPS weighted total alongside, status chip (live | degenerate | scheduled-v2).
  Classification must be PARSED from scripts/build_us_candidate.py constants
  (V1_ZERO_DEFAULTS, PUF_IMPUTE_VARS, _derive_person_columns assignments,
  PERSON_INCOME_COLUMNS) + V2_PARITY_GAPS.txt — never hand-typed.
- **Stage gates**: each pipeline stage gets criterion + measured value +
  verdict (pass/fail/running): pool (record counts), calibrate (loss, within-10%,
  weight bound), parity (gap count vs eCPS — gate 0), smoke (loads + people/HNI/snap),
  score (train AND holdout < baseline), publish (HF revision + bundle id).
- Implementation: extend ~/PolicyEngine/populace.dev/scripts/build_dashboard_data.py
  + dashboard.html/js (vanilla, observatory aesthetic, db-* classes); deploy
  vercel --prod; verify live via curl + screenshot.

## v1 SCORE (2026-06-10 23:47, ~/populace-score-work/score_out, matched 41,314):
candidate_refit optimized train 0.1274 / holdout 0.0319 / full 0.1593
baseline_refit  optimized train 1.0888 / holdout 0.3167 / full 1.4055
→ WIN on all three (8-10x). Per-target: candidate wins 1,192, baseline 2,458,
ties 54 (disclose). v1 loss gates PASS; parity gate (82 gaps) is the v2 work.
