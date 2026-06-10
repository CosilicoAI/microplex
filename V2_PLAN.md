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

## VERIFIED v2 family-1 rules (from usdata cps.py, importable from ~/.claude-worktrees/usdata-populace):
- tenure_type (HOUSEHOLD) = H_TENURE.map({0:"NONE",1:"OWNED_WITH_MORTGAGE",2:"RENTED",3:"NONE"})
- veterans_benefits=VET_VAL; workers_compensation=WC_VAL (cps.py:1559-60)
- educational_assistance=ED_VAL; financial_assistance=FIN_VAL; survivor_benefits=SRVS_VAL; (1493-95)
- strike_benefits=OI_VAL*(OI_OFF==12); miscellaneous_income=OI_VAL unless alimony/strike (1485-91)
- hours_worked_last_week=A_HRS1; weeks_worked=clip(WKSWORK,0,52) (1367-68)
- detailed_occupation_recode=POCCU2; treasury_tipped_occupation_code + is_tipped_occupation
  via `from policyengine_us_data.datasets.cps.tipped_occupation import derive_*` (1229-34)
- 401k/IRA/Roth contributions: RETCB_VAL proportional split by admin shares — read
  cps.py ~1505-1559 for exact proportions and replicate
- previous_year_income_available + *_last_year: PYEARN/SEMP_VAL prior-year join (1693+) — replicate or defer
- hourly_wage: rule not found yet — grep usdata again (maybe ERN_* or computed)
- RAW columns to add: person ED_VAL FIN_VAL SRVS_VAL VET_VAL WC_VAL OI_VAL OI_OFF CSP_VAL? A_HRS1
  WKSWORK POCCU2 RETCB_VAL PYEARN? SEMP_VAL SSI_VAL; household H_TENURE
- Family-3 (eCPS-donor populace-fit) confirmed remainder: net_worth+scf_*+stock/bond/bank assets,
  first/second_home_mortgage_*, auto_loan_*, vehicles, other/employer health premiums,
  tip_income($54B; eCPS imputes via tipped occ — could derive share*earnings instead),
  weekly_hours_worked_before_lsr, fsla_overtime_premium, sstb_*/qbi/unadjusted_basis (check PUF first).
- RETCB split (verified, cps.py 1505-1552): shares from
  policyengine_us_data/datasets/cps/imputation_parameters.yaml
  (se_pension_share_of_retirement_contributions, dc_share_of_retirement_contributions,
  roth_share_of_dc_contributions, traditional_share_of_ira_contributions).
  se_pension_desired = RETCB*se_share if SEMP_VAL>0; remaining→dc_pool (if WSAL_VAL>0)
  *dc_share, ira_pool=remaining-dc_pool (if earned income); 401k trad/roth and IRA
  trad/roth via the yaml fractions. Need raw RETCB_VAL WSAL_VAL SEMP_VAL.
  NOTE: only *_desired variables are set by eCPS (the non-desired
  traditional_401k_contributions etc. parity gaps may be PE-computed FROM desired —
  verify: if so, adding *_desired closes those gaps automatically).
- hourly_wage / is_paid_hourly / union: usdata imputes from CPS ORG (stage ~line 2948)
  — family-3 (eCPS-donor impute) or defer with documented note.

## Step A progress (committed to worktree git, syntax-checked):
DONE: raw person cols (ED/FIN/SRVS/VET/WC/OI_VAL/OI_OFF/A_HRS1/WKSWORK/POCCU2/
RETCB/WSAL/SEMP) + H_TENURE via extra_household_columns; tenure_type mapped at
Stage A households (flows to spine + export hh); _derive_person_columns v2 block
(benefits, strike/misc income, hours/weeks, occupation, RETCB split with yaml
shares 0.046/0.908/0.15/0.392 — only *_desired set; PE computes limited values).
NEXT in step A: (1) tipped occupation: sys.path usdata-populace +
`from policyengine_us_data.datasets.cps.tipped_occupation import derive_treasury_tipped_occupation_code, derive_is_tipped_occupation`
applied on POCCU2 in _derive_person_columns equivalent (needs occupation arg
form — read that module first); (2) PUF family-2: read src/microplex/data_sources/puf.py
for carried fields, extend PUF_IMPUTE_VARS+DONOR_TO_PE (estate_income e26390?,
misc, e01100 non_sch_d, e03220 educator, e03300 se_pension, e03290 hsa, e58990
form4952, alimony_expense e03500, casualty, 1250 gain, collectibles, qbi/sstb);
(3) prune V1_ZERO_DEFAULTS entries now derived (educator_expense? no — PUF; the
CPS ones: survivor_benefits, educational_assistance, financial_assistance,
tip_income stays, roth/trad 401k/ira *_desired, self_employed_pension_*_desired,
miscellaneous_income, estate_income stays til PUF done); (4) family-3 eCPS-donor
stage (populace-fit QRF, donor=eCPS hf_main flat: scf block+net_worth+assets,
mortgage block, auto/vehicles, premiums, tips fallback, hourly_wage block) —
insert after PUF imputation stage, person/household level as appropriate;
(5) smoke-mode test: `.venv/bin/python scripts/build_us_candidate.py --mode smoke`
must pass end-to-end before full rebuild; (6) parity script refinement: gate on
PE-KNOWN variables only (is_puf_clone flags etc. are bookkeeping, not outputs)
and at SIMULATION level for formula vars (e.g. traditional_401k_contributions
computes from *_desired).

## Step A status @ 00:15 (all committed to worktree git):
DONE substeps 1-3+5(launched): tipped occupation (usdata import, POCCU2);
PUF family-2 (microplex puf.py field map + estate=E26390-E26400 combine +
uprating; driver PUF_IMPUTE_VARS + DONOR_TO_PE incl. capital_gains_distributions
→non_sch_d_capital_gains, puf_miscellaneous_income→miscellaneous_income);
V1_ZERO_DEFAULTS pruned; family-3 stage F2 = scripts/ecps_donor_impute.py
(microimpute weighted QRF, donor=frozen eCPS baseline; person block 13 vars,
household block 28 vars incl. scf_*, mortgage, auto, vehicles, childcare→hh
grain bcast) wired after tail concat using args.baseline_h5 + args.seed.
SMOKE RUNNING: /tmp/v2_smoke.log (bg watcher armed). WATCH-ITEM: smoke warned
form-4952 constant-0 in 8k-row PUF sample — verify nonzero at FULL scale.
NEXT: smoke passes → parity-check smoke output (PE-known vars, simulation-level
for formula vars) → step B FULL rebuild (nohup, ~1-2h, usdata-populace) →
section H observatory lineage while it runs → C recalibrate → D export+surgeries
(GENERALIZE interest_deduction surgery: group-sum EVERY person-stored
tax-unit-entity head-carried variable in build_populace_us_dataset.py — the new
PUF vars land person-stored like interest_deduction did) → E gates → F/G publish.

## @ 00:50: SMOKE GREEN end-to-end (iteration 7; fixes: donor stored-names,
## grain routing + head-carry mortgage block, int64 weight join + drop zero-weight
## donors, one-hot tenure predictors). Smoke output verified: tenure mix
## 2599 owned/1131 rented, net_worth 99% nz, scf_retirement 66%, veterans 1.3%,
## estate 0.8%, tips 0.7%, mortgage 15%, 401k desired 19.9%. weeks_worked
## dropped=1 is BY DESIGN (formula_owned_excluded in export contract) — the
## parity checker must EXCLUDE formula-owned vars. defaulted=118 remain (take-up
## flags etc. — many legit; compare against eCPS-stored when gating).
## FULL REBUILD RUNNING: /tmp/v2_full_build.log (watcher bh4h2ktk6), ~1-2h.
## All 19+ worktree commits pushed to origin/claude/spec-build-20260610.

## @ 01:33 chain state: rebuild 3 GREEN (entity placement: 8 tax_unit moves incl
## interest_deduction at source, spm childcare; tail-unit NaN fix via built-table
## application). Extraction GREEN: v2_target_surface_raw.npz (3704 targets, b
## median 7e5, est/b 0.921, raw initial loss 10.28 — v2 pool starts further out
## than v1's 1.22, the bigger tail stratum at design weights; calibration pulls
## it in). CALIBRATION RUNNING: /tmp/v2_calib.log → populace_us_2024_v2.h5 +
## _timeperiod + npz (script /tmp/build_populace_us_v2.py verifies byte-identical
## weights + no dup columns). NEXT: parity gate (PE-known non-formula vars,
## exclude *_is_puf_clone + weeks_worked) → smoke (Microsimulation, people/HNI/
## snap) → sound comparison (update ~/populace-score-work/run_comparison.py
## candidate to populace_us_2024_v2_timeperiod.h5; train AND holdout beat
## baseline 1.0888/0.3167) → publish chain F/G.
