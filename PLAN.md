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

## @ 06:35 GATE STATUS (machine slept 02:00-06:29; chain survived):
- v2.1 calibration: loss 19.27->0.0205, 95.71% within10, max 284,679, 0>500k
  (vs v1 90.58% / v2.0 93.47%) — artifacts populace_us_2024_v2{.h5,_timeperiod.h5} verified.
- PARITY GATE: **PASS, 0 gaps** (PE-known input vars; formula-owned excluded via
  system.variables[var].formulas check).
- SMOKE GATE: running (/tmp/v2_smoke_gate.log) — people/HNI/snap/401k-from-desired/net_worth.
- SCORE GATE: running caffeinated (~/populace-score-work/run_v2.log, score_out_v2/),
  candidate populace_us_2024_v2_timeperiod.h5, gate train AND holdout < baseline
  (v1 precedent 0.1274/1.0888 + 0.0319/0.3167 matched 41,314).
- Observatory: source-family lineage + release switcher LIVE on populace.dev.

## @ 06:40 SMOKE GATE: MIXED. people 331.7M OK, snap $94.2B OK, net_worth
## $165.0T OK (eCPS 163.4T - wealth layer REAL). TWO ISSUES: (1)
## traditional_401k_contributions computes $0 despite desired nz 15.4% in
## timeperiod — PE formula is just desired*scale, so either the
## USSingleYearDataset lost the desired columns (timeperiod has them; check
## /tmp/v2_diag.log first line) or scale=0; (2) household_net_income $11.03T
## vs v1 $14.73T (eCPS $22.24T) — REGRESSION; diag decomposes market income/
## taxes/benefits (v5 pool market income was $15.98T per review). Hypotheses:
## chained-QRF draws shifted when 16 new PUF vars joined the chain; tail
## stratum 10,384 vs v5 15,000; income_tax spike from new deduction inputs is
## NOT plausible (deductions raise net income). SCORE still running on this
## pool (score_out_v2). DO NOT PUBLISH until HNI story understood + 401k fixed.

## @ 06:55 DIAGNOSIS PROGRESS:
- 401k-zero ROOT CAUSE: the export stores all-zero copies of PE FORMULA
  variables (traditional/roth_401k, roth_ira, self_employed_pension
  contributions + spm capped childcare) — a stored input MASKS the formula.
  FIX patched into /tmp/build_populace_us_v2.py: drop all-zero formula-variable
  columns from the published USSingleYearDataset (PE then computes from
  *_desired). Rerun script after HNI verdict (one run covers both).
- HNI: v1 full decomp: emp 10.31 selfemp 0.44 LTCG 1.22 qdiv 0.31 int 0.28
  pension 1.01 SS 1.54 -> mkt 15.98, ben 1.97, tax 3.21 -> HNI 14.74. v2: emp
  10.86 selfemp 0.68 SS 1.48 tax 3.80 ben 2.21 mkt 12.61. Gap must sit in
  capital/pension components and/or NEW NEGATIVE components (estate_income,
  partnership_se_income, rent/royalty losses — v1 lacked them; eCPS has estate
  -$522B, partnership_se -$280B). /tmp/v2_diag2.log computing v2 LTCG/divs/
  interest/pension/rental/partnership/farm + household_tax/benefits.
- SCORE v2 still running (score_out_v2, hb healthy).

## @ 07:05 HNI ROOT CAUSE FOUND + FIXED: short_term_capital_gains -$3.890T
## (eCPS +$0.035T, v1 -$0.148T). 2,020 records < -$1M in the MAIN pool (tail
## only -$0.015T; min -$140.6M = uprated PUF extreme). Cause: our tail-faithful
## QRF (the microimpute#196+interpolation fix) draws PUF extreme losses far
## more often than eCPS's old nearest-snap tail-thinning forest. FIX: 
## realized-support guard in the driver pre-export — clip every imputed
## person-grain value to the BASELINE's per-record [min,max] (parity-bounded,
## documented; STCG floor becomes -$10.0M). v2 SCORE killed (pool changes).
## REBUILD 5 RUNNING with guard; then: extract -> calibrate (also drops
## formula-masking zero columns now) -> parity/smoke (expect HNI to recover
## toward ~$14.5-15T and 401k >$100B) -> FRESH score -> publish chain.

## @ 07:15 REBUILD 5 GREEN: support guard live (STCG 210 clips to
## [-9.98M, +1.65B] = eCPS realized range; farm/ira/etc clipped too; NOTE
## farm_income floor 0 per eCPS support). Extraction running, auto-chains
## into calibration (watcher b9erfnbn6 launches build_populace_us_v2.py on
## SAVED). Then gates: parity (expect 0) / smoke (HNI ~$14.5T+, 401k >$100B,
## STCG ~±$0.2T) / fresh score (rm score_out_v2 first; copy v2 timeperiod to
## ~/populace-score-work). Then publish chain.

## @ 07:30 v2.2 GATES: parity PASS (0 gaps). Smoke: people 332.7M / snap $93.2B
## / 401k $338.4B (mask fix WORKS) / net_worth $163.6T / HNI $13.68T / STCG
## -$0.907T. STCG story CLOSED with full mechanism: donor weighted truth
## -$76.8B uprated; pool at DESIGN weights -$164B (faithful); CALIBRATION
## amplifies to -$907B (optimizer loads weight onto big-loss records to hit
## visible targets; STCG not on the surface — same blind spot exists for eCPS
## in principle, thinner tails in practice). v3 roadmap: net-STCG calibration
## target. v2 ships with honest card note. Calibration v2.2: 2.288->0.0220,
## 95.55% within10, max 382,478, 0>500k, formula-masks dropped. FRESH SCORE
## RUNNING (score_out_v2, caffeinated). On train+holdout WIN -> publish chain.

# V3: ECPS-FREE BUILD (Max, 2026-06-11: "this should REPLACE the ecps - we
# want to totally get rid of it" — eCPS may appear ONLY as the scoring
# baseline, never as a build input)
Contamination scope (confined to the v2 parity-fill): ecps_donor_impute.py's
~40 vars (wealth/mortgage/vehicles/premiums/tips/hourly/prior-year) + the
support guard's eCPS ranges. Core economics already primary (CPS+PUF).
Proven cost: investment_interest_expense $33.5T (donor-poisoned from the
broken eCPS layer flagged in #us-data; populace beats eCPS 4-14x on every
other thread metric — comparison in /tmp/itemized_compare.log).
## Source map (all primary, loaders exist in usdata-populace worktree):
1. RAW ASEC adds: spm_unit_pre_subsidy_childcare_expenses=SPM_CHILDCAREXPNS,
   spm_unit_capped_work_childcare_expenses=SPM_CAPWKCCXPNS (cps.py:1616) —
   SPM-record columns; carry via extra columns + spm-grain mapping.
2. PUF adds: investment_interest_expense (verify raw field E20300-class in
   puf_2015.csv; else usdata puf.py rule), anything else donor-block that PUF
   carries directly.
3. SCF stage (wealth block: net_worth, scf_*, bank/bond/stock assets,
   vehicles, auto loans): donor = Fed SCF via
   policyengine_us_data.datasets.scf (fed_scf.py loader), weighted QRF
   (microimpute, weight_col=SCF weights), predictors mirroring usdata's SCF
   imputation config; support-guard to SCF's OWN realized ranges.
4. ACS stage (first_home_mortgage_balance/interest/origination_year, rent):
   donor = census_acs loader; usdata's rent-imputation predictors.
5. CPS-ORG stage (hourly_wage, is_paid_hourly): datasets/org loader.
6. Prior-year income: usdata add_previous_year_income pattern (consecutive
   ASEC self-join) — port the join, primary.
7. ESI premiums: MEPS-IC parameters (usdata cps.py:204 block) applied to
   has_esi holders — parameter assignment, not a model.
8. tip_income: derive from is_tipped_occupation x tip share of earnings
   (usdata rule if present, else SIPP stage) — check usdata tip_income source.
## Execution: replace stage F2 (ecps_donor_impute) with stages F2a-F2e above;
## support guard generalized to per-donor ranges; delete eCPS baseline arg from
## the BUILD path (keep only in scoring); rebuild -> extract -> recalibrate ->
## all gates (parity 0, smoke incl. investment_interest sane, score train+
## holdout win) -> republish SAME filename new revision + new build-id ->
## update card (drop "eCPS as donor" language!), bundles PR #20, observatory,
## index evidence, memory. NAMING: no ordinals (policy set).
## Final source pins (verified): investment_interest_expense = residual of PUF
## interest paid (E19200/mortgage_interest_paid donor var we already impute)
## after the ACS mortgage split — port usdata utils/mortgage_interest.py
## (person-level residual at line ~154/320) inside the ACS stage. tip_income =
## SIPP model: `from policyengine_us_data.datasets.sipp import get_tip_model;
## model.predict(X_test=cps_frame, mean_quantile=0.5)` (cps.py:2828-2835 shows
## the exact call + required predictor columns upstream ~2815: pension/
## retirement/non_ssi_income aggregates). SCF loader: datasets/scf/fed_scf.py
## (.load() with download). ALL stages now have primary loaders in the
## usdata-populace worktree — no eCPS anywhere in the build.

## V3 PROGRESS @ 08:55: scripts/primary_source_impute.py v0 wired (SCF wealth
## block via SummarizedFedSCF + weighted microimpute + own-donor support guard;
## SIPP tips via get_tip_model; ORG wages via add_org_labor_market_inputs with
## a dict adapter). ecps_donor_impute REMOVED from the build path. STILL TO
## IMPLEMENT before parity passes: (a) ACS mortgage stage (census_acs loader +
## usdata utils/mortgage_interest.py person-residual rule -> first_home_
## mortgage_balance/interest/origination_year + investment_interest_expense);
## (b) prior-year income (PERIDNUM join vs prior ASEC: add PERIDNUM to loader
## raw columns, load asec_year-1 via load_cps_asec, map prior WSAL_VAL/
## SEMP_VAL -> employment/self_employment_income_last_year + previous_year_
## income_available; sentinels {-1,-9999}); (c) MEPS-IC ESI premiums (usdata
## cps.py ~line 195-230 parameter block: plan-type premiums applied to has_esi
## persons — copy the parameter table + rule); (d) childcare raw columns:
## SPM_CHILDCAREXPNS + SPM_CAPWKCCXPNS via extra_person_columns -> spm-grain
## (max per spm unit) at export (entity mover handles spm placement if set on
## hh or person? childcare is SPM entity — set person-level then entity mover
## needs spm handling for person-stored spm vars OR aggregate at export like
## v2 did from hh; simplest: person col -> groupby person_spm_unit_id max ->
## attach in unit_table spm block next to spm_unit_energy_subsidy precedent).
## Smoke-iterate after each addition: nohup .venv/bin/python -u
## scripts/build_us_candidate.py --mode smoke --usdata-repo
## ~/.claude-worktrees/usdata-populace > /tmp/v3_smoke.log. Donor downloads
## (SCF/ORG/SIPP model) happen on first run — may need network time. THEN full
## rebuild -> extract -> recalibrate -> gates -> republish per V3 section.

# ARCHITECTURE REVIEW (greenfield subagent, 2026-06-11 ~09:30) — ADOPTED DECISIONS
Full text delivered in-session. Ranked verdicts:
A1(L): build = 1,195-line imperative driver on legacy microplex; greenfield =
typed stage package populace.build.us over Frame, retire the spec DSL for the
US build ("sequencing step 5 is the whole job"). A2(M): the build uses legacy
microplex.units/microimpute while populace.frame.units/populace.fit (which fix
the exact weight_col bug by construction) sit unused — swap engines. B(M): one
declarative donor graph; heavy-tail fix = SIGNED calibration targets (net-STCG,
investment-interest, itemizer counts, charitable) appended to OUR TargetSet —
NOT range clips; ORG try/except silent fallback violates charter, make it fail.
C(M): replace the harness-extracted npz with a populace-owned versioned
TargetSpec registry (+SEs, sources); reverse build→scorer dependency. D(M):
kernel excellent/unused — needs policyengine_us RulesEngine adapter +
frame.place(column, entity); deletes ~150 driver lines. E(M): gate suite =
parity + support + AGGREGATE-VS-ADMIN (permanent, would have caught STCG +
investment-interest) + per-family + ROTATED holdout; FIX objective mismatch
(calibrator relative loss vs scorer absolute SSE). F: shipped artifact still
v2.2 eCPS-contaminated (v3 in flight); driver STILL opens baseline_h5 for the
v2 support-guard loop — DELETE IT in v3 (re-anchor PUF clips to PUF's own
ranges); calibrate solver DENSIFIES the sparse matrix (blocks 3M+ scaling —
fix with torch.sparse SpMM); add the charter-mandated pip-from-wheels CI job;
max_weight_ratio=50 into stage config + manifest.
## IMMEDIATE v3 adoptions (this build): (1) delete driver baseline_h5 guard
## loop, re-anchor PUF-imputed clips to PUF's own realized ranges (computed at
## the PUF stage); (2) append signed calibration targets to the TargetSet in
## /tmp/build_populace_us_v2.py: net STCG (SOI-scale small negative),
## investment_interest_expense total, from SOI-published values with sources;
## the scored comparison surface stays frozen (symmetric refit unaffected).
## DEFERRED to the re-base epic: stage package, donor graph, target registry,
## RulesEngine adapter, sparse solve, rotated holdout, pip-wheels CI.

## V3 @ 09:55: ALL SEVEN STAGES IMPLEMENTED. SCF/SIPP/ORG green (smoke 4);
## MEPS + prior-year (PERIDNUM join, downloads ASEC 2022 on first run) +
## mortgage conversion (SCF hints + PUF interest residual -> first/second_home_
## mortgage_*, home_mortgage_interest, investment_interest_expense) wired;
## childcare raw SPM columns at export. REVIEW ADOPTIONS DONE: support guard
## re-anchored to PUF's OWN uprated ranges (eCPS baseline now appears ONLY in
## the driver's scoring invocation — benchmark use); signed net-STCG target
## (PUF-weighted-uprated value, computed not typed) appended in
## /tmp/build_populace_us_v2.py. Smoke 5 running (/tmp/v3_smoke.log) covering
## MEPS/prior-year/mortgage for the first time — iterate on tracebacks
## (likely: converter dict KeyErrors -> add the key from person; prior-year
## ASEC 2022 download time; MEPS NOW_* column dtypes). Then full chain:
## rebuild -> extract -> calibrate (with STCG target) -> gates (parity 0,
## smoke investment_interest ~tens of B + STCG near PUF total, score) ->
## republish per V3 plan (card de-eCPS rewrite, diagram topology update,
## bundles PR #20, observatory dated release, #us-data comparison rerun).

## V3 PARITY ITERATION @ 10:35 (22 gaps -> fixes in flight):
FIXED this commit: (1) mortgage interest zero = missing person-grain
"deductible_mortgage_interest" key (getter never falls through to
interest_deduction; now provided = head-carried itd_person); (2) SCF raw names
(checking/saving/stocks/bond) mapped to PE bank/stock/bond_assets, head-carried
onto household-head persons (person-entity); net_worth computed on hh as
scf asset components - debt components (+ mapped assets); vehicles attempted
from raw names if present (verify in smoke — SCF column list had
total_vehicle_installments only; vehicles may need the full SCF loader vars or
stay a documented gap pending Fed SCF summary variable check).
REMAINING CLASS-2 SOURCES (pin): weeks_unemployed = raw weeks col (usdata
cps.py:1442 weeks_raw; find ASEC name e.g. WKSUNEM1, add to loader + derive
where -1 -> 0); other_health_insurance_premiums rule at usdata cps.py:829
(read block, port); pre_subsidy_rent = ACS rent imputation (usdata add_rent
cps.py:417 — port donor flow with microimpute or implement ACS donor directly;
LAST/heaviest); taxable_401k/403b/sep_distributions + QBI block (w2_wages,
unadjusted_basis, sstb_*, qualified_bdc/reit_and_ptp, partnership_se_income):
read usdata puf.py lines ~700-800 for the derivation rules (they are PUF
preprocessing outputs in the extended imputed list) and port into the
microplex puf.py preprocessing + PUF_IMPUTE_VARS like the v2.1 batch.
Then: smoke -> full rebuild -> extract -> calibrate -> parity MUST be 0 ->
smoke gate -> score -> republish (full protocol in prior wakeups).

## WORKSPACE CANONICALIZATION (2026-06-11, Max: "clean this up — there should not
## be multiple versions of anything") — one canonical name per thing, git is the
## versioning. Scripts now in-tree: scripts/build_dataset.py (was /tmp/build_populace_us_v2.py),
## scripts/extract_target_surface.py (was /tmp/extract_v2_surface.py); ecps_donor_impute.py
## DELETED (git history has it); this file renamed V2_PLAN.md -> PLAN.md. Artifacts:
## artifacts/{populace_us_2024.h5,populace_us_2024_timeperiod.h5,populace_us_2024_calibration.npz,
## target_surface_raw.npz} are THE names (currently the published build's bytes; the v3 chain
## overwrites in place), pools at spec_candidate_full_2024/ + spec_candidate_smoke_2024/ only —
## all iteration-suffixed files/dirs purged (~4GB; HF revisions hold shipped history).
## Score rig: ~/populace-score-work/{run_comparison.py,score_out,populace_us_2024_timeperiod.h5}.
## Smoke log: /tmp/populace_smoke.log. Bundles proposal moved to clean branch: PR #21
## (was #20; branch populace-us-2024-v2-proposal deleted). Standalone ~/PolicyEngine/populace-us
## scaffold deleted (monorepo build/us/ holds newer copies, diff-verified). Observatory builder
## points at canonical paths + primary_source_impute.py.
## FORWARD PROTOCOL (v3 chain): smoke green -> full build (build_us_candidate.py --mode full)
## -> scripts/extract_target_surface.py -> scripts/build_dataset.py -> parity MUST be 0 ->
## smoke gate -> score (run_comparison.py, beats 1.0888/0.3167 matched) -> republish.

## PAUSE STATE (2026-06-11 ~12:10, Claude restart): smoke 15 RUNNING detached
## (pid ppid=1, log /tmp/populace_smoke.log) with the FULL 35-gap parity sprint
## wired (ASEC derives, head-carry loud, SIPP vehicles, AOTC, SSTB persist,
## weeks_worked contract fix, person->spm moves). ON RESUME: re-arm watcher on
## the smoke; green -> full build -> scripts/extract_target_surface.py ->
## scripts/build_dataset.py (calibrate venv /tmp/populace-build-venv) ->
## scripts/enrich_artifact.py (worktree .venv, sqlmodel installed) ->
## scripts/check_parity.py (build venv; expect 0 gaps) -> score -> republish.
## Epic branch populace-rebase: all six commits + review fixes pushed; reviewer
## subagent a66cbd5268751880b was mid-re-verify (suite green) — re-verify, then
## PR to PolicyEngine/populace on green CI.

## EPIC MERGED (2026-06-11 ~13:20): populace PR #5 squashed to main (a542524) —
## frame.place, sparse solve + options record, TargetSpec registry,
## gates (parity/support/aggregate-vs-admin/per-family/EXPORTED-NONZERO) +
## rotated holdout, StagePlan + us_plan donor graph, wheels CI (3.13+3.14).
## Two review rounds; 1 actionable fixed. /tmp/populace-build-venv editable
## installs re-pointed at ~/PolicyEngine/populace main; rebase worktree gone.
## check_parity.py gate 0 = exported_nonzero (chain fails on any all-zero
## stored column); build_dataset.py drops all-zero cols with zero engine
## defaults. US implementations port = task #26, AFTER republish.
