# Microplex US Pack

This directory is the US content pack for Microplex. It ships declarative specs
and manifests only; Microplex owns the execution engine, microimpute owns donor
imputation, and microcalibrate owns calibration.

## Contents

- `specs/us-2024.yaml`: US 2024 construction spec.
- `manifests/ecps_export_contract.json`: frozen eCPS export column contract.
- `manifests/frozen_production_ecps_2024_benchmark_manifest.json`: pinned
  production-eCPS benchmark certificate metadata.
- `manifests/pe_source_impute_blocks.json`: source-imputation block
  declarations.
- `manifests/arch_targets.json`: declarative Arch target mapping and provider
  configuration hooks.
- `manifests/puf.json`: PUF source manifest.

## Construction Order

1. Load ASEC/CPS and PUF sources.
2. Build the seeded 50/50 ASEC+PUF support spine.
3. Assign atomic census geography within the lowest available CPS geography.
4. Run SCF, SIPP, and ACS source imputations on the resolved support universe.
5. Apply declared transforms and target construction.
6. Calibrate through Microplex's microcalibrate adapter.
7. Export the PolicyEngine-compatible dataset.

## Validation

The generic Microplex content-package check validates that the spec loads, the
variable manifest covers the frozen eCPS contract plus declared imputation
surface, and the pack contains no runtime Python files.
