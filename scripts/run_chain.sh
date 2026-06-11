#!/bin/zsh
# The v3 build chain, end to end. Each step's venv is the one that owns its
# dependencies (worktree .venv: engine + usdata; populace venv: calibrate +
# gates). Fails fast; the log narrates which step died.
set -e
WT=~/.claude-worktrees/microplex-spec-build
# One chain = one run id = one fresh log; never inherit a prior chain's.
rm -f /tmp/populace_run_id
echo $$ > /tmp/populace_chain.pid
trap 'rm -f /tmp/populace_chain.pid' EXIT
BUILD_PY=$WT/.venv/bin/python
POP_PY=/tmp/populace-build-venv/bin/python

echo "=== CHAIN step 1: full build ==="
$BUILD_PY -u $WT/scripts/build_us_candidate.py --mode full \
  --usdata-repo ~/.claude-worktrees/usdata-populace

echo "=== CHAIN step 2: extract target surface ==="
$BUILD_PY -u $WT/scripts/extract_target_surface.py

echo "=== CHAIN step 3: calibrate + artifact ==="
$POP_PY -u $WT/scripts/build_dataset.py

echo "=== CHAIN step 4: enrich (sim-dependent layers) ==="
$BUILD_PY -u $WT/scripts/enrich_artifact.py

echo "=== CHAIN step 5: gates (exported-nonzero + parity + smoke) ==="
$POP_PY -u $WT/scripts/check_parity.py

echo "=== CHAIN COMPLETE: all gates green ==="
