import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/CosilicoAI/microplex-us/src"))
from microplex_us.pipelines.ecps_replacement_comparison import _extract_pe_native_loss_inputs
from pathlib import Path
ART = Path.home()/".claude-worktrees"/"microplex-spec-build"/"artifacts"
cand = ART/"spec_candidate_full_2024"/"candidate_timeperiod.h5"
repo = Path.home()/".claude-worktrees"/"usdata-populace"
print("extracting v2 raw surface...", flush=True)
inp = _extract_pe_native_loss_inputs(
    input_dataset_path=str(cand), period=2024,
    policyengine_us_data_repo=repo, policyengine_us_data_python=repo/".venv/bin/python",
    skip_tax_expenditure_targets=False,
)
A_s = np.asarray(inp["scaled_matrix"], np.float64)
b_s = np.asarray(inp["scaled_target"], np.float64)
w0 = np.asarray(inp["initial_weights"], np.float64)
b_raw = np.asarray(inp["unscaled_target"], np.float64)
scaling = np.asarray(inp["scaling"], np.float64)
names = [str(x) for x in inp["metadata"]["target_names"]]
A_raw = A_s / scaling[None, :]
est = A_raw.T @ w0
print(f"targets {len(names)} | b median {np.median(b_raw):.3g} | est/b median {np.median((est+1)/(b_raw+1)):.3f}", flush=True)
out = ART/"target_surface_raw.npz"
np.savez_compressed(out, A=A_raw, b=b_raw, w0=w0, names=np.array(names, dtype=object))
loss = float(np.mean(((A_raw.T@w0 - b_raw)/(b_raw + 1.0))**2))
print(f"SAVED {out} | raw initial loss {loss:.4f}", flush=True)
