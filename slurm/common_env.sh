#!/bin/bash
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SLURM_DIR}/.." && pwd)"

module load mamba
mamba activate exam-sched
module load gurobi

# Keep the module-provided license environment, but use the conda env's gurobipy.
if [[ -n "${PYTHONPATH:-}" ]]; then
  cleaned_pythonpath="$(
    printf '%s' "$PYTHONPATH" \
      | tr ':' '\n' \
      | grep -v '/usr/local/pace-apps/manual/packages/gurobi/13\.0\.1/lib/python3\.11/site-packages' \
      || true
  )"
  cleaned_pythonpath="$(printf '%s\n' "$cleaned_pythonpath" | paste -sd: -)"
  if [[ -n "$cleaned_pythonpath" ]]; then
    export PYTHONPATH="$cleaned_pythonpath"
  else
    unset PYTHONPATH
  fi
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${REPO_ROOT}/logs/benchmarks"
cd "${REPO_ROOT}"
