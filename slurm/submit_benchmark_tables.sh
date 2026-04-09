#!/bin/bash
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SLURM_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
mkdir -p "${REPO_ROOT}/logs/benchmarks"

echo "Submitting block_assign benchmark array..."
sbatch "${SLURM_DIR}/benchmark_block_assign.sbatch"

echo "Submitting block_seq benchmark array..."
sbatch "${SLURM_DIR}/benchmark_block_seq.sbatch"
