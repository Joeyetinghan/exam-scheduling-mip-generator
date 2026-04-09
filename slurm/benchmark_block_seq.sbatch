#!/bin/bash
#SBATCH --account=gts-phentenryck3-coda20
#SBATCH --job-name=block_seq_bench
#SBATCH --output=logs/benchmarks/block_seq_%A_%a.out
#SBATCH --array=0-7
#SBATCH -N1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=02:00:00
#SBATCH -q embers
#SBATCH --gres-flags=enforce-binding

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SLURM_DIR="${REPO_ROOT}/slurm"
source "${SLURM_DIR}/common_env.sh"

CASE_FILE="${CASE_FILE:-${SLURM_DIR}/block_seq_benchmark_cases.csv}"
CASE_LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${CASE_FILE}")"
IFS=, read -r SIZE NUM_BLOCKS NUM_SLOTS CATEGORY <<< "${CASE_LINE}"

STATS_PATH="logs/benchmarks/block_seq_stats_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

RUN_LOG="logs/benchmarks/block_seq_cmd_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
TIME_FILE="$(mktemp)"

set +e
/usr/bin/time -f "%e" -o "${TIME_FILE}" \
  python block_seq/block_seq_generator.py \
    --num-blocks "${NUM_BLOCKS}" \
    --num-slots "${NUM_SLOTS}" \
    --size "${SIZE}" \
    --seed 42 \
    --stats-path "${STATS_PATH}" \
    --threads 1 \
    --time-limit 3600 \
    --no-frontload \
    --optimize \
  > "${RUN_LOG}" 2>&1
EXIT_CODE=$?
set -e

RUNTIME_SEC="$(
  awk 'match($0, /^[0-9]+(\.[0-9]+)?$/) { value=$0 } END { print value }' "${TIME_FILE}"
)"
rm -f "${TIME_FILE}"
GAP_PCT="$(
  awk '/Best objective/ { value=$NF; gsub(/%/, "", value) } END { print value }' "${RUN_LOG}"
)"

VAR_BIN=""
VAR_INT=""
VAR_CONT=""
CONSTRAINTS=""
if [[ -f "${STATS_PATH}" ]]; then
  IFS=, read -r _GENERATOR VAR_BIN VAR_INT VAR_CONT CONSTRAINTS < <(tail -n 1 "${STATS_PATH}")
fi

RESULTS_CSV="logs/benchmarks/block_seq_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}.csv"
STATUS="ok"
if rg -q "Time limit reached" "${RUN_LOG}"; then
  STATUS="time_limit"
elif [[ "${EXIT_CODE}" -ne 0 ]]; then
  STATUS="failed"
fi

{
  flock 9
  if [[ ! -f "${RESULTS_CSV}" ]]; then
    printf 'job_id,array_task,size,num_blocks,num_slots,category,seed,runtime_sec,gap_pct,exit_code,status,var_bin,var_int,var_cont,constraints,run_log\n' > "${RESULTS_CSV}"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" \
    "${SLURM_ARRAY_TASK_ID}" \
    "${SIZE}" \
    "${NUM_BLOCKS}" \
    "${NUM_SLOTS}" \
    "${CATEGORY}" \
    "42" \
    "${RUNTIME_SEC}" \
    "${GAP_PCT}" \
    "${EXIT_CODE}" \
    "${STATUS}" \
    "${VAR_BIN}" \
    "${VAR_INT}" \
    "${VAR_CONT}" \
    "${CONSTRAINTS}" \
    "${RUN_LOG}" >> "${RESULTS_CSV}"
} 9>>"${RESULTS_CSV}.lock"

echo "Benchmark row appended to ${RESULTS_CSV}"
if [[ "${EXIT_CODE}" -ne 0 ]]; then
  exit "${EXIT_CODE}"
fi
