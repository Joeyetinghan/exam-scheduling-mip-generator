# Exam Scheduling MIP Generator

Open-source instance generators for the Cornell final exam scheduling models in:

```bibtex
@article{ye2025cornell,
  title={Cornell university uses integer programming to optimize final exam scheduling},
  author={Ye, Tinghan and Jovine, Adam S and Van Osselaer, Willem and Zhu, Qihan and Shmoys, David B},
  journal={INFORMS Journal on Applied Analytics},
  year={2025},
  publisher={INFORMS}
}
```

The repo generates MIP instances for two stages:

- `block_assign`: assign exams to blocks from pairwise co-enrollment data
- `block_seq`: assign blocks to slots from block-level pair and triplet data

It also includes a synthetic data generator for larger instances.

## Inputs

Shared inputs live under `inputs/`:

- `anon_coenrol.csv`: pairwise co-enrollment matrix
- `anon_t_co.csv`: triplet co-enrollment data
- `exam_sizes.csv`: exam enrollments
- `generated/`: cached synthetic bundles created on demand

If `--size` exceeds the available exam count in the selected bundle, the repo generates and caches a synthetic bundle under:

```text
inputs/generated/synth_n{SIZE}_seed{SEED}/
```

## Setup

```bash
conda env create -f environment.yml
conda activate exam-sched
```

Gurobi must be available in the shell environment before running solver-backed commands.

## block_assign

Generate a block-assignment LP directly from the pairwise matrix:

```bash
python block_assign/block_assign_generator.py [--size SIZE] [--seed SEED] [--inputs-dir INPUTS_DIR]
```

Example:

```bash
python block_assign/block_assign_generator.py --size 200 --seed 42
```

This writes:

- `outputs/blockassign_n{SIZE}_seed{SEED}.lp`

Benchmark table:

| Size | Blocks | Runtime (s) | Gap (%) | Category  | # Var B | # Var I | # Var C | # Constr |
| ---- | ------ | ----------- | ------- | --------- | ------: | ------: | ------: | -------: |
| 200  | 24     | 100.94      | 0.00    | medium    |    4800 |       0 |       0 |      200 |
| 300  | 24     | 324.24      | 0.00    | medium    |    7200 |       0 |       0 |      300 |
| 400  | 24     | 706.09      | 0.00    | medium    |    9600 |       0 |       0 |      400 |
| 500  | 24     | 995.88      | 0.00    | medium    |   12000 |       0 |       0 |      500 |
| 600  | 24     | 3600 (time limit) | 100.00  | ext hard  |   14400 |       0 |       0 |      600 |
| 700  | 24     | 3600 (time limit) | 100.00  | ext hard  |   16800 |       0 |       0 |      700 |

## block_seq

`block_seq` first constructs a real block assignment, then builds the sequencing MIP. It supports:

- exact `--num-blocks`
- exact `--num-slots`
- optional front-loading
- virtual blocks when `num_blocks < num_slots`

Run the full sequencing generator with:

```bash
python block_seq/block_seq_generator.py \
  [--num-blocks NUM_BLOCKS] \
  [--num-slots NUM_SLOTS] \
  [--size SIZE] \
  [--seed SEED] \
  [--inputs-dir INPUTS_DIR] \
  [--no-frontload] \
  [--frontload-block-size-cutoff CUTOFF] \
  [--frontload-slot-cutoff SLOT_CUTOFF] \
  [--optimize]
```

Example:

```bash
python block_seq/block_seq_generator.py \
  --num-blocks 22 \
  --num-slots 24 \
  --size 200 \
  --seed 42 \
  --optimize
```

Defaults:

- `--num-blocks 24`
- `--num-slots 24`
- `--seed 3`
- front-loading enabled with:
  - `--frontload-block-size-cutoff 300`
  - `--frontload-slot-cutoff 21`

For block-assignment-only preprocessing, run:

```bash
python block_seq/greedy_block_assignment.py [--num-blocks NUM_BLOCKS] [--size SIZE] [--seed SEED] [--inputs-dir INPUTS_DIR]
```

Each `block_seq` run writes an instance folder:

```text
block_seq/outputs/blockseq_n{N}_blocks{B}_slots{S}_seed{SEED}/
```

Contents:

```text
blockmap.csv
block_summary.csv
instance.json
pair_counts.csv
triplet_counts.csv
model.lp
slot_summary.csv   # only when --optimize is used
```

`instance.json` plus the pair/triplet CSVs are the minimal exported data needed to rebuild the sequencing MIP.

Benchmark table:

| Size | Blocks | Slots | Runtime (s) | Gap (%) | Category  | # Var B | # Var I | # Var C | # Constr |
| ---- | ------ | ----- | ----------- | ------- | --------- | ------: | ------: | ------: | -------: |
| 60   | 6      | 6     | 3.72        | 0.00    | easy      |    2808 |       0 |       0 |     2400 |
| 80   | 7      | 7     | 16.00       | 0.00    | easy      |    5145 |       0 |       0 |     4144 |
| 100  | 8      | 8     | 167.31      | 0.00    | medium    |    8704 |       0 |       0 |     6688 |
| 150  | 10     | 10    | 3600 (time limit) | 14.35   | ext hard  |   21000 |       0 |       0 |    15040 |
| 200  | 12     | 12    | 3600 (time limit) | 20.52   | ext hard  |   43200 |       0 |       0 |    29424 |
| 250  | 14     | 14    | 3600 (time limit) | 21.70   | ext hard  |   79576 |       0 |       0 |    52192 |
| 300  | 16     | 16    | 3600 (time limit) | 24.18   | ext hard  |  135168 |       0 |       0 |    86080 |
| 350  | 18     | 18    | 3600 (time limit) | 29.41   | ext hard  |  215784 |       0 |       0 |   134208 |
| 450  | 20     | 20    | 3600 (time limit) | 32.89   | ext hard  |  328000 |       0 |       0 |   200080 |
| 550  | 22     | 22    | 3600 (time limit) | 33.47   | ext hard  |  479160 |       0 |       0 |   287584 |

The `block_seq` benchmark runs used `--no-frontload`. The variable counts are unchanged with front-loading, but the listed constraint counts are exact for the no-frontload benchmark regime.

Rows shown as `3600 (time limit)` reached the 1-hour solver limit; the gap column reports the final gap at that limit.

Category labels are aligned to the Distributional MIPLIB hardness thresholds: solved under `100s` = `easy`, `100-1000s` = `medium`, solved over `1000s` = `hard`, and time-limited rows are labeled `very hard` or `ext hard` by final gap. Distributional MIPLIB defines these at the distribution level; here they are used as concise instance-level labels for the benchmark rows.

## synthetic_dataset

Generate a larger synthetic input bundle directly:

```bash
python synthetic_dataset.py --size SIZE --seed SEED [--inputs-dir INPUTS_DIR]
```

This writes:

- `exam_sizes.csv`
- `anon_coenrol.csv`
- `anon_t_co.csv`
- `metadata.json`

## Benchmark Config

The benchmark runs use:

- 1 Gurobi thread
- `--time-limit 3600`
- `5G` memory by default
- `15G` for the hard `block_assign` cases at sizes `600` and `700`

## Repository Layout

```text
block_assign/
block_seq/
inputs/
synthetic_dataset.py
environment.yml
```
