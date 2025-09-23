# block_assign (block assignment MIP instance)

This module generates a block-assignment MIP instance (pairwise co-enrollment driven).

## Prerequisites

* Python 3.7+
* `pandas`
* `numpy`
* `gurobipy` (with a valid license)

## Installation

```bash
# (Optionally) create a virtual env
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install pandas numpy gurobipy
```

## Usage

```bash
python block_assign/block_assign_generator.py  [--seed SEED] [--size SIZE]
```


**Example:**
```bash
python block_assign/block_assign_generator.py --seed 42 --size 200
```

This will:

1. Read `inputs/anon_coenrol.csv` & `inputs/hist_totals.csv`.
2. Sample pairwise co-enrollments for the top-20 (or random extra) course IDs.
3. Build and write the LP model to `outputs/blockassign_n20_seed42.lp` (with your chosen seed).
4. Print a stats summary, e.g.:

```
Done. LP ⇒ outputs/blockassign_n20_seed42.lp, stats: {
  "samples": 80,
  "weighted": 56,
  "random": 24,
  "unique": 78,
  "max": 5
}
```

---

## Runtime Benchmark Table

Use the table below to record how long the block assignment generator takes for different `size` values. You can measure runtime in bash, e.g.:

```bash
time python block_assign/block_assign_generator.py 50
```

| Size  | Runtime (s) | Category |
| :--:  | :---------: | :------: |
|  200  |    438      | easy     |
|  300  |    523      | easy     |
|  400  |    2207     | medium   |
|  500  |    10829    | hard     |
|  600  |    32256    | hard     |




## Files

Structure:

block_assign/
- outputs/
- block_assign_generator.py

Note: `setup.py` and its data-generation pipeline were removed for open-sourcing. Bring your own anonymized CSVs listed above.

# block_seq (block sequencing MIP instance)

Run the simulation & optimization in one command:

```bash
python block_seq/block_seq_generator.py [--seed SEED] [--slots SLOTS] [--size SIZE]
```

* `<size>`: Number of (anonymous) courses to simulate (e.g. `300`). Increasing this increases the model size polynomially, so don't go above 1000 unless you have a lot of RAM.
* `--seed SEED`: Random seed for reproducibility (default: `3`).
* `--slots SLOTS`: Number of time slots (blocks) to use (default: `24`). This must be more than 24.


**Example:**

```bash
python block_seq/block_seq_generator.py --seed 42 --slots 10 --size 200
```

Output LP is named `outputs/blockseq_n<size}_seed{seed}.lp`.

Structure:

block_seq/
- outputs/
- block_seq_generator.py

Root-level inputs/ directory (shared by both modules):

inputs/
- anon_coenrol.csv
- hist_totals.csv
- anon24.csv
- anon_t_co.csv

## Citation
If you use this repository, please cite the following paper:

Cornell University Uses Integer Programming to Optimize Final Exam Scheduling. Authors: Tinghan Ye, Adam Jovine, Willem van Osselaer, Qihan Zhu, David Shmoys. arXiv: [arXiv:2409.04959](https://arxiv.org/abs/2409.04959).

## Benchmark Table
These are runs on a personal laptop using gurobipy and python 3.11. 

| Size | Slots | Runtime (s) | Category |
| ---- | ----- | ----------- | -------- |
| 200  | 10    | 25          | easy     |
| 400  | 10    | 54          | easy     |
| 200  | 11    | 51          | easy     |
| 400  | 11    | 267         | easy     |
| 200  | 12    | 2384        | medium   |
| 400  | 12    | 3001        | medium   |
| 200  | 13    | 2102        | medium   |
| 400  | 13    | 1824        | medium   |
| 200  | 14    | 2950        | medium   |
| 400  | 14    | 5322        | hard     |
| 200  | 15    | 10651       | hard     |
| 400  | 15    | 5+ hours    | hard     |
| 200  | 20    | 5+ hours    | hard     |
| 400  | 20    | 5+ hours    | hard     |



