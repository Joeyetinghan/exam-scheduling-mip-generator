# Problem Description

**Task.** Final exam scheduling for a large university over a 7–9 day exam window with three daily slots (e.g., 9am, 2pm, 7pm). The goal is to assign each exam to **exactly one** time slot while minimizing student burden and operational risk.

**Inputs.**
- **Exam list** with section enrollments (exam sizes).
- **Co-enrollment data** at the **pair** and **triplet** levels (number of students taking two/three exams together).
- **Available time slots** (some slots may be excluded, e.g., Sat night, Sun morning).
- (Optional) **Policy levers/constraints** provided by the registrar (e.g., *front-load large exams*, upper bounds on specific slots).  
  *Room assignments are handled separately by the registrar and are not modeled here.*

**Undesirable events (metrics).** We quantify schedule “discomfort” by counting:
- **Direct conflicts:** a student has two exams at the same time.
- **Back-to-back (B2B):** a student has exams in consecutive slots.
- **Two-in-24hr (2-in-24):** a student has two exams within any three consecutive slots.
- **Three-in-a-row (Triple):** a student has exams in three consecutive slots.
- **Three-in-four (3-in-4):** a student has three exams within any four consecutive slots.  
  (No double counting across metrics; e.g., pairs inside a Triple are not also counted as B2B/2-in-24.)

**Objective.** Minimize a **weighted sum** of the above undesirable events, with higher priority typically given to eliminating **direct conflicts** and **triples**, while also reducing B2B, 2-in-24, and 3-in-4 where possible.

**Hard constraints.**
- Each exam is assigned to exactly one time slot.
- Time slot availability and registrar exclusions are respected.
- (Optional) **Front-loading**: exams exceeding a chosen enrollment threshold must be scheduled before a chosen early-slot cutoff.

**Outputs.**
- A mapping from exams to time slots.
- Schedule-quality metrics for the five events above, plus summary distributions useful for registrar review and communication.

**Modeling/Algorithms.**
- **Group-then-Sequence (GtS):** (i) *Block Assignment* clusters exams to blocks by reducing direct conflicts; (ii) *Block Sequencing* assigns blocks to time slots to optimize the weighted metrics;

**Data format in this repo.**
- `inputs/anon_coenrol.csv`, `inputs/hist_totals.csv`, `inputs/anon24.csv`, `inputs/anon_t_co.csv` (see module READMEs below).
- Generators produce `.lp` models for reproducibility and solver-side inspection.

**Reference.**  
This problem statement follows the formulation and evaluation in our paper:  
*Cornell University Uses Integer Programming to Optimize Final Exam Scheduling* ([arXiv:2409.04959](https://arxiv.org/abs/2409.04959)). See §2 *Problem Description*, §4 *Modeling Approach*, and the **Appendix** for the full MIP formulations (variables, constraints, objectives, and parameter definitions).

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
time python block_assign/block_assign_generator.py --seed 42 --size 200 --optimize
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
These are runs on a personal laptop using gurobipy and python 3.11.  You can measure runtime in bash, e.g.:

```bash
time python block_seq/block_seq_generator.py --seed 42 --slots 10 --size 200 --optimize
```


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



