import argparse
import csv
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple

import gurobipy as gp
import pandas as pd
from gurobipy import GRB


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
INPUTS_DIR = ROOT_DIR / "inputs"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from block_seq.greedy_block_assignment import (
    build_output_dir,
    prepare_block_assignment,
    write_block_assignment_outputs,
)

DEFAULT_ALPHA = 10
DEFAULT_BETA = 10
DEFAULT_GAMMA1 = 1
DEFAULT_GAMMA2 = 1
DEFAULT_DELTA = 5
DEFAULT_TIME_LIMIT = 1500
DEFAULT_NUM_BLOCKS = 24
DEFAULT_NUM_SLOTS = 24
DEFAULT_FRONTLOAD_BLOCK_SIZE_CUTOFF = 300
DEFAULT_FRONTLOAD_SLOT_CUTOFF = 21
DEFAULT_SEED = 3


@dataclass(frozen=True)
class PreparedSequencingInstance:
    exam_ids: list[int]
    greedy_exam_to_block: dict[int, int]
    exam_to_block: dict[int, int]
    greedy_block_count: int
    real_blocks: list[int]
    virtual_blocks: list[int]
    all_blocks: list[int]
    pair_counts: dict[tuple[int, int], int]
    triplet_counts: dict[tuple[int, int, int], int]
    block_enrollment: dict[int, int]
    block_exam_counts: dict[int, int]
    within_block_conflicts: int
    large_blocks: set[int]
    early_slots: list[int]
    triple_day_start: list[int]
    triple_24_start: list[int]
    eve_morn_start: list[int]
    other_b2b_start: list[int]


@dataclass(frozen=True)
class BlockSequencingModelData:
    all_blocks: list[int]
    virtual_blocks: list[int]
    pair_counts: dict[tuple[int, int], int]
    triplet_counts: dict[tuple[int, int, int], int]
    large_blocks: set[int]
    early_slots: list[int]
    triple_day_start: list[int]
    triple_24_start: list[int]
    eve_morn_start: list[int]
    other_b2b_start: list[int]


def compute_slot_categories(
    num_slots: int, slots_per_day: int = 3
) -> tuple[list[int], list[int], list[int], list[int]]:
    if num_slots <= 0:
        raise ValueError("num_slots must be positive.")

    triple_day_start: list[int] = []
    triple_24_start: list[int] = []
    eve_morn_start: list[int] = []
    other_b2b_start: list[int] = []

    for slot in range(1, num_slots):
        if slot % slots_per_day == 0:
            eve_morn_start.append(slot)
        else:
            other_b2b_start.append(slot)

    for slot in range(1, num_slots - 1):
        if slot % slots_per_day == 1:
            triple_day_start.append(slot)
        else:
            triple_24_start.append(slot)

    return triple_day_start, triple_24_start, eve_morn_start, other_b2b_start


def build_pair_counts(
    pair_df: pd.DataFrame, exam_to_block: dict[int, int], all_blocks: list[int]
) -> dict[tuple[int, int], int]:
    pair_counts = {(i, j): 0 for i in all_blocks for j in all_blocks}
    for exam_i in pair_df.index:
        block_i = exam_to_block[int(exam_i)]
        for exam_j, count in pair_df.loc[exam_i].items():
            block_j = exam_to_block[int(exam_j)]
            pair_counts[(block_i, block_j)] += int(count)
    return pair_counts


def load_canonical_triplets(
    selected_exam_ids: set[int], inputs_dir: Path = INPUTS_DIR
) -> dict[tuple[int, int, int], int]:
    canonical_triplets: dict[tuple[int, int, int], int] = {}
    with open(inputs_dir / "anon_t_co.csv", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            triple = tuple(
                sorted(
                    (
                        int(row["a"]) + 1,
                        int(row["b"]) + 1,
                        int(row["c"]) + 1,
                    )
                )
            )
            if not all(exam_id in selected_exam_ids for exam_id in triple):
                continue

            coenrollment = int(row["co"])
            previous = canonical_triplets.get(triple)
            if previous is None:
                canonical_triplets[triple] = coenrollment
            elif previous != coenrollment:
                raise ValueError(
                    "anon_t_co.csv contains inconsistent co-enrollment counts for "
                    f"the same canonical triplet {triple}."
                )
    return canonical_triplets


def build_triplet_counts(
    canonical_triplets: dict[tuple[int, int, int], int],
    exam_to_block: dict[int, int],
    all_blocks: list[int],
) -> dict[tuple[int, int, int], int]:
    triplet_counts = {
        (i, j, k): 0 for i in all_blocks for j in all_blocks for k in all_blocks
    }

    for exam_triplet, count in canonical_triplets.items():
        block_triplet = tuple(exam_to_block[exam_id] for exam_id in exam_triplet)
        for permutation in set(itertools.permutations(block_triplet)):
            triplet_counts[permutation] += count

    return triplet_counts


def compute_block_enrollment(
    exam_to_block: dict[int, int],
    exam_sizes: pd.Series,
    real_blocks: list[int],
    virtual_blocks: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    block_enrollment = {block: 0 for block in real_blocks + virtual_blocks}
    block_exam_counts = {block: 0 for block in real_blocks + virtual_blocks}

    for exam_id, block_id in exam_to_block.items():
        block_enrollment[block_id] += int(exam_sizes.at[exam_id])
        block_exam_counts[block_id] += 1

    return block_enrollment, block_exam_counts


def compute_large_blocks(
    exam_to_block: dict[int, int],
    exam_sizes: pd.Series,
    real_blocks: list[int],
    cutoff: int,
) -> set[int]:
    large_blocks: set[int] = set()
    for exam_id, block_id in exam_to_block.items():
        if block_id in real_blocks and int(exam_sizes.at[exam_id]) > cutoff:
            large_blocks.add(block_id)
    return large_blocks


def prepare_block_sequencing_instance(
    num_blocks: int,
    num_slots: int,
    size: Optional[int] = None,
    frontload_block_size_cutoff: Optional[int] = None,
    frontload_slot_cutoff: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    inputs_dir: Path = INPUTS_DIR,
) -> PreparedSequencingInstance:
    if num_blocks <= 0:
        raise ValueError("--num-blocks must be positive.")
    if num_slots <= 0:
        raise ValueError("--num-slots must be positive.")
    if num_blocks > num_slots:
        raise ValueError("--num-blocks must be less than or equal to --num-slots.")

    if (frontload_block_size_cutoff is None) != (frontload_slot_cutoff is None):
        raise ValueError(
            "Front-loading requires both --frontload-block-size-cutoff and "
            "--frontload-slot-cutoff."
        )
    if frontload_block_size_cutoff is not None and frontload_block_size_cutoff < 0:
        raise ValueError("--frontload-block-size-cutoff must be nonnegative.")
    if frontload_slot_cutoff is not None and frontload_slot_cutoff <= 0:
        raise ValueError("--frontload-slot-cutoff must be positive when provided.")

    assignment = prepare_block_assignment(
        num_blocks=num_blocks,
        size=size,
        seed=seed,
        inputs_dir=inputs_dir,
    )
    resolved_inputs_dir = assignment.resolved_inputs_dir
    pair_df = assignment.pair_df
    exam_ids = assignment.exam_ids
    exam_sizes = assignment.exam_sizes
    greedy_exam_to_block = assignment.greedy_exam_to_block
    exam_to_block = assignment.exam_to_block
    greedy_block_count = assignment.greedy_block_count

    real_block_count = len(set(exam_to_block.values()))
    real_blocks = list(range(1, real_block_count + 1))
    virtual_blocks = list(range(real_block_count + 1, num_slots + 1))
    all_blocks = real_blocks + virtual_blocks

    canonical_triplets = load_canonical_triplets(
        set(exam_ids),
        inputs_dir=resolved_inputs_dir,
    )
    pair_counts = build_pair_counts(pair_df, exam_to_block, all_blocks)
    triplet_counts = build_triplet_counts(canonical_triplets, exam_to_block, all_blocks)

    block_enrollment, block_exam_counts = compute_block_enrollment(
        exam_to_block=exam_to_block,
        exam_sizes=exam_sizes,
        real_blocks=real_blocks,
        virtual_blocks=virtual_blocks,
    )

    triple_day_start, triple_24_start, eve_morn_start, other_b2b_start = (
        compute_slot_categories(num_slots)
    )

    early_slots: list[int] = []
    large_blocks: set[int] = set()
    if frontload_block_size_cutoff is not None and frontload_slot_cutoff is not None:
        effective_slot_cutoff = min(frontload_slot_cutoff, num_slots)
        early_slots = list(range(1, effective_slot_cutoff + 1))
        large_blocks = compute_large_blocks(
            exam_to_block=exam_to_block,
            exam_sizes=exam_sizes,
            real_blocks=real_blocks,
            cutoff=frontload_block_size_cutoff,
        )

    return PreparedSequencingInstance(
        exam_ids=exam_ids,
        greedy_exam_to_block=greedy_exam_to_block,
        exam_to_block=exam_to_block,
        greedy_block_count=greedy_block_count,
        real_blocks=real_blocks,
        virtual_blocks=virtual_blocks,
        all_blocks=all_blocks,
        pair_counts=pair_counts,
        triplet_counts=triplet_counts,
        block_enrollment=block_enrollment,
        block_exam_counts=block_exam_counts,
        within_block_conflicts=assignment.within_block_conflicts,
        large_blocks=large_blocks,
        early_slots=early_slots,
        triple_day_start=triple_day_start,
        triple_24_start=triple_24_start,
        eve_morn_start=eve_morn_start,
        other_b2b_start=other_b2b_start,
    )


def build_block_summary(instance: PreparedSequencingInstance) -> pd.DataFrame:
    rows = []
    for block_id in instance.all_blocks:
        rows.append(
            {
                "block": block_id,
                "is_virtual": block_id in instance.virtual_blocks,
                "num_exams": instance.block_exam_counts[block_id],
                "block_enrollment": instance.block_enrollment[block_id],
                "is_large_block": block_id in instance.large_blocks,
            }
        )
    return pd.DataFrame(rows)


def extract_model_data(instance: PreparedSequencingInstance) -> BlockSequencingModelData:
    return BlockSequencingModelData(
        all_blocks=list(instance.all_blocks),
        virtual_blocks=list(instance.virtual_blocks),
        pair_counts=dict(instance.pair_counts),
        triplet_counts=dict(instance.triplet_counts),
        large_blocks=set(instance.large_blocks),
        early_slots=list(instance.early_slots),
        triple_day_start=list(instance.triple_day_start),
        triple_24_start=list(instance.triple_24_start),
        eve_morn_start=list(instance.eve_morn_start),
        other_b2b_start=list(instance.other_b2b_start),
    )


def write_model_data_outputs(
    model_data: BlockSequencingModelData, output_dir: Path
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "instance.json"
    manifest = {
        "all_blocks": model_data.all_blocks,
        "virtual_blocks": model_data.virtual_blocks,
        "large_blocks": sorted(model_data.large_blocks),
        "early_slots": model_data.early_slots,
        "triple_day_start": model_data.triple_day_start,
        "triple_24_start": model_data.triple_24_start,
        "eve_morn_start": model_data.eve_morn_start,
        "other_b2b_start": model_data.other_b2b_start,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    pair_counts_path = output_dir / "pair_counts.csv"
    pair_rows = [
        {"block_i": block_i, "block_j": block_j, "count": count}
        for (block_i, block_j), count in sorted(model_data.pair_counts.items())
        if count != 0
    ]
    pd.DataFrame(pair_rows, columns=["block_i", "block_j", "count"]).to_csv(
        pair_counts_path, index=False
    )

    triplet_counts_path = output_dir / "triplet_counts.csv"
    triplet_rows = [
        {"block_i": block_i, "block_j": block_j, "block_k": block_k, "count": count}
        for (block_i, block_j, block_k), count in sorted(model_data.triplet_counts.items())
        if count != 0
    ]
    pd.DataFrame(
        triplet_rows,
        columns=["block_i", "block_j", "block_k", "count"],
    ).to_csv(triplet_counts_path, index=False)

    return manifest_path, pair_counts_path, triplet_counts_path


def load_model_data(instance_manifest_path: Path) -> BlockSequencingModelData:
    manifest = json.loads(instance_manifest_path.read_text())
    pair_counts_path = instance_manifest_path.with_name("pair_counts.csv")
    triplet_counts_path = instance_manifest_path.with_name("triplet_counts.csv")

    all_blocks = [int(block) for block in manifest["all_blocks"]]
    pair_counts = {(i, j): 0 for i in all_blocks for j in all_blocks}
    if pair_counts_path.exists():
        pair_df = pd.read_csv(pair_counts_path)
        for row in pair_df.itertuples(index=False):
            pair_counts[(int(row.block_i), int(row.block_j))] = int(row.count)

    triplet_counts = {(i, j, k): 0 for i in all_blocks for j in all_blocks for k in all_blocks}
    if triplet_counts_path.exists():
        triplet_df = pd.read_csv(triplet_counts_path)
        for row in triplet_df.itertuples(index=False):
            triplet_counts[(int(row.block_i), int(row.block_j), int(row.block_k))] = int(
                row.count
            )

    return BlockSequencingModelData(
        all_blocks=all_blocks,
        virtual_blocks=[int(block) for block in manifest["virtual_blocks"]],
        pair_counts=pair_counts,
        triplet_counts=triplet_counts,
        large_blocks={int(block) for block in manifest["large_blocks"]},
        early_slots=[int(slot) for slot in manifest["early_slots"]],
        triple_day_start=[int(slot) for slot in manifest["triple_day_start"]],
        triple_24_start=[int(slot) for slot in manifest["triple_24_start"]],
        eve_morn_start=[int(slot) for slot in manifest["eve_morn_start"]],
        other_b2b_start=[int(slot) for slot in manifest["other_b2b_start"]],
    )


def write_preprocessing_outputs(
    instance: PreparedSequencingInstance, output_prefix: str
) -> tuple[Path, Path, Path]:
    instance_output_dir = build_output_dir(output_prefix, OUTPUTS_DIR)
    instance_output_dir.mkdir(parents=True, exist_ok=True)

    block_map_path = write_block_assignment_outputs(
        exam_to_block=instance.exam_to_block,
        output_prefix=output_prefix,
        output_dir=OUTPUTS_DIR,
    )

    block_summary_path = instance_output_dir / "block_summary.csv"
    build_block_summary(instance).to_csv(block_summary_path, index=False)

    return instance_output_dir, block_map_path, block_summary_path


def build_block_sequencing_model(
    model_data: BlockSequencingModelData,
    alpha: int = DEFAULT_ALPHA,
    beta: int = DEFAULT_BETA,
    gamma1: int = DEFAULT_GAMMA1,
    gamma2: int = DEFAULT_GAMMA2,
    delta: int = DEFAULT_DELTA,
    time_limit: int = DEFAULT_TIME_LIMIT,
    threads: Optional[int] = None,
):
    blocks = model_data.all_blocks
    next_slot = {slot: blocks[(idx + 1) % len(blocks)] for idx, slot in enumerate(blocks)}
    triple_slots = sorted(model_data.triple_day_start + model_data.triple_24_start)

    block_sequence_slot = [
        (i, j, k, s) for i in blocks for j in blocks for k in blocks for s in blocks
    ]
    block_sequence_trip = [(i, j, k) for i in blocks for j in blocks for k in blocks]
    block_sequence_quad = [
        (i, j, k, l)
        for i in blocks
        for j in blocks
        for k in blocks
        for l in blocks
    ]

    model = gp.Model("BlockSequencing")
    model.setParam("TimeLimit", time_limit)
    if threads is not None:
        if threads <= 0:
            raise ValueError("threads must be positive when provided.")
        model.setParam("Threads", int(threads))

    x = model.addVars(block_sequence_slot, vtype=GRB.BINARY, name="x")
    y = model.addVars(block_sequence_trip, vtype=GRB.BINARY, name="y")
    z = model.addVars(block_sequence_quad, vtype=GRB.BINARY, name="z")

    model.addConstrs(
        (
            gp.quicksum(x[i, j, k, s] for j in blocks for k in blocks for s in blocks)
            == 1
            for i in blocks
        ),
        name="each_i",
    )
    model.addConstrs(
        (
            gp.quicksum(x[i, j, k, s] for i in blocks for k in blocks for s in blocks)
            == 1
            for j in blocks
        ),
        name="each_j",
    )
    model.addConstrs(
        (
            gp.quicksum(x[i, j, k, s] for i in blocks for j in blocks for s in blocks)
            == 1
            for k in blocks
        ),
        name="each_k",
    )
    model.addConstrs(
        (
            gp.quicksum(x[i, j, k, s] for i in blocks for j in blocks for k in blocks)
            == 1
            for s in blocks
        ),
        name="each_slot",
    )

    model.addConstrs(
        (x[i, i, k, s] == 0 for i in blocks for k in blocks for s in blocks),
        name="no_ii",
    )
    model.addConstrs(
        (x[i, j, i, s] == 0 for i in blocks for j in blocks for s in blocks),
        name="no_ik",
    )
    model.addConstrs(
        (x[i, j, j, s] == 0 for i in blocks for j in blocks for s in blocks),
        name="no_jj",
    )

    model.addConstrs(
        (
            gp.quicksum(x[i, j, k, s] for i in blocks)
            == gp.quicksum(x[j, k, l, next_slot[s]] for l in blocks)
            for j in blocks
            for k in blocks
            for s in blocks
        ),
        name="continuity",
    )

    model.addConstrs(
        (
            y[i, j, k] == gp.quicksum(x[i, j, k, s] for s in triple_slots)
            for i, j, k in block_sequence_trip
        ),
        name="define_y",
    )
    model.addConstrs(
        (
            z[i, j, k, l] >= y[i, j, k] + y[j, k, l] - 1
            for i, j, k, l in block_sequence_quad
        ),
        name="define_z",
    )

    if model_data.large_blocks:
        model.addConstrs(
            (
                gp.quicksum(
                    x[i, j, k, s]
                    for j in blocks
                    for k in blocks
                    for s in model_data.early_slots
                )
                == 1
                for i in sorted(model_data.large_blocks)
            ),
            name="frontload",
        )

    objective = (
        gp.quicksum(
            gamma1 * model_data.pair_counts[(i, j)] * x[i, j, k, s]
            for i in blocks
            for j in blocks
            for k in blocks
            for s in model_data.eve_morn_start
        )
        + gp.quicksum(
            gamma2 * model_data.pair_counts[(i, j)] * x[i, j, k, s]
            for i in blocks
            for j in blocks
            for k in blocks
            for s in model_data.other_b2b_start
        )
        + gp.quicksum(
            alpha * model_data.triplet_counts[(i, j, k)] * x[i, j, k, s]
            for i in blocks
            for j in blocks
            for k in blocks
            for s in model_data.triple_day_start
        )
        + gp.quicksum(
            beta * model_data.triplet_counts[(i, j, k)] * x[i, j, k, s]
            for i in blocks
            for j in blocks
            for k in blocks
            for s in model_data.triple_24_start
        )
        + gp.quicksum(
            delta
            * (
                model_data.triplet_counts[(i, j, k)]
                + model_data.triplet_counts[(i, k, l)]
            )
            * z[i, j, k, l]
            for i in blocks
            for j in blocks
            for k in blocks
            for l in blocks
        )
    )
    model.setObjective(objective, GRB.MINIMIZE)

    return model, x


def write_model_outputs(
    model,
    output_dir: Path,
    optimize: bool,
    x_vars=None,
    virtual_blocks: Optional[Set[int]] = None,
    stats_path: Optional[Path] = None,
) -> Tuple[Path, Optional[Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.update()

    lp_path = output_dir / "model.lp"
    model.write(str(lp_path))

    slot_summary_path = None
    if optimize:
        model.optimize()
        if x_vars is not None and model.SolCount > 0:
            slot_assignment: dict[int, int] = {}
            for (block_i, _, _, slot), var in x_vars.items():
                if var.X > 0.5:
                    slot_assignment[slot] = block_i

            slot_rows = [
                {
                    "slot": slot,
                    "block": slot_assignment[slot],
                    "is_virtual": slot_assignment[slot] in (virtual_blocks or set()),
                }
                for slot in sorted(slot_assignment)
            ]
            slot_summary_path = output_dir / "slot_summary.csv"
            pd.DataFrame(slot_rows).to_csv(slot_summary_path, index=False)

    total_vars = int(model.NumVars)
    bin_vars = int(model.NumBinVars)
    gen_int_vars = max(int(model.NumIntVars) - bin_vars, 0)
    cont_vars = max(total_vars - bin_vars - gen_int_vars, 0)
    stats_row = {
        "generator": "block_seq",
        "var_bin": bin_vars,
        "var_int": gen_int_vars,
        "var_cont": cont_vars,
        "constraints": int(model.NumConstrs),
    }
    resolved_stats_path = stats_path if stats_path is not None else OUTPUTS_DIR / "stats.csv"
    resolved_stats_path.parent.mkdir(parents=True, exist_ok=True)
    header = not resolved_stats_path.exists()
    pd.DataFrame([stats_row]).to_csv(
        resolved_stats_path,
        mode="a",
        header=header,
        index=False,
    )
    print(f"Model stats      => {stats_row}")
    print(f"Stats path       => {resolved_stats_path}")

    return lp_path, slot_summary_path


def build_output_prefix(
    num_exams: int,
    num_blocks: int,
    num_slots: int,
    seed: int,
) -> str:
    return f"blockseq_n{num_exams}_blocks{num_blocks}_slots{num_slots}_seed{seed}"


def run_block_sequencing(
    num_blocks: int,
    num_slots: int,
    size: Optional[int] = None,
    frontload_block_size_cutoff: Optional[int] = None,
    frontload_slot_cutoff: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    inputs_dir: Path = INPUTS_DIR,
    optimize: bool = False,
    stats_path: Optional[Path] = None,
    time_limit: int = DEFAULT_TIME_LIMIT,
    threads: Optional[int] = None,
):
    instance = prepare_block_sequencing_instance(
        num_blocks=num_blocks,
        num_slots=num_slots,
        size=size,
        frontload_block_size_cutoff=frontload_block_size_cutoff,
        frontload_slot_cutoff=frontload_slot_cutoff,
        seed=seed,
        inputs_dir=inputs_dir,
    )

    output_prefix = build_output_prefix(
        num_exams=len(instance.exam_ids),
        num_blocks=num_blocks,
        num_slots=num_slots,
        seed=seed,
    )
    (
        instance_output_dir,
        block_map_path,
        block_summary_path,
    ) = write_preprocessing_outputs(
        instance=instance, output_prefix=output_prefix
    )
    model_data = extract_model_data(instance)
    (
        instance_manifest_path,
        pair_counts_path,
        triplet_counts_path,
    ) = write_model_data_outputs(model_data, instance_output_dir)
    loaded_model_data = load_model_data(instance_manifest_path)

    model, x_vars = build_block_sequencing_model(
        loaded_model_data,
        time_limit=time_limit,
        threads=threads,
    )
    lp_path, slot_summary_path = write_model_outputs(
        model=model,
        output_dir=instance_output_dir,
        optimize=optimize,
        x_vars=x_vars,
        virtual_blocks=set(instance.virtual_blocks),
        stats_path=stats_path,
    )

    print(
        "Real blocks      => "
        f"{len(instance.real_blocks)} requested, {instance.greedy_block_count} in greedy baseline"
    )
    print(f"Within conflicts => {instance.within_block_conflicts}")
    print(f"Output dir       => {instance_output_dir}")
    print(f"Block map        => {block_map_path}")
    print(f"Block summary    => {block_summary_path}")
    print(f"Instance file    => {instance_manifest_path}")
    print(f"Pair counts      => {pair_counts_path}")
    print(f"Triplet counts   => {triplet_counts_path}")
    print(f"LP model         => {lp_path}")
    if slot_summary_path is not None:
        print(f"Slot summary     => {slot_summary_path}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--size", type=int, help="Use only exams 1..N from the input data")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible synthetic inputs (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=INPUTS_DIR,
        help="Input bundle directory; synthetic caches are derived from this bundle when needed",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=DEFAULT_NUM_BLOCKS,
        help=(
            "Exact number of real blocks to build before adding virtual blocks "
            f"(default: {DEFAULT_NUM_BLOCKS})"
        ),
    )
    parser.add_argument(
        "--num-slots",
        type=int,
        default=DEFAULT_NUM_SLOTS,
        help=(
            "Total number of exam slots; must be at least the number of real blocks "
            f"(default: {DEFAULT_NUM_SLOTS})"
        ),
    )
    parser.add_argument(
        "--frontload-block-size-cutoff",
        type=int,
        default=DEFAULT_FRONTLOAD_BLOCK_SIZE_CUTOFF,
        help=(
            "Front-load real blocks that contain any exam above this size cutoff "
            f"(default: {DEFAULT_FRONTLOAD_BLOCK_SIZE_CUTOFF})"
        ),
    )
    parser.add_argument(
        "--frontload-slot-cutoff",
        type=int,
        default=DEFAULT_FRONTLOAD_SLOT_CUTOFF,
        help=(
            "Restrict large blocks to the first K slots; values above --num-slots "
            f"use all slots (default: {DEFAULT_FRONTLOAD_SLOT_CUTOFF})"
        ),
    )
    parser.add_argument(
        "--no-frontload",
        action="store_true",
        help="Disable front-loading even if the default cutoff values are present",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Gurobi after writing the LP; default is save-only",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=DEFAULT_TIME_LIMIT,
        help=f"Gurobi time limit in seconds (default: {DEFAULT_TIME_LIMIT})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Optional Gurobi thread limit",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        help="Optional CSV path for model stats; defaults to block_seq/outputs/stats.csv",
    )
    args = parser.parse_args()

    frontload_block_size_cutoff = args.frontload_block_size_cutoff
    frontload_slot_cutoff = args.frontload_slot_cutoff
    if args.no_frontload:
        frontload_block_size_cutoff = None
        frontload_slot_cutoff = None

    run_block_sequencing(
        num_blocks=args.num_blocks,
        num_slots=args.num_slots,
        size=args.size,
        frontload_block_size_cutoff=frontload_block_size_cutoff,
        frontload_slot_cutoff=frontload_slot_cutoff,
        seed=args.seed,
        inputs_dir=args.inputs_dir,
        optimize=args.optimize,
        stats_path=args.stats_path,
        time_limit=args.time_limit,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
