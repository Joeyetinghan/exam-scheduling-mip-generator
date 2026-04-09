import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
INPUTS_DIR = ROOT_DIR / "inputs"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from synthetic_dataset import load_exam_sizes as load_all_exam_sizes
from synthetic_dataset import load_subset_pairwise_matrix, resolve_inputs_dir_for_size

DEFAULT_NUM_BLOCKS = 24
DEFAULT_SEED = 3


@dataclass(frozen=True)
class PreparedBlockAssignment:
    pair_df: pd.DataFrame
    exam_sizes: pd.Series
    exam_ids: list[int]
    resolved_inputs_dir: Path
    greedy_exam_to_block: dict[int, int]
    exam_to_block: dict[int, int]
    greedy_block_count: int
    within_block_conflicts: int


def load_pairwise_coenrollment(
    inputs_dir: Path = INPUTS_DIR, size: Optional[int] = None
) -> pd.DataFrame:
    return load_subset_pairwise_matrix(inputs_dir, size=size)


def load_exam_sizes(
    exam_ids: list[int], inputs_dir: Path = INPUTS_DIR
) -> pd.Series:
    exam_sizes = load_all_exam_sizes(inputs_dir)
    missing = sorted(set(exam_ids) - set(exam_sizes.index))
    if missing:
        raise ValueError(
            "exam_sizes.csv is missing exam sizes for the selected exams: "
            f"{missing[:10]}"
        )
    return exam_sizes.loc[exam_ids]


def build_greedy_block_assignment(pair_df: pd.DataFrame) -> dict[int, int]:
    graph = nx.from_pandas_adjacency(pair_df)
    color_map = nx.greedy_color(graph, strategy="largest_first", interchange=True)

    used_colors = sorted(set(color_map.values()))
    color_to_block = {color: idx + 1 for idx, color in enumerate(used_colors)}
    return {int(exam_id): color_to_block[color] for exam_id, color in color_map.items()}


def invert_block_assignment(exam_to_block: dict[int, int]) -> dict[int, list[int]]:
    block_to_exams: dict[int, list[int]] = {}
    for exam_id, block_id in sorted(exam_to_block.items()):
        block_to_exams.setdefault(block_id, []).append(int(exam_id))
    return block_to_exams


def normalize_block_assignment(block_to_exams: dict[int, list[int]]) -> dict[int, int]:
    exam_to_block: dict[int, int] = {}
    for new_block_id, old_block_id in enumerate(sorted(block_to_exams), start=1):
        exams = sorted(block_to_exams[old_block_id])
        if not exams:
            raise ValueError("Cannot normalize a block assignment that contains empty blocks.")
        for exam_id in exams:
            exam_to_block[int(exam_id)] = new_block_id
    return exam_to_block


def _pair_key(block_i: int, block_j: int) -> tuple[int, int]:
    return (block_i, block_j) if block_i < block_j else (block_j, block_i)


def _block_loads(
    block_to_exams: dict[int, list[int]], exam_sizes: pd.Series
) -> dict[int, int]:
    return {
        block_id: int(exam_sizes.loc[exams].sum()) for block_id, exams in block_to_exams.items()
    }


def _build_block_pair_conflicts(
    pair_df: pd.DataFrame, block_to_exams: dict[int, list[int]]
) -> dict[tuple[int, int], int]:
    pair_conflicts: dict[tuple[int, int], int] = {}
    ordered_blocks = sorted(block_to_exams)
    for idx, block_i in enumerate(ordered_blocks):
        exams_i = block_to_exams[block_i]
        for block_j in ordered_blocks[idx + 1 :]:
            exams_j = block_to_exams[block_j]
            pair_conflicts[_pair_key(block_i, block_j)] = int(
                pair_df.loc[exams_i, exams_j].to_numpy().sum()
            )
    return pair_conflicts


def merge_blocks_to_target(
    pair_df: pd.DataFrame,
    exam_sizes: pd.Series,
    exam_to_block: dict[int, int],
    num_blocks: int,
) -> dict[int, int]:
    block_to_exams = invert_block_assignment(exam_to_block)
    if num_blocks <= 0:
        raise ValueError("--num-blocks must be positive.")
    if num_blocks > len(block_to_exams):
        raise ValueError("merge_blocks_to_target requires a smaller target block count.")

    block_loads = _block_loads(block_to_exams, exam_sizes)
    pair_conflicts = _build_block_pair_conflicts(pair_df, block_to_exams)
    target_load = float(exam_sizes.sum()) / num_blocks
    next_block_id = max(block_to_exams)

    while len(block_to_exams) > num_blocks:
        best_choice: Optional[tuple[int, float, int, int, int]] = None
        ordered_blocks = sorted(block_to_exams)
        for idx, block_i in enumerate(ordered_blocks):
            for block_j in ordered_blocks[idx + 1 :]:
                merge_conflict = pair_conflicts[_pair_key(block_i, block_j)]
                merged_load = block_loads[block_i] + block_loads[block_j]
                load_gap = abs(merged_load - target_load)
                choice = (
                    merge_conflict,
                    load_gap,
                    merged_load,
                    min(block_i, block_j),
                    max(block_i, block_j),
                )
                if best_choice is None or choice < best_choice:
                    best_choice = choice

        if best_choice is None:
            raise ValueError("No merge candidates were available while reducing blocks.")

        _, _, _, block_i, block_j = best_choice
        next_block_id += 1
        block_to_exams[next_block_id] = sorted(
            block_to_exams[block_i] + block_to_exams[block_j]
        )
        block_loads[next_block_id] = block_loads[block_i] + block_loads[block_j]

        for other_block in sorted(block_to_exams):
            if other_block in {block_i, block_j, next_block_id}:
                continue
            pair_conflicts[_pair_key(next_block_id, other_block)] = (
                pair_conflicts.get(_pair_key(block_i, other_block), 0)
                + pair_conflicts.get(_pair_key(block_j, other_block), 0)
            )

        pair_conflicts = {
            key: value
            for key, value in pair_conflicts.items()
            if block_i not in key and block_j not in key
        }
        del block_to_exams[block_i]
        del block_to_exams[block_j]
        del block_loads[block_i]
        del block_loads[block_j]

    return normalize_block_assignment(block_to_exams)


def split_blocks_to_target(
    exam_sizes: pd.Series,
    exam_to_block: dict[int, int],
    num_blocks: int,
) -> dict[int, int]:
    block_to_exams = invert_block_assignment(exam_to_block)
    if num_blocks < len(block_to_exams):
        raise ValueError("split_blocks_to_target requires a larger target block count.")

    next_block_id = max(block_to_exams)
    while len(block_to_exams) < num_blocks:
        splittable_blocks = [block for block, exams in block_to_exams.items() if len(exams) >= 2]
        if not splittable_blocks:
            raise ValueError(
                f"Cannot create {num_blocks} nonempty real blocks from the selected exams."
            )

        block_to_split = max(
            splittable_blocks,
            key=lambda block: (
                int(exam_sizes.loc[block_to_exams[block]].sum()),
                len(block_to_exams[block]),
                -block,
            ),
        )
        exams = sorted(
            block_to_exams[block_to_split],
            key=lambda exam_id: (-int(exam_sizes.at[exam_id]), exam_id),
        )

        group_a = [exams[0]]
        group_b = [exams[1]]
        load_a = int(exam_sizes.at[exams[0]])
        load_b = int(exam_sizes.at[exams[1]])
        for exam_id in exams[2:]:
            exam_size = int(exam_sizes.at[exam_id])
            if load_a <= load_b:
                group_a.append(exam_id)
                load_a += exam_size
            else:
                group_b.append(exam_id)
                load_b += exam_size

        block_to_exams[block_to_split] = sorted(group_a)
        next_block_id += 1
        block_to_exams[next_block_id] = sorted(group_b)

    return normalize_block_assignment(block_to_exams)


def _exam_conflict_weight(
    pair_df: pd.DataFrame, exam_id: int, block_exams: list[int]
) -> int:
    other_exams = [other_exam for other_exam in block_exams if other_exam != exam_id]
    if not other_exams:
        return 0
    return int(pair_df.loc[exam_id, other_exams].sum())


def rebalance_block_assignment(
    pair_df: pd.DataFrame,
    exam_sizes: pd.Series,
    exam_to_block: dict[int, int],
    preserve_zero_conflicts: bool,
    max_passes: int = 25,
) -> dict[int, int]:
    if max_passes <= 0:
        return normalize_block_assignment(invert_block_assignment(exam_to_block))

    assignment = dict(exam_to_block)
    ordered_exams = sorted(assignment, key=lambda exam_id: (-int(exam_sizes.at[exam_id]), exam_id))

    for _ in range(max_passes):
        improved = False
        block_to_exams = invert_block_assignment(assignment)
        block_loads = _block_loads(block_to_exams, exam_sizes)

        for exam_id in ordered_exams:
            current_block = assignment[exam_id]
            if len(block_to_exams[current_block]) <= 1:
                continue

            current_conflict = _exam_conflict_weight(
                pair_df, exam_id, block_to_exams[current_block]
            )
            exam_size = int(exam_sizes.at[exam_id])
            candidate_blocks = sorted(block_to_exams)

            for target_block in candidate_blocks:
                if target_block == current_block:
                    continue

                target_conflict = _exam_conflict_weight(
                    pair_df, exam_id, block_to_exams[target_block]
                )
                if preserve_zero_conflicts and target_conflict > 0:
                    continue

                current_difference = abs(
                    block_loads[current_block] - block_loads[target_block]
                )
                new_difference = abs(
                    (block_loads[current_block] - exam_size)
                    - (block_loads[target_block] + exam_size)
                )

                if target_conflict < current_conflict or (
                    target_conflict == current_conflict
                    and new_difference < current_difference
                ):
                    block_to_exams[current_block].remove(exam_id)
                    block_to_exams[target_block].append(exam_id)
                    block_to_exams[target_block].sort()
                    block_loads[current_block] -= exam_size
                    block_loads[target_block] += exam_size
                    assignment[exam_id] = target_block
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return normalize_block_assignment(invert_block_assignment(assignment))


def build_target_block_assignment(
    pair_df: pd.DataFrame, exam_sizes: pd.Series, num_blocks: int
) -> tuple[dict[int, int], dict[int, int], int]:
    if num_blocks <= 0:
        raise ValueError("--num-blocks must be positive.")
    if num_blocks > len(pair_df.index):
        raise ValueError(
            f"--num-blocks={num_blocks} exceeds the {len(pair_df.index)} selected exams."
        )

    greedy_assignment = build_greedy_block_assignment(pair_df)
    greedy_block_count = len(set(greedy_assignment.values()))

    if num_blocks == greedy_block_count:
        final_assignment = rebalance_block_assignment(
            pair_df=pair_df,
            exam_sizes=exam_sizes,
            exam_to_block=greedy_assignment,
            preserve_zero_conflicts=True,
        )
    elif num_blocks < greedy_block_count:
        final_assignment = rebalance_block_assignment(
            pair_df=pair_df,
            exam_sizes=exam_sizes,
            exam_to_block=merge_blocks_to_target(
                pair_df=pair_df,
                exam_sizes=exam_sizes,
                exam_to_block=greedy_assignment,
                num_blocks=num_blocks,
            ),
            preserve_zero_conflicts=False,
        )
    else:
        final_assignment = rebalance_block_assignment(
            pair_df=pair_df,
            exam_sizes=exam_sizes,
            exam_to_block=split_blocks_to_target(
                exam_sizes=exam_sizes,
                exam_to_block=greedy_assignment,
                num_blocks=num_blocks,
            ),
            preserve_zero_conflicts=True,
        )

    return greedy_assignment, final_assignment, greedy_block_count


def count_within_block_conflicts(
    pair_df: pd.DataFrame, exam_to_block: dict[int, int]
) -> int:
    conflicts = 0
    for exam_i in pair_df.index:
        for exam_j in pair_df.columns:
            if exam_i < exam_j and exam_to_block[int(exam_i)] == exam_to_block[int(exam_j)]:
                conflicts += int(pair_df.at[exam_i, exam_j])
    return conflicts


def prepare_block_assignment(
    num_blocks: int,
    size: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    inputs_dir: Path = INPUTS_DIR,
) -> PreparedBlockAssignment:
    resolved_inputs_dir = resolve_inputs_dir_for_size(
        inputs_dir=inputs_dir,
        size=size,
        seed=seed,
    )
    pair_df = load_pairwise_coenrollment(inputs_dir=resolved_inputs_dir, size=size)
    exam_ids = [int(exam_id) for exam_id in pair_df.index]
    exam_sizes = load_exam_sizes(exam_ids, inputs_dir=resolved_inputs_dir)
    greedy_exam_to_block, exam_to_block, greedy_block_count = build_target_block_assignment(
        pair_df=pair_df,
        exam_sizes=exam_sizes,
        num_blocks=num_blocks,
    )
    within_block_conflicts = count_within_block_conflicts(pair_df, exam_to_block)
    return PreparedBlockAssignment(
        pair_df=pair_df,
        exam_sizes=exam_sizes,
        exam_ids=exam_ids,
        resolved_inputs_dir=resolved_inputs_dir,
        greedy_exam_to_block=greedy_exam_to_block,
        exam_to_block=exam_to_block,
        greedy_block_count=greedy_block_count,
        within_block_conflicts=within_block_conflicts,
    )


def build_output_prefix(num_exams: int, num_blocks: int, seed: int) -> str:
    return f"blockseq_blockassign_n{num_exams}_blocks{num_blocks}_seed{seed}"


def build_output_dir(output_prefix: str, base_output_dir: Path = OUTPUTS_DIR) -> Path:
    return base_output_dir / output_prefix


def write_block_assignment_outputs(
    exam_to_block: dict[int, int],
    output_prefix: str,
    output_dir: Path = OUTPUTS_DIR,
) -> Path:
    instance_output_dir = build_output_dir(output_prefix, output_dir)
    instance_output_dir.mkdir(parents=True, exist_ok=True)

    block_map_path = instance_output_dir / "blockmap.csv"
    pd.DataFrame(
        sorted(exam_to_block.items()), columns=["exam", "block"]
    ).to_csv(block_map_path, index=False)

    return block_map_path


def run_greedy_block_assignment(
    num_blocks: int,
    size: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    inputs_dir: Path = INPUTS_DIR,
) -> tuple[PreparedBlockAssignment, Path]:
    assignment = prepare_block_assignment(
        num_blocks=num_blocks,
        size=size,
        seed=seed,
        inputs_dir=inputs_dir,
    )
    output_prefix = build_output_prefix(
        num_exams=len(assignment.exam_ids),
        num_blocks=num_blocks,
        seed=seed,
    )
    block_map_path = write_block_assignment_outputs(
        exam_to_block=assignment.exam_to_block,
        output_prefix=output_prefix,
    )
    print(f"Input bundle      => {assignment.resolved_inputs_dir}")
    print(f"Exam count        => {len(assignment.exam_ids)}")
    print(f"Requested blocks  => {num_blocks}")
    print(f"Greedy baseline   => {assignment.greedy_block_count}")
    print(f"Within conflicts  => {assignment.within_block_conflicts}")
    print(f"Final block map   => {block_map_path}")
    return assignment, block_map_path


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
        help=f"Exact number of real blocks to build (default: {DEFAULT_NUM_BLOCKS})",
    )
    args = parser.parse_args()

    run_greedy_block_assignment(
        num_blocks=args.num_blocks,
        size=args.size,
        seed=args.seed,
        inputs_dir=args.inputs_dir,
    )


if __name__ == "__main__":
    main()
