import argparse
import sys
from pathlib import Path
from typing import Optional

import gurobipy as gp
import pandas as pd
from gurobipy import GRB


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
INPUTS_DIR = ROOT_DIR / "inputs"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
DEFAULT_SEED = 3
DEFAULT_BLOCKS = 24

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from synthetic_dataset import load_subset_pairwise_matrix, resolve_inputs_dir_for_size


def prepare_block_assignment_data(
    size: Optional[int],
    seed: int = DEFAULT_SEED,
    inputs_dir: Path = INPUTS_DIR,
) -> tuple[pd.DataFrame, Path]:
    resolved_inputs_dir = resolve_inputs_dir_for_size(
        inputs_dir=inputs_dir,
        size=size,
        seed=seed,
    )
    pair_df = load_subset_pairwise_matrix(resolved_inputs_dir, size=size)
    return pair_df, resolved_inputs_dir


def build_block_assignment_model(
    pair_df: pd.DataFrame,
    block_count: int = DEFAULT_BLOCKS,
):
    if block_count <= 0:
        raise ValueError("block_count must be positive.")

    exams = [int(exam_id) for exam_id in pair_df.index]
    blocks = list(range(1, block_count + 1))

    model = gp.Model("BlockAssignment")
    x = model.addVars(exams, blocks, vtype=GRB.BINARY, name="x")
    model.addConstrs(
        (gp.quicksum(x[exam_id, block_id] for block_id in blocks) == 1 for exam_id in exams),
        name="assign_once",
    )
    positive_pairs = [
        (exam_i, exam_j, int(pair_df.at[exam_i, exam_j]))
        for idx, exam_i in enumerate(exams)
        for exam_j in exams[idx + 1 :]
        if int(pair_df.at[exam_i, exam_j]) > 0
    ]
    model.setObjective(
        gp.quicksum(
            pair_count * x[exam_i, block_id] * x[exam_j, block_id]
            for block_id in blocks
            for exam_i, exam_j, pair_count in positive_pairs
        ),
        GRB.MINIMIZE,
    )

    return model


def run_simulation(
    size: Optional[int],
    seed: int = DEFAULT_SEED,
    inputs_dir: Path = INPUTS_DIR,
    block_count: int = DEFAULT_BLOCKS,
    optimize: bool = False,
    stats_path: Optional[Path] = None,
    time_limit: Optional[float] = None,
    threads: Optional[int] = None,
):
    pair_df, resolved_inputs_dir = prepare_block_assignment_data(
        size=size,
        seed=seed,
        inputs_dir=inputs_dir,
    )
    model = build_block_assignment_model(pair_df, block_count=block_count)
    if time_limit is not None:
        if time_limit <= 0:
            raise ValueError("time_limit must be positive when provided.")
        model.setParam("TimeLimit", float(time_limit))
    if threads is not None:
        if threads <= 0:
            raise ValueError("threads must be positive when provided.")
        model.setParam("Threads", int(threads))
    if optimize:
        model.optimize()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    effective_size = int(size) if size is not None else len(pair_df.index)
    out_path = OUTPUTS_DIR / f"blockassign_n{effective_size}_seed{seed}.lp"
    model.write(str(out_path))
    print(f"Input bundle    => {resolved_inputs_dir}")
    print(f"Exam count      => {len(pair_df.index)}")
    print(f"LP model        => {out_path}")

    try:
        model.update()
        total_vars = int(model.NumVars)
        bin_vars = int(model.NumBinVars)
        gen_int_vars = max(int(model.NumIntVars) - bin_vars, 0)
        cont_vars = max(total_vars - bin_vars - gen_int_vars, 0)
        mip_stats = {
            "generator": "block_assign",
            "size": effective_size,
            "seed": int(seed),
            "var_bin": bin_vars,
            "var_int": gen_int_vars,
            "var_cont": cont_vars,
            "constraints": int(model.NumConstrs),
        }
        resolved_stats_path = stats_path if stats_path is not None else OUTPUTS_DIR / "stats.csv"
        resolved_stats_path.parent.mkdir(parents=True, exist_ok=True)
        header = not resolved_stats_path.exists()
        pd.DataFrame([mip_stats]).to_csv(
            resolved_stats_path,
            mode="a",
            header=header,
            index=False,
        )
        print(f"Model stats     => {mip_stats}")
        print(f"Stats path      => {resolved_stats_path}")
    except Exception as exc:  # pragma: no cover - defensive stats recording
        print(f"Warning: unable to record model stats: {exc}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--size", type=int, help="Use only exams 1..N from the selected input bundle")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=INPUTS_DIR,
        help="Input bundle directory; synthetic caches are derived from this bundle when needed",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        help="Optional CSV path for model stats; defaults to block_assign/outputs/stats.csv",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        help="Optional Gurobi time limit in seconds",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Optional Gurobi thread limit",
    )
    parser.add_argument("--optimize", action="store_true", help="Run optimizer; default is save-only")
    args = parser.parse_args()
    run_simulation(
        args.size,
        seed=args.seed,
        inputs_dir=args.inputs_dir,
        optimize=args.optimize,
        stats_path=args.stats_path,
        time_limit=args.time_limit,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
