import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
from itertools import combinations
from gurobipy import Env, Model, GRB, quicksum

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
INPUTS_DIR = ROOT_DIR / "inputs"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

ANON_FILE = INPUTS_DIR / "anon_coenrol.csv"
HIST_TOTALS_FILE = INPUTS_DIR / "hist_totals.csv"


def pick_course_subset(n, total_courses):
    """
    Top-n are simply IDs 1…n. If n>total, append new IDs above total.
    """
    if n <= 0:
        return []
    if n <= total_courses:
        return list(range(1, n + 1))
    extra = n - total_courses
    new_ids = list(range(total_courses + 1, total_courses + 1 + extra))
    random.shuffle(new_ids)
    return list(range(1, total_courses + 1)) + new_ids


def simulate_future_semester(selected, m_samples, seed=None, rand_frac=0.3):
    selected = list(dict.fromkeys(selected))
    if seed is not None:
        np.random.seed(seed)

    pairs = list(combinations(selected, 2))
    idx_map = {c: i for i, c in enumerate(selected)}
    hist = pd.read_csv(HIST_TOTALS_FILE, index_col=["course1", "course2"]).iloc[:, 0]

    weights = np.array([hist.get(p, 0) for p in pairs], float)
    if weights.sum() > 0:
        probs = weights / weights.sum()
    else:
        probs = np.full(len(pairs), 1 / len(pairs))

    n_w = int(m_samples * (1 - rand_frac))
    n_r = m_samples - n_w
    w_idxs = np.random.choice(len(pairs), size=n_w, p=probs)
    r_idxs = np.random.randint(0, len(pairs), size=n_r)

    counts = np.bincount(np.concatenate([w_idxs, r_idxs]), minlength=len(pairs))
    nz = np.nonzero(counts)[0]

    # build symmetric matrix
    n = len(selected)
    M = np.zeros((n, n), int)
    for i in nz:
        c1, c2 = pairs[i]
        cnt = counts[i]
        a, b = idx_map[c1], idx_map[c2]
        M[a, b] = M[b, a] = cnt

    dfM = pd.DataFrame(M, index=selected, columns=selected)
    stats = {
        "samples": m_samples,
        "weighted": n_w,
        "random": n_r,
        "unique": len(nz),
        "max": int(counts.max()),
    }
    return dfM, stats


def run_simulation(size, seed=3, optimize=False):
    anon = pd.read_csv(ANON_FILE, index_col=0)
    total = anon.shape[0]
    subset = pick_course_subset(size, total)
    m_samples = int(size * size / 5)
    co, stats = simulate_future_semester(subset, m_samples, seed)

    exams = list(co.columns)
    blocks = list(range(1, 25))

    # Use default Gurobi license discovery; credentials must come from environment or license file
    env = Env()
    model = Model(env=env)

    x = model.addVars(exams, blocks, vtype=GRB.BINARY, name="x")
    model.addConstrs(
        (quicksum(x[e, b] for b in blocks) == 1 for e in exams), name="assign_once"
    )
    conflict = quicksum(
        co.at[e1, e2] * x[e1, b] * x[e2, b]
        for b in blocks
        for e1 in exams
        for e2 in exams
        if e1 != e2
    )
    model.setObjective(conflict, GRB.MINIMIZE)
    if optimize:
        model.optimize()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / f"blockassign_n{size}_seed{seed}.lp"
    model.write(str(out_path))
    print(f"Done. LP ⇒ {out_path}, stats: {stats}")

    # Record model statistics (variable/constraint counts)
    try:
        model.update()
        total_vars = int(model.NumVars)
        bin_vars = int(model.NumBinVars)
        # Some Gurobi versions include binary vars in NumIntVars — subtract them
        gen_int_vars = max(int(model.NumIntVars) - bin_vars, 0)
        cont_vars = max(total_vars - bin_vars - gen_int_vars, 0)
        mip_stats = {
            "generator": "block_assign",
            "size": int(size) if size is not None else None,
            "seed": int(seed) if seed is not None else None,
            "var_bin": bin_vars,
            "var_int": gen_int_vars,
            "var_cont": cont_vars,
            "constraints": int(model.NumConstrs),
        }
        stats_path = OUTPUTS_DIR / "stats.csv"
        header = not stats_path.exists()
        pd.DataFrame([mip_stats]).to_csv(stats_path, mode="a", header=header, index=False)
        print(f"Model stats ⇒ {mip_stats}")
    except Exception as e:
        print(f"Warning: unable to record model stats: {e}")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--size", type=int, help="Number of courses to schedule")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--optimize", action="store_true", help="Run optimizer; default is save-only")
    args = parser.parse_args()
    run_simulation(args.size, seed=args.seed, optimize=args.optimize)


if __name__ == "__main__":
    main()


