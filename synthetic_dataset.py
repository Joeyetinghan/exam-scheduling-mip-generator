import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd


GENERATOR_VERSION = 1
EXPECTED_BUNDLE_FILES = (
    "exam_sizes.csv",
    "anon_coenrol.csv",
    "anon_t_co.csv",
    "metadata.json",
)
DEFAULT_BUNDLE_PROBS = {2: 0.35, 3: 0.45, 4: 0.20}


def load_pairwise_matrix(inputs_dir: Path) -> pd.DataFrame:
    pair_df = pd.read_csv(inputs_dir / "anon_coenrol.csv", index_col=0)
    pair_df.index = pair_df.index.astype(int)
    pair_df.columns = pair_df.columns.astype(int)
    pair_df = pair_df.sort_index().sort_index(axis=1)

    if list(pair_df.index) != list(pair_df.columns):
        raise ValueError(f"{inputs_dir}/anon_coenrol.csv must be square with matching IDs.")

    return pair_df


def load_subset_pairwise_matrix(inputs_dir: Path, size: Optional[int] = None) -> pd.DataFrame:
    pair_df = load_pairwise_matrix(inputs_dir)
    if size is None:
        return pair_df
    if size <= 0:
        raise ValueError("--size must be positive when provided.")
    if size > len(pair_df.index):
        raise ValueError(
            f"--size={size} exceeds the {len(pair_df.index)} exams available in "
            f"{inputs_dir / 'anon_coenrol.csv'}."
        )
    exam_ids = [exam_id for exam_id in pair_df.index if exam_id <= size]
    return pair_df.loc[exam_ids, exam_ids]


def load_exam_sizes(inputs_dir: Path) -> pd.Series:
    size_df = pd.read_csv(inputs_dir / "exam_sizes.csv")
    size_df["exam"] = size_df["exam"].astype(int)
    size_df["size"] = size_df["size"].astype(int)
    return size_df.set_index("exam")["size"].sort_index()


def available_exam_count(inputs_dir: Path) -> int:
    return len(load_pairwise_matrix(inputs_dir).index)


def generated_dataset_dir(inputs_dir: Path, size: int, seed: int) -> Path:
    return inputs_dir / "generated" / f"synth_n{size}_seed{seed}"


def _load_triplet_frame(inputs_dir: Path) -> pd.DataFrame:
    trip_df = pd.read_csv(inputs_dir / "anon_t_co.csv")
    required = {"a", "b", "c", "co"}
    if not required.issubset(trip_df.columns):
        raise ValueError(f"{inputs_dir}/anon_t_co.csv must contain columns {sorted(required)}.")
    return trip_df


def _load_public_bundle(inputs_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    return (
        load_pairwise_matrix(inputs_dir),
        load_exam_sizes(inputs_dir),
        _load_triplet_frame(inputs_dir),
    )


def _build_public_communities(pair_df: pd.DataFrame) -> dict[int, int]:
    graph = nx.from_pandas_adjacency(pair_df)
    communities = list(nx.community.greedy_modularity_communities(graph, weight="weight"))
    if not communities:
        communities = [{exam_id} for exam_id in pair_df.index]

    exam_to_community: dict[int, int] = {}
    for community_id, community in enumerate(communities, start=1):
        for exam_id in community:
            exam_to_community[int(exam_id)] = community_id

    for exam_id in pair_df.index:
        exam_to_community.setdefault(int(exam_id), len(exam_to_community) + 1)
    return exam_to_community


def _sample_target_sizes(
    public_sizes: pd.Series, target_size: int, rng: np.random.Generator
) -> tuple[dict[int, int], dict[int, int]]:
    public_exam_ids = public_sizes.index.to_numpy(dtype=int)
    template_choices = rng.choice(public_exam_ids, size=target_size, replace=True)

    min_size = int(public_sizes.min())
    max_size = int(public_sizes.max())
    exam_templates: dict[int, int] = {}
    synthetic_sizes: dict[int, int] = {}
    for synthetic_exam_id, template_exam_id in enumerate(template_choices, start=1):
        template_size = int(public_sizes.at[int(template_exam_id)])
        sampled_size = int(
            round(
                rng.normal(
                    loc=template_size,
                    scale=max(2.0, template_size * 0.08),
                )
            )
        )
        synthetic_sizes[synthetic_exam_id] = int(np.clip(sampled_size, min_size, max_size))
        exam_templates[synthetic_exam_id] = int(template_exam_id)

    return exam_templates, synthetic_sizes


def _sample_bundle_sizes(total_exam_seats: int, rng: np.random.Generator) -> list[int]:
    bundle_values = np.array(sorted(DEFAULT_BUNDLE_PROBS), dtype=int)
    bundle_probs = np.array([DEFAULT_BUNDLE_PROBS[value] for value in bundle_values], dtype=float)
    expected_bundle_size = float(np.dot(bundle_values, bundle_probs))
    student_count = max(1, int(round(total_exam_seats / expected_bundle_size)))

    bundle_sizes = list(
        rng.choice(bundle_values, size=student_count, replace=True, p=bundle_probs).astype(int)
    )
    seat_delta = total_exam_seats - sum(bundle_sizes)

    while seat_delta != 0:
        if seat_delta > 0:
            expandable = [idx for idx, size in enumerate(bundle_sizes) if size < 4]
            if expandable:
                bundle_sizes[int(rng.choice(expandable))] += 1
                seat_delta -= 1
            else:
                bundle_sizes.append(2)
                seat_delta -= 2
        else:
            shrinkable = [idx for idx, size in enumerate(bundle_sizes) if size > 2]
            if shrinkable:
                bundle_sizes[int(rng.choice(shrinkable))] -= 1
                seat_delta += 1
            else:
                removable = [idx for idx, size in enumerate(bundle_sizes) if size == 2]
                if not removable:
                    raise ValueError("Unable to shrink sampled bundle sizes to match target seats.")
                del bundle_sizes[int(rng.choice(removable))]
                seat_delta += 2

    return sorted(bundle_sizes, reverse=True)


def _candidate_weights(
    candidate_exam_ids: np.ndarray,
    selected_exams: list[int],
    remaining_seats: np.ndarray,
    synthetic_communities: np.ndarray,
    template_affinity: np.ndarray,
) -> np.ndarray:
    weights = remaining_seats[candidate_exam_ids].astype(float)
    if not selected_exams:
        return weights

    selected_array = np.array(selected_exams, dtype=int)
    same_community = (
        synthetic_communities[candidate_exam_ids][:, None]
        == synthetic_communities[selected_array][None, :]
    ).sum(axis=1)
    affinity = template_affinity[np.ix_(candidate_exam_ids, selected_array)].mean(axis=1)

    weights *= 1.0 + same_community + affinity
    weights[weights <= 0] = 1.0
    return weights


def _assign_students(
    synthetic_sizes: dict[int, int],
    synthetic_communities: dict[int, int],
    exam_templates: dict[int, int],
    public_pair_df: pd.DataFrame,
    rng: np.random.Generator,
) -> list[list[int]]:
    total_exam_seats = sum(synthetic_sizes.values())
    bundle_sizes = _sample_bundle_sizes(total_exam_seats, rng)
    num_exams = len(synthetic_sizes)

    remaining_seats = np.zeros(num_exams + 1, dtype=int)
    synthetic_community_array = np.zeros(num_exams + 1, dtype=int)
    template_index = np.zeros(num_exams + 1, dtype=int)
    for exam_id, size in synthetic_sizes.items():
        remaining_seats[exam_id] = int(size)
        synthetic_community_array[exam_id] = int(synthetic_communities[exam_id])
        template_index[exam_id] = int(exam_templates[exam_id])

    max_pair_value = float(public_pair_df.to_numpy().max())
    if max_pair_value <= 0:
        template_affinity = np.ones((num_exams + 1, num_exams + 1), dtype=float)
    else:
        template_affinity = np.ones((num_exams + 1, num_exams + 1), dtype=float)
        public_exam_order = [int(exam_id) for exam_id in public_pair_df.index]
        public_exam_to_position = {
            exam_id: position for position, exam_id in enumerate(public_exam_order)
        }
        template_positions = np.array(
            [public_exam_to_position[template_index[exam_id]] for exam_id in range(1, num_exams + 1)],
            dtype=int,
        )
        normalized_public_pair = public_pair_df.to_numpy(dtype=float) / max_pair_value
        template_affinity[1:, 1:] = 1.0 + normalized_public_pair[
            np.ix_(template_positions, template_positions)
        ]

    students: list[list[int]] = []
    for bundle_size in bundle_sizes:
        student_bundle: list[int] = []
        for _ in range(bundle_size):
            available_exam_ids = np.flatnonzero(remaining_seats > 0)
            if len(available_exam_ids) == 0:
                raise ValueError("Synthetic seat assignment ran out of available exam seats.")
            if student_bundle:
                available_exam_ids = np.array(
                    [exam_id for exam_id in available_exam_ids if exam_id not in student_bundle],
                    dtype=int,
                )
            if len(available_exam_ids) == 0:
                raise ValueError(
                    "Synthetic generator could not fill a student bundle without duplicates."
                )

            weights = _candidate_weights(
                candidate_exam_ids=available_exam_ids,
                selected_exams=student_bundle,
                remaining_seats=remaining_seats,
                synthetic_communities=synthetic_community_array,
                template_affinity=template_affinity,
            )
            weights = weights / weights.sum()
            chosen_exam = int(rng.choice(available_exam_ids, p=weights))
            student_bundle.append(chosen_exam)
            remaining_seats[chosen_exam] -= 1

        students.append(sorted(student_bundle))

    if int(remaining_seats.sum()) != 0:
        raise ValueError("Synthetic generator failed to assign all requested exam seats.")

    return students


def _aggregate_students(
    students: list[list[int]], target_size: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exam_sizes = Counter()
    pair_counts = np.zeros((target_size, target_size), dtype=int)
    triplet_counts: dict[tuple[int, int, int], int] = defaultdict(int)

    for bundle in students:
        unique_bundle = sorted(set(bundle))
        for exam_id in unique_bundle:
            exam_sizes[exam_id] += 1
        for exam_i, exam_j in combinations(unique_bundle, 2):
            pair_counts[exam_i - 1, exam_j - 1] += 1
            pair_counts[exam_j - 1, exam_i - 1] += 1
        for exam_i, exam_j, exam_k in combinations(unique_bundle, 3):
            triplet_counts[(exam_i, exam_j, exam_k)] += 1

    exam_size_df = pd.DataFrame(
        [{"exam": exam_id, "size": int(exam_sizes[exam_id])} for exam_id in range(1, target_size + 1)]
    )
    pair_df = pd.DataFrame(
        pair_counts,
        index=list(range(1, target_size + 1)),
        columns=list(range(1, target_size + 1)),
    )
    triplet_df = pd.DataFrame(
        [
            {
                "a": exam_i - 1,
                "b": exam_j - 1,
                "c": exam_k - 1,
                "triplets_mapped": f"({exam_i - 1}, {exam_j - 1}, {exam_k - 1})",
                "co": int(count),
            }
            for (exam_i, exam_j, exam_k), count in sorted(triplet_counts.items())
        ],
        columns=["a", "b", "c", "triplets_mapped", "co"],
    )
    return exam_size_df, pair_df, triplet_df


def generate_synthetic_dataset(
    base_inputs_dir: Path, target_size: int, seed: int
) -> dict[str, object]:
    if target_size <= 0:
        raise ValueError("target_size must be positive.")

    public_pair_df, public_sizes, public_triplets = _load_public_bundle(base_inputs_dir)
    exam_to_community = _build_public_communities(public_pair_df)
    rng = np.random.default_rng(seed)

    exam_templates, synthetic_sizes = _sample_target_sizes(public_sizes, target_size, rng)
    synthetic_communities = {
        synthetic_exam_id: exam_to_community[template_exam_id]
        for synthetic_exam_id, template_exam_id in exam_templates.items()
    }
    students = _assign_students(
        synthetic_sizes=synthetic_sizes,
        synthetic_communities=synthetic_communities,
        exam_templates=exam_templates,
        public_pair_df=public_pair_df,
        rng=rng,
    )
    exam_size_df, pair_df, triplet_df = _aggregate_students(students, target_size)

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "target_size": int(target_size),
        "student_count": int(len(students)),
        "source_inputs_dir": str(base_inputs_dir.resolve()),
        "source_exam_count": int(len(public_pair_df.index)),
        "source_triplet_rows": int(len(public_triplets)),
        "bundle_size_distribution": {
            str(bundle_size): int(count)
            for bundle_size, count in sorted(Counter(len(bundle) for bundle in students).items())
        },
        "max_exam_size": int(exam_size_df["size"].max()),
        "min_exam_size": int(exam_size_df["size"].min()),
        "pair_sum": int(pair_df.to_numpy().sum()),
        "triplet_rows": int(len(triplet_df)),
    }

    return {
        "exam_sizes": exam_size_df,
        "anon_coenrol": pair_df,
        "anon_t_co": triplet_df,
        "metadata": metadata,
    }


def _cache_is_valid(output_dir: Path, metadata: dict[str, object]) -> bool:
    if not output_dir.exists():
        return False
    if any(not (output_dir / file_name).exists() for file_name in EXPECTED_BUNDLE_FILES):
        return False

    try:
        cached_metadata = json.loads((output_dir / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False

    for key in ("generator_version", "seed", "target_size", "source_inputs_dir", "source_exam_count"):
        if cached_metadata.get(key) != metadata.get(key):
            return False
    return True


def ensure_synthetic_dataset(base_inputs_dir: Path, target_size: int, seed: int) -> Path:
    base_inputs_dir = Path(base_inputs_dir)
    output_dir = generated_dataset_dir(base_inputs_dir, target_size, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "target_size": int(target_size),
        "source_inputs_dir": str(base_inputs_dir.resolve()),
        "source_exam_count": int(available_exam_count(base_inputs_dir)),
    }
    if _cache_is_valid(output_dir, metadata):
        return output_dir

    generated = generate_synthetic_dataset(base_inputs_dir=base_inputs_dir, target_size=target_size, seed=seed)
    generated["exam_sizes"].to_csv(output_dir / "exam_sizes.csv", index=False)
    generated["anon_coenrol"].to_csv(output_dir / "anon_coenrol.csv")
    generated["anon_t_co"].to_csv(output_dir / "anon_t_co.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(generated["metadata"], indent=2, sort_keys=True)
    )
    return output_dir


def resolve_inputs_dir_for_size(inputs_dir: Path, size: Optional[int], seed: int) -> Path:
    inputs_dir = Path(inputs_dir)
    if size is None:
        return inputs_dir

    available_count = available_exam_count(inputs_dir)
    if size <= available_count:
        return inputs_dir
    return ensure_synthetic_dataset(base_inputs_dir=inputs_dir, target_size=size, seed=seed)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--size", type=int, required=True, help="Exact number of synthetic exams")
    parser.add_argument("--seed", type=int, default=3, help="Random seed for reproducible bundles")
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "inputs",
        help="Base input bundle used to calibrate the synthetic dataset",
    )
    args = parser.parse_args()

    output_dir = ensure_synthetic_dataset(
        base_inputs_dir=args.inputs_dir,
        target_size=args.size,
        seed=args.seed,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
