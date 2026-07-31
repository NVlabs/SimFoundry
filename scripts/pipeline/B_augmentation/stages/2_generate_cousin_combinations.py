# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Should be run from simfoundry

Requires installing:

- simfoundry, see the main README
"""

import hydra
import json
import math
import random
from pathlib import Path
from itertools import product
from hydra.utils import to_absolute_path

from simfoundry import CFG_DIR
from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir

bootstrap_hydra_workdir(__file__)

SUBDIRS = ["geometry", "topology", "visual"]


def collect_iter_pool(iter_dir: Path):
    """
    Collect all *_transparent.png under geometry/topology/visual

    Rule:
      - topology is OPTIONAL
      - at least one *_transparent.png must exist somewhere
    """
    pool = []

    for sub in SUBDIRS:
        subdir = iter_dir / sub
        if not subdir.exists():
            continue

        pool.extend(subdir.glob("*_transparent.png"))

    if len(pool) == 0:
        return None

    # sort for determinism
    return sorted(pool)


def list_iter_dirs(source_dir: Path):
    """
    Discover existing iter directories without assuming contiguous numbering.
    """
    iter_dirs = []
    for p in source_dir.glob("iter_*"):
        if not p.is_dir():
            continue
        try:
            idx = int(p.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        iter_dirs.append((idx, p.name))

    # numeric sort: iter_2 before iter_10
    iter_dirs.sort(key=lambda x: x[0])
    return [name for _, name in iter_dirs]


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    # --------------------------------------------------
    # Resolve paths (Hydra-safe)
    # --------------------------------------------------
    source_dir = Path(to_absolute_path(cfg.prompt_cousin_structured.out_dir))
    out_dir = Path(to_absolute_path(cfg.generate_cousins_combination.out_dir))

    K = cfg.generate_cousins_combination.num_variations_used_per_object
    M = cfg.generate_cousins_combination.num_obj_to_swap
    seed = cfg.generate_cousins_combination.seed
    max_combinations = cfg.generate_cousins_combination.max_combinations

    random.seed(seed)

    if K <= 0:
        raise ValueError("num_variations_used_per_object must be > 0")
    if M <= 0:
        raise ValueError("num_obj_to_swap must be > 0")
    if max_combinations is not None and max_combinations <= 0:
        raise ValueError("max_combinations must be > 0 or null")

    # --------------------------------------------------
    # --------------------------------------------------
    # Discover all swappable objects from disk
    # --------------------------------------------------
    all_iters = list_iter_dirs(source_dir)
    if len(all_iters) == 0:
        raise RuntimeError(f"No valid iter_* directories found under: {source_dir}")

    iter_pools = {}

    for name in all_iters:
        iter_dir = source_dir / name
        pool = collect_iter_pool(iter_dir)
        if pool is None:
            raise RuntimeError(f"{name} has no *_transparent.png variants")

        iter_pools[name] = pool

    if M > len(all_iters):
        print(
            f"Warning: num_obj_to_swap={M} exceeds available objects={len(all_iters)}; "
            f"using M={len(all_iters)}."
        )
        M = len(all_iters)

    # --------------------------------------------------
    # STEP 1: randomly choose M objects to swap
    # --------------------------------------------------
    swap_iters = sorted(random.sample(all_iters, M))
    fixed_iters = [it for it in all_iters if it not in swap_iters]

    print(f"Swapping {M} objects: {swap_iters}")
    print(f"Fixed objects: {fixed_iters}")

    # --------------------------------------------------
    # STEP 2: sample K variants for swapped objects
    # --------------------------------------------------
    swap_variants = {}
    available_k = min(len(iter_pools[it]) for it in swap_iters)
    if available_k < K:
        print(
            f"Warning: requested K={K}, but the selected objects only have "
            f"{available_k} variant(s); using K={available_k}."
        )
        K = available_k

    for it in swap_iters:
        pool = iter_pools[it]
        if len(pool) < K:
            raise RuntimeError(
                f"{it} has only {len(pool)} variants, but K={K}"
            )
        swap_variants[it] = random.sample(pool, K)

    # --------------------------------------------------
    # STEP 3: Randomly sample combinations (unordered)
    # --------------------------------------------------
    results = []
    swap_lists = [swap_variants[it] for it in swap_iters]
    total_combinations = math.prod(len(lst) for lst in swap_lists)

    def combo_from_index(index):
        combo = []
        for lst in reversed(swap_lists):
            index, rem = divmod(index, len(lst))
            combo.append(lst[rem])
        return list(reversed(combo))

    target_count = (
        total_combinations
        if max_combinations is None
        else min(max_combinations, total_combinations)
    )

    if target_count == total_combinations:
        for combo in product(*swap_lists):
            entry = {
                it: str(path.relative_to(source_dir))
                for it, path in zip(swap_iters, combo)
            }
            results.append(entry)
        random.shuffle(results)
    else:
        sampled_indices = random.sample(range(total_combinations), target_count)
        for idx in sampled_indices:
            combo = combo_from_index(idx)
            entry = {
                it: str(path.relative_to(source_dir))
                for it, path in zip(swap_iters, combo)
            }
            results.append(entry)


    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    output_file = out_dir / "combinations.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(
        f"Generated {len(results)} combinations out of {total_combinations} "
        f"(swapping {M} objects)"
    )
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
