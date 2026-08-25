# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import shutil
from pathlib import Path


#: Marker inserted before the extension of the temporary file an artifact is built under.
#: It goes *before* the extension, not after, because trimesh and PIL pick their output
#: format from it -- `foo.glb.partial` would not export as a GLB. `foo.partial.glb` keeps
#: the format and still fails the watcher's `_mesh.glb` / `_transparent.png` suffix match.
PARTIAL_MARKER = ".partial"


def partial_path(fpath):
    """Temp path an artifact is built under before publication: `a/b.glb` -> `a/b.partial.glb`."""
    root, ext = os.path.splitext(str(fpath))
    return f"{root}{PARTIAL_MARKER}{ext}"


@contextlib.contextmanager
def atomic_output_path(fpath):
    """Yield a temp path to write to, then publish it to `fpath` with an atomic rename.

    The streaming subsequence (stages 5-8) dispatches a consumer as soon as a producer's
    artifact appears in the watched directory, matching on filename. Writing in place makes
    that a race: the name appears when the file is *created*, not when it is finished, so a
    consumer can open a partial file. Stage 8 hit exactly this, reading a half-copied 41 MB
    .glb as a mesh with zero triangles.

    Waiting for the file to stop changing cannot close it, because a producer may stall
    mid-write for an unbounded time -- `shutil.copyfile` creates the destination and only
    then streams the bytes, and under I/O contention that gap ran past a 5 s settle window.

    Renaming within a directory is atomic on POSIX, so a consumer observes either no file or
    the finished one. The temp file is removed if the body raises, so a failed write leaves
    no artifact behind rather than a corrupt one.

    Args:
        fpath (str): Final path to publish to. Its parent directory must already exist.

    Yields:
        str: Path to write to. Renamed onto `fpath` when the block exits without error.
    """
    fpath = str(fpath)
    tmp_path = partial_path(fpath)
    try:
        yield tmp_path
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise
    # os.replace is atomic within a filesystem and overwrites an existing destination.
    os.replace(tmp_path, fpath)


def atomic_copyfile(src, dst):
    """Copy `src` to `dst`, publishing it atomically. See atomic_output_path."""
    with atomic_output_path(dst) as tmp:
        shutil.copyfile(src, tmp)
    return dst


def assert_valid_key(key, valid_keys, name=None):
    """
    Helper function that asserts that @key is in dictionary @valid_keys keys. If not, it will raise an error.

    Args:
        key (any): key to check for in dictionary @dic's keys
        valid_keys (Iterable): contains keys should be checked with @key
        name (str or None): if specified, is the name associated with the key that will be printed out if the
            key is not found. If None, default is "value"
    """
    if name is None:
        name = "value"
    assert key in valid_keys, "Invalid {} received! Valid options are: {}, got: {}".format(
        name, valid_keys.keys() if isinstance(valid_keys, dict) else valid_keys, key
    )





def sanitize_path_component(value):
    """Normalize a scene or object name into a filesystem path component.

    Stage 8b writes articulation results to
    ``<out_dir>/<scene>/<object>/results/`` and stage 10 reads them back. Both sides
    must agree exactly, so this is the single definition they share: spaces and
    slashes become underscores, and the result is lowercased so that a scene named
    ``Laptop`` and one named ``laptop`` resolve to the same directory.
    """
    return str(value).replace(" ", "_").replace("/", "_").lower()


def resolve_task_config_path(cfg_dir, task_name, *, group_choice=None, scene_name=None):
    """Locate the task YAML a run is asking for.

    Hydra selects a task by its *config-group path*, so a config filed under a
    subdirectory -- ``task=droid/cluttered_scene/nv_desk_place_baseball_in_bowl``
    -- still carries a bare ``task_name``, and the flat ``task/<task_name>.yaml``
    the stages used to build simply does not exist. That surfaced as an eval
    that connected to a policy, loaded a scene, and then died naming a file
    nobody had asked for.

    ``task_name`` is tried first, and a scene-specific copy before that: the
    ``task=load_scene task.task_name=serve_the_orange`` indirection is how a
    scene picks its own task, and it has to keep winning over the group Hydra
    selected.

    Args:
        cfg_dir (str or Path): ``scripts/cfg``.
        task_name (str): ``cfg.task.task_name``.
        group_choice (str or None): The task group Hydra resolved, e.g.
            ``droid/cluttered_scene/nv_desk_place_baseball_in_bowl``.
        scene_name (str or None): Checked first, as ``task/<scene>/<task>.yaml``.

    Returns:
        str: Path to the config.

    Raises:
        FileNotFoundError: Naming every path that was tried, since "which file
            did it want" is the only question worth answering here.
    """
    root = Path(cfg_dir) / "task"
    candidates = []
    if scene_name:
        candidates.append(root / str(scene_name) / f"{task_name}.yaml")
    candidates.append(root / f"{task_name}.yaml")
    if group_choice:
        candidates.append(root / f"{group_choice}.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"no task config for task_name={task_name!r}; tried "
        + ", ".join(str(c) for c in candidates))
