# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
A shared, read-only cache of USD stages, so each asset is composed once.

Callers only traverse and read, so sharing a stage is safe. Sharing a *stale*
one is not -- this editor writes USDs itself -- so the key is the file's
identity and content: resolved path, modification time and size.

``SdfLayer`` is registered globally by identifier and outlives every stage
built on it, so a plain ``Usd.Stage.Open`` of a rewritten path hands back the
contents from this process's first read. Noticing the mtime change here is
what makes it possible to call ``Reload()`` and pick up the new bytes.
"""

import threading
from collections import OrderedDict
from pathlib import Path

#: Stages kept alive at once: a couple of scenes' worth, so a scene switch does
#: not re-compose everything. Stages hold prim indices, not mesh data.
CACHE_LIMIT = 64

_STAGES = OrderedDict()

#: The last revision opened at each resolved path, and the lock that serialises
#: opening it. Both deliberately outlive LRU eviction: the global ``SdfLayer``
#: registry does too, so the decision to reload cannot depend on a `_STAGES`
#: entry that may already be gone. Keyed on the path alone.
_REVISIONS = {}
_OPENING = {}

#: Guards `_STAGES`, `_REVISIONS` and `_OPENING`: the editor's server is
#: threaded, and the LRU bookkeeping is not safe under concurrent access.
_LOCK = threading.Lock()


def open_stage(usd_path):
    """Open a USD, reusing the stage when the file has not changed on disk.

    A drop-in for ``Usd.Stage.Open(str(path))``, including how it fails: a
    stage OpenUSD declines to open is None here too, and an exception from the
    open propagates rather than being flattened into None. Callers already
    branch on those, and this is not the place to change what they mean.

    Args:
        usd_path (str or Path or None): The asset to open.

    Returns:
        Usd.Stage or None: The stage, or None when there is nothing to open.
    """
    if not usd_path:
        return None
    try:
        from pxr import Usd
    except ImportError:
        return None

    path = Path(usd_path)
    key = cache_key(path)
    if key is None:
        # No stat means no way to tell a later revision from this one, so this
        # open is not cacheable; let OpenUSD decide how a missing file fails.
        return Usd.Stage.Open(str(path))

    with _LOCK:
        cached = _STAGES.get(key)
        if cached is not None:
            _STAGES.move_to_end(key)
            return cached
        opening = _OPENING.setdefault(key[0], threading.Lock())

    # `_LOCK` is not held across the open: threads reading different assets
    # need not wait on each other. Threads opening the *same* asset serialise
    # on the per-path lock so the file is composed once.
    with opening:
        with _LOCK:
            cached = _STAGES.get(key)
            if cached is not None:
                _STAGES.move_to_end(key)
                return cached
            # Any entry for the same file under a different mtime/size is a
            # previous revision, and nothing should be handed it again.
            for stale in [k for k in _STAGES if k[0] == key[0]]:
                del _STAGES[stale]
            previous = _REVISIONS.get(key[0])

        stage = Usd.Stage.Open(str(path))
        if stage is None:
            return None
        if previous is not None and previous != key:
            # The layer under a fresh stage is still the globally registered
            # one from this process's first read; the mtime change seen here
            # is the only signal to Reload() and pick up the new bytes.
            stage.Reload()
        with _LOCK:
            _REVISIONS[key[0]] = key
            _STAGES[key] = stage
            while len(_STAGES) > CACHE_LIMIT:
                _STAGES.popitem(last=False)
    return stage


def cache_key(usd_path):
    """Identity of a USD's current contents: resolved path, mtime and size.

    Shared with `scene_io.read_usd_facts`, so the facts read out of a stage are
    invalidated by exactly what invalidates the stage.

    Returns:
        tuple or None: ``(path, mtime_ns, size)``, or None if it cannot be
        stat'd -- which is the signal not to cache anything about it.
    """
    if not usd_path:
        return None
    path = Path(usd_path)
    try:
        stat = path.stat()
        return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return None


def clear():
    """Forget every cached stage; exists so a test can measure a cold open.

    `_REVISIONS` is kept: it describes the global ``SdfLayer`` registry, which
    dropping a stage does not touch, so clearing it would lose the reload.
    """
    with _LOCK:
        _STAGES.clear()
