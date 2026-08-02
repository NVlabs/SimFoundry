# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
