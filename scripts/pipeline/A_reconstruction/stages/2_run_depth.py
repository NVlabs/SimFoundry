# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified stage-2 depth runner.

Supports both stereo (FoundationStereo) and monocular/video (DepthAnythingV3)
backends with a shared interface and shared stage metadata output.
"""

from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

# pycolmap needs to be imported first to avoid environment issues
try:
    import pycolmap  # noqa: F401 - must happen before torch
except ImportError:
    logger.warning("pycolmap not found, skipping import")

import hydra

from simfoundry.pipeline.depth_backends import create_backend
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage


from simfoundry import CFG_DIR
bootstrap_hydra_workdir(__file__)


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    backend_name = cfg.s2_depth.backend
    backend = create_backend(backend_name)

    logger.info("=" * 60)
    logger.info("Starting stage-2 depth pipeline with backend: %s", backend_name)
    logger.info("=" * 60)
    result = backend.run(cfg)
    logger.info("Completed backend=%s with %s inputs", result.backend, result.n_inputs)
    logger.info("=" * 60)

    # Mirror legacy stage-info placement for compatibility.
    target_stage_cfg = cfg.s2_fs if backend_name == "fs" else cfg.s2_da
    target_out_dir = cfg.s2_fs.out_dir if backend_name == "fs" else cfg.s2_da.out_dir
    finalize_stage(
        stage_cfg=target_stage_cfg,
        out_dir=target_out_dir,
        result=StageResult(
            success=True,
            additional_info={"backend": result.backend, "n_inputs": result.n_inputs, "backend_output_dir": result.output_dir},
        ),
    )


if __name__ == "__main__":
    main()
