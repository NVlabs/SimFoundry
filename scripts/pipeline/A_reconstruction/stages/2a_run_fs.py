# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility wrapper for stereo depth stage.

This script preserves the historical entrypoint name while delegating to the
unified depth runner with `s2_depth.backend=fs`.
"""

from __future__ import annotations

import hydra

from simfoundry.pipeline.depth_backends import create_backend
from simfoundry.pipeline.stage_utils import StageResult, bootstrap_hydra_workdir, finalize_stage


from simfoundry import CFG_DIR
bootstrap_hydra_workdir(__file__)


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    backend = create_backend("fs")
    result = backend.run(cfg)
    finalize_stage(
        stage_cfg=cfg.s2_fs,
        out_dir=cfg.s2_fs.out_dir,
        result=StageResult(
            success=True,
            additional_info={"backend": result.backend, "n_inputs": result.n_inputs, "backend_output_dir": result.output_dir},
        ),
    )


if __name__ == "__main__":
    main()
