# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Run Nerfstudio video pipeline and train splatfacto-big (Gaussian Splatting).

Step 1: ns-process-data video --data <video> --output-dir <processed_dir>
Step 2: ns-train splatfacto-big (with TORCH_DYNAMO_DISABLE=1) using processed data.
Step 3: ns-export gaussian-splat to PLY (export dir under s2c_gs).

All intermediate outputs live under Data/{scene_name}/s2c_gs/:
  - s2c_gs/processed/   <- ns-process-data output
  - s2c_gs/outputs/     <- ns-train output (created in base dir so cwd=s2c_gs)
  - s2c_gs/export/      <- exported Gaussian splat PLY

transforms.json only lists frames that COLMAP successfully registered. Use
matching_method=vocab_tree (default here) so most frames get poses; sequential
often registers only a few.

Should be run from the `nerfstudio_simfoundry` environment.
"""
import logging
import os
import shutil
import subprocess
from pathlib import Path

import hydra

logger = logging.getLogger(__name__)

# Resolve scripts/cfg the same way the sibling stages do (location-independent,
# without importing stage_utils and its image-processing dependencies into the
# intentionally minimal Nerfstudio environment.
from simfoundry import CFG_DIR

os.chdir(CFG_DIR)


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    s2c = cfg.s2c_gs
    base_dir = Path(s2c.out_dir).resolve()
    processed_dir = (base_dir / s2c.processed_dirname).resolve()
    video_fpath = s2c.video_fpath

    base_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_fpath).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Step 1: process video with nerfstudio (skip if processed dir already exists)
    if processed_dir.exists() and (processed_dir / "transforms.json").exists():
        logger.info("Processed data already exists at %s, skipping ns-process-data", processed_dir)
    else:
        # Remove leftover COLMAP output so we don't reuse a corrupted database
        # (e.g. "SQLite error: database disk image is malformed" from a previous aborted run)
        colmap_dir = processed_dir / "colmap"
        if colmap_dir.exists():
            logger.info("Removing existing colmap dir to avoid reusing corrupted DB: %s", colmap_dir)
            shutil.rmtree(colmap_dir, ignore_errors=True)
        num_frames = getattr(s2c, "num_frames_target", 1000)
        matching = getattr(s2c, "matching_method", "vocab_tree")
        logger.info(
            "Running ns-process-data video -> %s (num-frames-target=%s, matching-method=%s)",
            processed_dir, num_frames, matching,
        )
        subprocess.run(
            [
                "ns-process-data",
                "video",
                "--data", str(video_path),
                "--output-dir", str(processed_dir),
                "--num-frames-target", str(num_frames),
                "--matching-method", str(matching),
            ],
            check=True,
        )

    # Step 2: train splatfacto-big from base_dir so outputs go to base_dir/outputs
    env = os.environ.copy()
    # Disable torch dynamo/compile to avoid JIT compilation issues
    env["TORCHDYNAMO_DISABLE"] = "1"  # Note: no underscore, this is the correct variable name
    env["TORCH_COMPILE_DISABLE"] = "1"
    # Limit gsplat JIT compilation to the current GPU only (sm_120 = RTX 5090).
    # Without this, PyTorch adds gencode targets for sm_50/52/60/61 (pre-Volta),
    # which don't define _CG_HAS_MATCH_COLLECTIVE, making cg::labeled_partition
    # unavailable and breaking gsplat's Projection*.cu compilation.
    env["TORCH_CUDA_ARCH_LIST"] = "12.0"
    
    # Set CUDA environment variables to help gsplat find CUDA libraries during compilation
    # First check conda environment for CUDA libraries (prioritize these)
    conda_prefix = os.environ.get("CONDA_PREFIX", "")

    # nvcc calls cicc (CUDA's internal compiler) via system(), so cicc must be in PATH.
    # In conda CUDA installs, cicc lives in nvvm/bin/ which isn't added to PATH by default.
    if conda_prefix:
        nvvm_bin = os.path.join(conda_prefix, "nvvm", "bin")
        if os.path.isdir(nvvm_bin):
            existing_path = env.get("PATH", "")
            env["PATH"] = f"{nvvm_bin}:{existing_path}" if existing_path else nvvm_bin
            logger.info("Prepended %s to PATH (cicc location)", nvvm_bin)
    lib_paths = []
    cuda_home = None
    
    # Check conda environment lib directory first
    if conda_prefix:
        # Check multiple possible locations for CUDA libraries in conda env
        conda_cuda_paths = [
            os.path.join(conda_prefix, "lib"),
            os.path.join(conda_prefix, "targets", "x86_64-linux", "lib"),
            os.path.join(conda_prefix, "lib", "python3.8", "site-packages", "nvidia", "cuda_runtime", "lib"),
        ]
        
        # Check for libcudart in various locations
        # Prioritize targets/x86_64-linux/lib where the actual library file is
        for check_path in conda_cuda_paths:
            if os.path.exists(check_path):
                # Check for any libcudart.so file (including versioned ones)
                try:
                    found_lib = False
                    for item in os.listdir(check_path):
                        if item.startswith("libcudart.so"):
                            # Check if it's an actual file (not broken symlink)
                            lib_file = os.path.join(check_path, item)
                            if os.path.exists(lib_file) and (os.path.islink(lib_file) or os.path.isfile(lib_file)):
                                # Resolve symlinks to check if target exists
                                real_path = os.path.realpath(lib_file)
                                if os.path.exists(real_path):
                                    if check_path not in lib_paths:
                                        # Prioritize targets directory
                                        if "targets" in check_path:
                                            lib_paths.insert(0, check_path)
                                        else:
                                            lib_paths.append(check_path)
                                    found_lib = True
                                    break
                except (OSError, PermissionError):
                    pass
        
        # Use conda env as CUDA_HOME. Prefer the targets/x86_64-linux subtree because
        # that's where CUDA headers (cuda_runtime_api.h) actually live in conda's layout.
        # PyTorch cpp_extension looks for ${CUDA_HOME}/include/cuda_runtime_api.h, so
        # pointing at the conda root (which lacks an include/ dir) causes build failures.
        targets_dir = os.path.join(conda_prefix, "targets", "x86_64-linux")
        if os.path.exists(os.path.join(targets_dir, "include", "cuda_runtime_api.h")):
            cuda_home = targets_dir
        elif os.path.exists(os.path.join(conda_prefix, "bin", "nvcc")):
            cuda_home = conda_prefix
    
    # Check system CUDA installation paths (prioritize these over conda for linking)
    # System CUDA installations have libraries in standard locations that build systems expect
    cuda_paths = [
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12",
        "/usr/local/cuda",
    ]
    system_cuda_found = False
    for path in cuda_paths:
        if os.path.exists(path):
            # Check for libcudart in lib64 or lib
            lib64_path = os.path.join(path, "lib64")
            lib_path = os.path.join(path, "lib")
            if os.path.exists(os.path.join(lib64_path, "libcudart.so")):
                # Prefer system CUDA over conda for CUDA_HOME (better compatibility)
                cuda_home = path
                if lib64_path not in lib_paths:
                    lib_paths.insert(0, lib64_path)  # Prioritize system CUDA libs
                system_cuda_found = True
                break
            elif os.path.exists(os.path.join(lib_path, "libcudart.so")):
                cuda_home = path
                if lib_path not in lib_paths:
                    lib_paths.insert(0, lib_path)  # Prioritize system CUDA libs
                system_cuda_found = True
                break
    
    # Set environment variables
    if cuda_home:
        env["CUDA_HOME"] = cuda_home
        env["CUDA_PATH"] = cuda_home
        logger.info("Set CUDA_HOME=%s", cuda_home)
    
    if lib_paths:
        existing_ld_path = env.get("LD_LIBRARY_PATH", "")
        new_ld_path = ":".join(lib_paths)
        env["LD_LIBRARY_PATH"] = f"{new_ld_path}:{existing_ld_path}" if existing_ld_path else new_ld_path
        logger.info("Updated LD_LIBRARY_PATH with CUDA library paths: %s", ":".join(lib_paths))
        
        # Set LDFLAGS for linker - PyTorch's build system should respect this
        existing_ldflags = env.get("LDFLAGS", "")
        ldflags_paths = " ".join([f"-L{path}" for path in lib_paths])
        env["LDFLAGS"] = f"{ldflags_paths} {existing_ldflags}".strip()
        logger.info("Updated LDFLAGS with CUDA library paths")
        
        # Also set LIBRARY_PATH which some build systems use
        existing_lib_path = env.get("LIBRARY_PATH", "")
        new_lib_path = ":".join(lib_paths)
        env["LIBRARY_PATH"] = f"{new_lib_path}:{existing_lib_path}" if existing_lib_path else new_lib_path
        logger.info("Updated LIBRARY_PATH with CUDA library paths")
        
        # For PyTorch's cpp_extension, also try setting CUDA_LIBRARY_PATH
        # This might help the build system find the libraries
        cuda_lib_path = ":".join(lib_paths)
        env["CUDA_LIBRARY_PATH"] = cuda_lib_path
        logger.info("Set CUDA_LIBRARY_PATH=%s", cuda_lib_path)
    
    if not cuda_home and not lib_paths:
        logger.warning("Could not find CUDA installation. gsplat compilation may fail.")
    
    # Add system include paths to help find headers like crypt.h during compilation
    # This is needed for CUDA extension compilation (gsplat, Triton, etc.)
    # nvcc needs these paths to find system headers when compiling CUDA code
    system_include_paths = [
        "/usr/include",
        "/usr/include/x86_64-linux-gnu",
    ]
    # NOTE: Injecting /usr/include into C_INCLUDE_PATH / CPLUS_INCLUDE_PATH breaks
    # libstdc++'s `#include_next <math.h>` chain on a g++-11 + nvcc-12.x host
    # toolchain (the chain expects /usr/include to be a *later* search dir, not an
    # explicit one), so gsplat's JIT compile dies with
    # "/usr/include/c++/11/cmath: fatal error: math.h: No such file or directory".
    # This is the canonical nerfstudio_v2 toolchain, so default to NOT injecting.
    # Opt back in (for envs that genuinely need it, e.g. to find crypt.h) via
    # SIMFOUNDRY_INJECT_SYSTEM_INCLUDES=1.
    if os.environ.get("SIMFOUNDRY_INJECT_SYSTEM_INCLUDES", "0") == "1":
        existing_paths = [p for p in system_include_paths if os.path.exists(p)]
    else:
        existing_paths = []
    
    if existing_paths:
        # Set standard compiler include path environment variables
        # These are automatically used by gcc/g++/nvcc
        existing_c_include = env.get("C_INCLUDE_PATH", "")
        existing_cplus_include = env.get("CPLUS_INCLUDE_PATH", "")
        include_path_str = ":".join(existing_paths)
        
        env["C_INCLUDE_PATH"] = f"{include_path_str}:{existing_c_include}" if existing_c_include else include_path_str
        env["CPLUS_INCLUDE_PATH"] = f"{include_path_str}:{existing_cplus_include}" if existing_cplus_include else include_path_str
        logger.info("Set C_INCLUDE_PATH and CPLUS_INCLUDE_PATH with system include paths")
        
        # Also set CPPFLAGS, CFLAGS, CXXFLAGS for compatibility
        existing_cppflags = env.get("CPPFLAGS", "")
        include_flags = " ".join([f"-I{path}" for path in existing_paths])
        if include_flags:
            env["CPPFLAGS"] = f"{include_flags} {existing_cppflags}".strip()
            env["CFLAGS"] = f"{include_flags} {env.get('CFLAGS', '')}".strip()
            env["CXXFLAGS"] = f"{include_flags} {env.get('CXXFLAGS', '')}".strip()
            env["CUDAFLAGS"] = f"{include_flags} {env.get('CUDAFLAGS', '')}".strip()
            logger.info("Added system include paths to CPPFLAGS, CFLAGS, CXXFLAGS, CUDAFLAGS")
    
    logger.info("Running ns-train %s (max %s iters) from %s", s2c.method, s2c.max_num_iterations, base_dir)
    subprocess.run(
        [
            "ns-train",
            s2c.method,
            "--max-num-iterations", str(s2c.max_num_iterations),
            "--data", str(processed_dir),
            "--viewer.quit-on-train-completion", "True",  # exit when done so script can run export
        ],
        cwd=str(base_dir),
        env=env,
        check=True,
    )
    outputs_dir = base_dir / "outputs"
    logger.info("Training done. Outputs under %s", outputs_dir)

    # Step 3: export Gaussian splat to PLY (latest run)
    export_dir = base_dir / getattr(s2c, "export_dirname", "export")
    configs = sorted(outputs_dir.glob("**/config.yml"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not configs:
        raise FileNotFoundError(f"No training config found under {outputs_dir}")
    load_config = configs[0]
    export_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting GS from %s -> %s", load_config, export_dir)
    subprocess.run(
        [
            "ns-export",
            "gaussian-splat",
            "--load-config", str(load_config),
            "--output-dir", str(export_dir),
        ],
        cwd=str(base_dir),
        env=env,
        check=True,
    )
    logger.info("Done. Export under %s", export_dir)


if __name__ == "__main__":
    main()
