# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Step 17: Point-to-Point Matching between Base Mesh and Cousin Meshes

This script:
1. Reads combinations.json from the cousins combination step to determine which cousins to process
2. Finds the corresponding cousin meshes from step 16 (textured_mesh output)
3. Finds the base (canonical) meshes from step 8
4. Runs PartField inference for each base-cousin pair
5. Runs smooth functional map to compute p2p correspondence
6. Saves all results in s17 directory
"""

import os
import sys
import subprocess
import yaml
import shutil
import json
import re
from pathlib import Path
import glob

from simfoundry import CFG_DIR, REPO_DIR
from simfoundry.pipeline.stage_utils import bootstrap_hydra_workdir

# Add project root to path
project_root = Path(REPO_DIR)
sys.path.insert(0, str(project_root))
bootstrap_hydra_workdir(__file__)


def find_cousin_meshes(s16_result_dir, combinations_dir, canonical_mesh_dir, texture_model):
    """
    Find all cousin meshes using the combinations.json and the actual s16 output structure.
    
    The cousins generation pipeline produces:
      1. combinations.json (from generate_cousins_combination.py) listing which cousin
         images were selected per object, e.g.:
           [{"iter_0": "iter_0/geometry/cousin_001_v1_transparent.png", ...}, ...]
      2. Textured meshes (from 3_generate_cousin_meshes.py) saved at:
           s16_cousin_generation/textured_mesh/{texture_model}/{object_name}/{dimension}/{cousin_name}_mesh.glb
      3. Base (canonical) meshes from step 8 at:
           s8_pose/canonical_mesh/{object_name}.glb
    
    Args:
        s16_result_dir: Root s16 output directory (e.g., .../s16_cousin_generation)
        combinations_dir: Directory containing combinations.json (e.g., .../cousins_combination)
        canonical_mesh_dir: Directory containing canonical meshes (e.g., .../s8_pose/canonical_mesh)
        texture_model: Name of the texture model used, i.e. `cousin_generation.texture_model`
    
    Returns:
        dict: {object_name: {'base_mesh': path, 'cousins': [cousin_paths]}}
    """
    result_dict = {}
    
    # Read combinations.json
    comb_file = os.path.join(combinations_dir, "combinations.json")
    if not os.path.exists(comb_file):
        print(f"Warning: combinations.json not found: {comb_file}")
        return result_dict
    
    with open(comb_file, "r") as f:
        combinations = json.load(f)
    
    # Collect all unique cousin image paths per object from the combinations
    # combinations is a list of dicts, e.g.:
    #   [{"iter_0": "iter_0/geometry/cousin_001_v1_transparent.png", "iter_1": ...}, ...]
    object_cousins = {}  # {object_name: set of relative image paths}
    for combo in combinations:
        for obj_name, img_rel_path in combo.items():
            if obj_name not in object_cousins:
                object_cousins[obj_name] = set()
            object_cousins[obj_name].add(img_rel_path)
    
    # Textured mesh directory
    textured_mesh_dir = os.path.join(s16_result_dir, "textured_mesh", texture_model)
    if not os.path.exists(textured_mesh_dir):
        print(f"Warning: Textured mesh directory not found: {textured_mesh_dir}")
        return result_dict
    
    # For each object, resolve base mesh and cousin mesh paths
    for obj_name, img_paths in sorted(object_cousins.items()):
        result_dict[obj_name] = {
            'base_mesh': None,
            'cousins': []
        }
        
        # Base mesh is the canonical mesh from step 8
        base_mesh_path = os.path.join(canonical_mesh_dir, f"{obj_name}.glb")
        if os.path.exists(base_mesh_path):
            result_dict[obj_name]['base_mesh'] = base_mesh_path
        else:
            print(f"Warning: Base (canonical) mesh not found for {obj_name}: {base_mesh_path}")
        
        # Resolve each cousin mesh path from the image path
        # Image path: "iter_0/geometry/cousin_001_v1_transparent.png"
        # Mesh path:  textured_mesh/{texture_model}/iter_0/geometry/cousin_001_v1_transparent_mesh.glb
        for img_rel_path in sorted(img_paths):
            # Remove .png extension and derive mesh filename
            mesh_name = img_rel_path.rsplit('.png', 1)[0]
            cousin_mesh_path = os.path.join(textured_mesh_dir, f"{mesh_name}_mesh.glb")
            
            if os.path.exists(cousin_mesh_path):
                result_dict[obj_name]['cousins'].append(cousin_mesh_path)
            else:
                print(f"Warning: Cousin mesh not found: {cousin_mesh_path}")
        
        # Sort cousins for deterministic ordering
        result_dict[obj_name]['cousins'] = sorted(result_dict[obj_name]['cousins'])
    
    return result_dict


def create_correspondence_config(base_mesh_path, cousin_mesh_path, output_config_path, 
                                 data_dir, result_name):
    """
    Create a temporary config file for PartField correspondence.
    """
    config = {
        'result_name': result_name,
        'continue_ckpt': 'model/model_objaverse.ckpt',
        'triplane_channels_low': 128,
        'triplane_channels_high': 512,
        'triplane_resolution': 128,
        'vertex_feature': True,
        'n_point_per_face': 1000,
        'n_sample_each': 10000,
        'is_pc': False,
        'remesh_demo': False,
        'correspondence_demo': True,
        'preprocess_mesh': True,
        'dataset': {
            'type': 'Mix',
            'data_path': data_dir,
            'train_batch_size': 1,
            'val_batch_size': 1,
            'train_num_workers': 8,
            'all_files': [
                os.path.basename(cousin_mesh_path),  # source (all_files[0])
                os.path.basename(base_mesh_path)     # target (all_files[1])
            ]
        },
        'loss': {
            'triplet': 1.0
        },
        'use_2d_feat': False,
        'pvcnn': {
            'point_encoder_type': 'pvcnn',
            'z_triplane_channels': 256,
            'z_triplane_resolution': 128
        }
    }
    
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return output_config_path


def process_mesh_pair(base_mesh, cousin_mesh, base_name, cousin_idx, 
                     s17_out_dir, partfield_root):
    """
    Process a single base-cousin mesh pair.
    
    Args:
        base_mesh: Path to base mesh
        cousin_mesh: Path to cousin mesh
        base_name: Name of base object
        cousin_idx: Index of cousin
        s17_out_dir: Output directory for s17
        partfield_root: Root directory of PartField
    """
    print(f"\n{'='*80}")
    print(f"Processing: {base_name} <-> Cousin {cousin_idx}")
    print(f"{'='*80}")
    
    # Create subdirectories for this pair
    pair_name = f"{base_name}_cousin_{cousin_idx:03d}"
    pair_data_dir = os.path.join(s17_out_dir, pair_name, "data")
    pair_config_dir = os.path.join(s17_out_dir, pair_name, "config")
    pair_features_dir = os.path.join(s17_out_dir, pair_name, "features")
    pair_correspondence_dir = os.path.join(s17_out_dir, pair_name, "correspondence")
    
    os.makedirs(pair_data_dir, exist_ok=True)
    os.makedirs(pair_config_dir, exist_ok=True)
    os.makedirs(pair_features_dir, exist_ok=True)
    os.makedirs(pair_correspondence_dir, exist_ok=True)
    
    # Copy meshes to data directory
    base_mesh_copy = os.path.join(pair_data_dir, os.path.basename(base_mesh))
    cousin_mesh_copy = os.path.join(pair_data_dir, os.path.basename(cousin_mesh))
    shutil.copy2(base_mesh, base_mesh_copy)
    shutil.copy2(cousin_mesh, cousin_mesh_copy)
    print(f"✓ Copied meshes to {pair_data_dir}")
    
    # Create config file
    config_path = os.path.join(pair_config_dir, "correspondence_config.yaml")
    # PartField result_name is relative to exp_results directory
    # We'll use a temporary name and move files later
    result_name = f"temp_correspondence/{pair_name}"
    create_correspondence_config(
        base_mesh_copy, cousin_mesh_copy, config_path,
        pair_data_dir, result_name
    )
    print(f"✓ Created config: {config_path}")
    
    # Step 1: Run PartField inference
    print(f"\nStep 1: Running PartField inference...")
    inference_cmd = [
        "python", "partfield_inference.py",
        "-c", config_path,
        "--opts",
        "continue_ckpt", "model/model_objaverse.ckpt",
        "preprocess_mesh", "True"
    ]
    
    try:
        subprocess.run(
            inference_cmd,
            cwd=partfield_root,
            check=True,
            capture_output=False
        )
        print("✓ PartField inference completed")
    except subprocess.CalledProcessError as e:
        print(f"✗ PartField inference failed: {e}")
        return False
    
    # Move features to pair directory
    temp_feature_source = os.path.join(partfield_root, "exp_results", "temp_correspondence", pair_name)
    if os.path.exists(temp_feature_source):
        for item in os.listdir(temp_feature_source):
            src = os.path.join(temp_feature_source, item)
            dst = os.path.join(pair_features_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
        # Clean up temp directory
        shutil.rmtree(os.path.join(partfield_root, "exp_results", "temp_correspondence", pair_name))
        print(f"✓ Moved features to {pair_features_dir}")
    else:
        print(f"✗ Warning: Features not found at {temp_feature_source}")
    
    # Step 2: Run smooth functional map
    print(f"\nStep 2: Running smooth functional map...")
    
    # Create a new config for functional map that points to the moved features
    # We need to copy features back to partfield exp_results temporarily for functional map to find them
    temp_fm_result = f"temp_fm/{pair_name}"
    temp_fm_path = os.path.join(partfield_root, "exp_results", temp_fm_result)
    os.makedirs(temp_fm_path, exist_ok=True)
    
    # Copy features to temp location for functional map
    for item in os.listdir(pair_features_dir):
        src = os.path.join(pair_features_dir, item)
        dst = os.path.join(temp_fm_path, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    
    # Create functional map config
    functional_map_config = os.path.join(pair_config_dir, "functional_map_config.yaml")
    with open(config_path, 'r') as f:
        fm_config = yaml.safe_load(f)
    fm_config['result_name'] = temp_fm_result
    with open(functional_map_config, 'w') as f:
        yaml.dump(fm_config, f, default_flow_style=False)
    
    functional_map_cmd = [
        "python", "run_smooth_functional_map.py",
        "-c", functional_map_config,
        "--opts"
    ]
    
    try:
        subprocess.run(
            functional_map_cmd,
            cwd=os.path.join(partfield_root, "applications"),
            check=True,
            capture_output=False
        )
        print("✓ Functional map completed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Functional map failed: {e}")
        return False
    
    # Move correspondence results to pair directory
    # Only move files for this specific pair based on mesh names
    correspondence_source = os.path.join(partfield_root, "exp_results", "correspondence")
    if os.path.exists(correspondence_source):
        # Generate the UIDs used in filenames (based on how run_smooth_functional_map.py names files)
        base_uid = os.path.basename(base_mesh_copy).split(".")[-2].replace("/", "_")
        cousin_uid = os.path.basename(cousin_mesh_copy).split(".")[-2].replace("/", "_")
        
        # Exact filenames for files generated for this pair
        # run_smooth_functional_map creates:
        # - p2p_mapping_{uid0}_{uid1}.npy
        # - p2p_mapping_{uid0}_{uid1}.json
        # - correspondence_{uid0}_{uid1}_0.ply
        # - correspondence_{uid0}_{uid1}_1.ply
        expected_files = [
            f"p2p_mapping_{cousin_uid}_{base_uid}.npy",
            f"p2p_mapping_{cousin_uid}_{base_uid}.json",
            f"correspondence_{cousin_uid}_{base_uid}_0.ply",
            f"correspondence_{cousin_uid}_{base_uid}_1.ply"
        ]
        
        moved_count = 0
        for filename in expected_files:
            src = os.path.join(correspondence_source, filename)
            if os.path.exists(src):
                dst = os.path.join(pair_correspondence_dir, filename)
                shutil.copy2(src, dst)
                moved_count += 1
        
        if moved_count > 0:
            print(f"✓ Moved {moved_count} correspondence files to {pair_correspondence_dir}")
        else:
            print(f"✗ Warning: No correspondence files found.")
    else:
        print(f"✗ Warning: Correspondence directory not found: {correspondence_source}")
    
    # Clean up temporary directories
    if os.path.exists(temp_fm_path):
        shutil.rmtree(temp_fm_path)
    temp_fm_parent = os.path.join(partfield_root, "exp_results", "temp_fm")
    if os.path.exists(temp_fm_parent) and not os.listdir(temp_fm_parent):
        os.rmdir(temp_fm_parent)
    temp_corr_parent = os.path.join(partfield_root, "exp_results", "temp_correspondence")
    if os.path.exists(temp_corr_parent) and not os.listdir(temp_corr_parent):
        os.rmdir(temp_corr_parent)
    
    print(f"\n✓ Successfully processed {pair_name}")
    return True


def main():
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(version_base=None, config_path=CFG_DIR, config_name="real2sim_cfg")
    def run(cfg: DictConfig):
        print("="*80)
        print("Step 17: Point-to-Point Matching for Digital Cousins")
        print("="*80)
        
        # Get directories
        # s16 textured meshes: {s16_out_dir}/textured_mesh/{texture_model}/{obj_name}/{dimension}/{cousin}_mesh.glb
        # Combinations file:   {combinations_dir}/combinations.json
        # Base (canonical) meshes: {s8_pose_out_dir}/canonical_mesh/{obj_name}.glb
        s16_result_dir = cfg.cousin_generation.out_dir
        combinations_dir = cfg.generate_cousins_combination.out_dir
        canonical_mesh_dir = os.path.join(cfg.s8_pose.out_dir, "canonical_mesh")
        texture_model = cfg.cousin_generation.texture_model
        s17_out_dir = cfg.cousin_p2p_match.out_dir
        os.makedirs(s17_out_dir, exist_ok=True)
        
        # PartField root directory
        partfield_root = os.path.join(project_root, "deps", "PartField")
        
        print(f"\nConfiguration:")
        print(f"  S16 result dir: {s16_result_dir}")
        print(f"  Combinations dir: {combinations_dir}")
        print(f"  Canonical mesh dir: {canonical_mesh_dir}")
        print(f"  Texture model: {texture_model}")
        print(f"  S17 output dir: {s17_out_dir}")
        print(f"  PartField root: {partfield_root}")
        
        # Find all cousin meshes using combinations.json and actual s16 structure
        print(f"\nScanning for cousin meshes...")
        mesh_dict = find_cousin_meshes(
            s16_result_dir=s16_result_dir,
            combinations_dir=combinations_dir,
            canonical_mesh_dir=canonical_mesh_dir,
            texture_model=texture_model,
        )
        
        if not mesh_dict:
            print("✗ No meshes found from combinations")
            return
        
        print(f"\nFound {len(mesh_dict)} base objects:")
        total_pairs = 0
        for base_name, data in mesh_dict.items():
            n_cousins = len(data['cousins'])
            total_pairs += n_cousins
            print(f"  - {base_name}: {n_cousins} cousins")
        
        print(f"\nTotal pairs to process: {total_pairs}")
        
        # Process each base-cousin pair
        successful = 0
        failed = 0
        
        for base_name, data in mesh_dict.items():
            base_mesh = data['base_mesh']
            
            if base_mesh is None:
                print(f"\n✗ Warning: No base mesh found for {base_name}, skipping")
                failed += len(data['cousins'])
                continue
            
            if not data['cousins']:
                print(f"\n✗ Warning: No cousin meshes found for {base_name}, skipping")
                continue
            
            for idx, cousin_mesh in enumerate(data['cousins']):
                # Skip if cousin and base are the same file
                if os.path.abspath(cousin_mesh) == os.path.abspath(base_mesh):
                    print(f"\n⚠️  Skipping: Cousin is same as base mesh ({os.path.basename(cousin_mesh)})")
                    continue
                
                # Extract cousin number from filename if available
                # Filenames look like: cousin_001_v1_transparent_mesh.glb
                cousin_filename = os.path.basename(cousin_mesh)
                match = re.search(r'cousin[_-]?(\d+)', cousin_filename, re.IGNORECASE)
                if match:
                    cousin_idx = int(match.group(1))
                else:
                    cousin_idx = idx
                
                success = process_mesh_pair(
                    base_mesh, cousin_mesh, base_name, cousin_idx,
                    s17_out_dir, partfield_root
                )
                
                if success:
                    successful += 1
                else:
                    failed += 1
        
        # Summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Total pairs: {total_pairs}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"\nAll results saved to: {s17_out_dir}")
        print(f"{'='*80}")
    
    run()


if __name__ == "__main__":
    main()
