# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Download HDRIs from Poly Haven for domain randomization.

This script downloads HDRI (High Dynamic Range Image) files from Poly Haven
to use for lighting randomization in IsaacLab environments.

Usage:
    python scripts/0_download_polyhaven_backgrounds.py --category indoor --resolution 4k --format hdr
"""

import requests
import os
import argparse
from pathlib import Path
HDRI_DIR = str(Path(__file__).resolve().parents[4] / "assets" / "hdr_backgrounds")

# Optional progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback: simple progress indicator
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"[INFO] {desc}...")
        return iterable


def download_polyhaven_backgrounds(
    category: str = "indoor",
    resolution: str = "4k",
    format: str = "hdr",
    max_downloads: int = None
):
    """Download HDRI backgrounds from Poly Haven.
    
    Args:
        category: Category of HDRIs to download (e.g., "indoor", "outdoor", "studio")
        resolution: Resolution to download ("1k", "2k", "4k", "8k")
        format: File format ("hdr" or "exr")
        max_downloads: Maximum number of files to download (None = all)
    """
    print(f"[INFO] Fetching HDRI list from Poly Haven API...")
    
    # Get HDRIs specifically (type=hdris filters to only HDRIs, not all assets)
    # This matches the website structure where HDRIs are a separate section
    try:
        response = requests.get("https://api.polyhaven.com/assets?type=hdris", timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch HDRIs list: {e}")
        return
    
    # Filter by category
    backgrounds_to_download = []
    for name, metadata in data.items():
        # Verify it's actually an HDRI (type: 0 = HDRIs, 1 = Textures, 2 = Models)
        if metadata.get("type") != 0:
            continue
        # Filter by category
        if "categories" in metadata and category in set(metadata["categories"]):
            backgrounds_to_download.append(name)
    
    if not backgrounds_to_download:
        print(f"[WARNING] No HDRIs found with category '{category}'")
        print(f"[INFO] Available categories: {set().union(*[m.get('categories', []) for m in data.values()])}")
        return
    
    # Limit downloads if specified
    if max_downloads is not None:
        backgrounds_to_download = backgrounds_to_download[:max_downloads]
    
    print(f"[INFO] Found {len(backgrounds_to_download)} HDRIs with category '{category}'")
    print(f"[INFO] Downloading {format.upper()} format at {resolution} resolution...")
    
    # Create save directory
    save_dir = f"{HDRI_DIR}/{category.replace(' ', '_')}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving to: {save_dir}")
    
    # Download each HDRI
    downloaded = 0
    skipped = 0
    failed = 0
    
    for background_name in tqdm(backgrounds_to_download, desc="Downloading HDRIs"):
        file_name = f"{background_name}-{resolution}"
        file_ext = format.lower()
        save_fpath = f"{save_dir}/{file_name}.{file_ext}"
        
        # Skip if already downloaded
        if os.path.exists(save_fpath):
            skipped += 1
            continue
        
        try:
            # Get file metadata
            files_url = f"https://api.polyhaven.com/files/{background_name}"
            files_response = requests.get(files_url, timeout=30)
            files_response.raise_for_status()
            files_data = files_response.json()
            
            # Check if HDRI files are available
            if "hdri" not in files_data:
                print(f"[WARNING] No HDRI files found for {background_name}, skipping...")
                failed += 1
                continue
            
            # Check if requested resolution and format are available
            if resolution not in files_data["hdri"]:
                print(f"[WARNING] Resolution {resolution} not available for {background_name}, skipping...")
                failed += 1
                continue
            
            resolution_data = files_data["hdri"][resolution]
            if format.lower() not in resolution_data:
                print(f"[WARNING] Format {format} not available for {background_name} at {resolution}, skipping...")
                failed += 1
                continue
            
            # Get download URL - API structure: files_data["hdri"][resolution][format]["url"]
            format_data = resolution_data[format.lower()]
            if "url" not in format_data:
                print(f"[WARNING] No download URL found for {background_name} ({format} at {resolution}), skipping...")
                failed += 1
                continue
            
            download_url = format_data["url"]
            
            # Download file
            file_response = requests.get(download_url, timeout=300, stream=True)  # Longer timeout for large files
            file_response.raise_for_status()
            
            # Save file
            with open(save_fpath, "wb") as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            downloaded += 1
            
        except requests.RequestException as e:
            print(f"[ERROR] Failed to download {background_name}: {e}")
            failed += 1
            continue
        except KeyError as e:
            print(f"[ERROR] Unexpected API response format for {background_name}: {e}")
            failed += 1
            continue
        except Exception as e:
            print(f"[ERROR] Unexpected error downloading {background_name}: {e}")
            failed += 1
            continue
    
    # Print summary
    print(f"\n[INFO] Download complete!")
    print(f"  - Downloaded: {downloaded}")
    print(f"  - Skipped (already exists): {skipped}")
    print(f"  - Failed: {failed}")
    print(f"  - Total: {len(backgrounds_to_download)}")
    print(f"\n[INFO] HDRIs saved to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HDRIs from Poly Haven")
    parser.add_argument(
        "--category",
        type=str,
        default="indoor",
        choices=["indoor", "outdoor", "studio"],
        help="Category of HDRIs to download (default: indoor)"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="4k",
        choices=["1k", "2k", "4k", "8k"],
        help="Resolution to download (default: 4k)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="hdr",
        choices=["hdr", "exr"],
        help="File format to download (default: hdr). Note: DoorMan repo uses .hdr format for dome light randomization."
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=None,
        help="Maximum number of files to download (default: all)"
    )
    
    args = parser.parse_args()
    
    download_polyhaven_backgrounds(
        category=args.category,
        resolution=args.resolution,
        format=args.format,
        max_downloads=args.max_downloads
    )