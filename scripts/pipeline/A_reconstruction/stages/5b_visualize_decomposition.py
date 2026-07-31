# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualize the output from 5_decompose_scene.py

Should be run from `simfoundry` env

This script provides a GUI to visualize the iterative scene decomposition process,
showing:
- RGB scene image at each step (before and after object removal)
- Segmented object image at each step
- Labeled category and validity status for each removed object

Interactive features:
- Toggle valid/invalid labels (saves to original JSON files)
- Mark iterations for rerun with optional category override
- Specify whether to rerun just that step or all downstream steps
- Click on image to specify a point prompt for SAM segmentation

Usage:
    python 5b_visualize_decomposition.py [optional hydra overrides]
    
Example:
    python 5b_visualize_decomposition.py scene_name=my_scene
"""
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.widgets import Button, Slider, TextBox, CheckButtons
from omegaconf import OmegaConf
from PIL import Image
from simfoundry import CFG_DIR

# Set up logger
logger = logging.getLogger(__name__)

### At the start of every script, we cd into the scripts/config directory
scripts_dir = os.path.dirname(os.path.abspath(__file__))
cfg_dir = CFG_DIR
os.chdir(cfg_dir)

# Feedback file name (saved in s5_scene output directory)
FEEDBACK_FILENAME = "rerun_feedback.json"


class DecompositionVisualizer:
    """
    GUI visualizer for scene decomposition results.
    
    Displays tiled views of:
    - Scene RGB before removal (with annotations)
    - Scene RGB after removal
    - Segmented object (with and without background)
    - Object category and validity info
    
    Interactive features:
    - Toggle valid/invalid labels
    - Mark iterations for rerun
    - Specify category overrides
    - Click on image to specify point prompt for SAM segmentation
    """
    
    def __init__(self, decomposition_dir: str):
        """
        Initialize the visualizer.
        
        Args:
            decomposition_dir: Path to the s5_scene output directory
        """
        self.decomposition_dir = Path(decomposition_dir)
        self.iterations = self._load_iterations()
        self.current_idx = 0
        self.fig = None
        self.axes = None
        
        # Track rerun requests: {iter_idx: {"rerun": bool, "category": str or None, "rerun_downstream": bool, "point_prompt": [x, y] or None}}
        self.rerun_requests = {}
        
        # Point selection mode
        self.point_selection_mode = False
        self.selected_point = None  # (x, y) coordinates on the image
        self.point_marker = None    # matplotlib artist for the point marker
        
        # UI elements
        self.slider = None
        self.validity_btn = None
        self.rerun_btn = None
        self.downstream_check = None
        self.category_textbox = None
        self.save_btn = None
        self.clear_btn = None
        self.point_btn = None
        self.clear_point_btn = None
        
        if len(self.iterations) == 0:
            raise ValueError(f"No decomposition iterations found in {decomposition_dir}")
        
        logger.info(f"Found {len(self.iterations)} decomposition iterations")
        
        # Load existing feedback if present
        self._load_existing_feedback()
        
    def _load_iterations(self) -> List[Dict]:
        """
        Load all iteration data from the decomposition directory.
        
        Returns:
            List of dictionaries containing iteration data
        """
        iterations = []
        
        # Get list of iteration indices from obj_cat_list directory
        obj_cat_dir = self.decomposition_dir / "obj_cat_list"
        if not obj_cat_dir.exists():
            logger.error(f"obj_cat_list directory not found at {obj_cat_dir}")
            return iterations
            
        iter_files = sorted(obj_cat_dir.glob("iter_*.json"), 
                           key=lambda x: int(x.stem.split("_")[1]))
        
        for iter_file in iter_files:
            iter_idx = int(iter_file.stem.split("_")[1])
            iter_data = self._load_iteration(iter_idx)
            if iter_data is not None:
                iterations.append(iter_data)
                
        return iterations
    
    def _load_iteration(self, iter_idx: int) -> Optional[Dict]:
        """
        Load data for a single iteration.
        
        Args:
            iter_idx: Iteration index
            
        Returns:
            Dictionary containing iteration data, or None if loading fails
        """
        try:
            # Load metadata JSON
            json_path = self.decomposition_dir / "obj_cat_list" / f"iter_{iter_idx}.json"
            with open(json_path, "r") as f:
                metadata = json.load(f)
            
            # Build paths to images
            iter_data = {
                "iter_idx": iter_idx,
                "metadata": metadata,
                "json_path": str(json_path),
                "images": {}
            }
            
            # Define image paths to load
            image_paths = {
                "detected_phrases": self.decomposition_dir / "detected_phrases" / f"iter_{iter_idx}.png",
                "pre_removal": self.decomposition_dir / "pre_object_removal" / f"iter_{iter_idx}.png",
                "post_removal": self.decomposition_dir / "post_object_removal" / f"iter_{iter_idx}.png",
                "masked_object": self.decomposition_dir / "masked_object" / f"iter_{iter_idx}.png",
                "masked_object_focus": self.decomposition_dir / "masked_object_focus" / f"iter_{iter_idx}.png",
                "masked_object_background": self.decomposition_dir / "masked_object_background" / f"iter_{iter_idx}.png",
                "removal_mask": self.decomposition_dir / "removal_mask" / f"iter_{iter_idx}.png",
            }
            
            # Load each image if it exists
            for img_name, img_path in image_paths.items():
                if img_path.exists():
                    iter_data["images"][img_name] = np.array(Image.open(img_path))
                else:
                    logger.warning(f"Image not found: {img_path}")
                    iter_data["images"][img_name] = None
                    
            # Also load source image if this is iteration 0
            if iter_idx == 0:
                source_path = self.decomposition_dir / "source_padded_resized_upsampled.png"
                if source_path.exists():
                    iter_data["images"]["source"] = np.array(Image.open(source_path))
                else:
                    # Try non-upsampled version
                    source_path = self.decomposition_dir / "source_padded_resized.png"
                    if source_path.exists():
                        iter_data["images"]["source"] = np.array(Image.open(source_path))
                        
            return iter_data
            
        except Exception as e:
            logger.error(f"Error loading iteration {iter_idx}: {e}")
            return None
    
    def _load_existing_feedback(self):
        """Load existing feedback file if present."""
        feedback_path = self.decomposition_dir / FEEDBACK_FILENAME
        if feedback_path.exists():
            try:
                with open(feedback_path, "r") as f:
                    feedback = json.load(f)
                self.rerun_requests = {int(k): v for k, v in feedback.get("rerun_requests", {}).items()}
                logger.info(f"Loaded existing feedback with {len(self.rerun_requests)} rerun requests")
                
                # Load point prompt for current iteration if exists
                iter_data = self.iterations[self.current_idx] if self.iterations else None
                if iter_data:
                    iter_idx = iter_data["iter_idx"]
                    if iter_idx in self.rerun_requests:
                        point = self.rerun_requests[iter_idx].get("point_prompt")
                        if point:
                            self.selected_point = tuple(point)
            except Exception as e:
                logger.warning(f"Failed to load existing feedback: {e}")
    
    def _format_metadata_text(self, metadata: Dict, iter_idx: int) -> Tuple[str, str]:
        """
        Format metadata as display text.
        
        Args:
            metadata: Metadata dictionary
            iter_idx: Current iteration index
            
        Returns:
            Tuple of (formatted string for display, validity color)
        """
        lines = []
        
        # Removed object info
        removed_obj = metadata.get("removed_obj_phrase", "Unknown")
        is_valid = metadata.get("is_valid_removed_obj", None)
        validity_str = "Valid" if is_valid else "Invalid" if is_valid is not None else "Unknown"
        validity_color = "green" if is_valid else "red" if is_valid is not None else "gray"
        
        lines.append(f"Removed Object: {removed_obj}")
        lines.append(f"Validity: {validity_str}")
        
        # Show rerun status if marked
        if iter_idx in self.rerun_requests:
            req = self.rerun_requests[iter_idx]
            lines.append(f"\n** MARKED FOR RERUN **")
            if req.get("category"):
                lines.append(f"  Category: {req['category']}")
            if req.get("point_prompt"):
                pt = req["point_prompt"]
                lines.append(f"  Point: ({pt[0]:.0f}, {pt[1]:.0f})")
            if req.get("rerun_downstream"):
                lines.append(f"  + All downstream")
        
        # VLM categories detected
        vlm_cats = metadata.get("vlm_categories", [])
        if vlm_cats:
            lines.append(f"\nVLM Categories ({len(vlm_cats)}):")
            for cat in vlm_cats[:5]:  # Show first 5
                lines.append(f"  - {cat}")
            if len(vlm_cats) > 5:
                lines.append(f"  ... and {len(vlm_cats) - 5} more")
                
        # Pruned phrases
        pruned = metadata.get("pruned_phrases", [])
        if pruned:
            lines.append(f"\nPruned Phrases ({len(pruned)}):")
            for phrase in pruned[:5]:
                lines.append(f"  - {phrase}")
            if len(pruned) > 5:
                lines.append(f"  ... and {len(pruned) - 5} more")
                
        return "\n".join(lines), validity_color
    
    def _update_display(self):
        """Update the display with current iteration data."""
        if self.fig is None:
            return
            
        iter_data = self.iterations[self.current_idx]
        iter_idx = iter_data["iter_idx"]
        metadata = iter_data["metadata"]
        images = iter_data["images"]
        
        # Clear main display axes (not control axes)
        for ax in self.axes.flat:
            ax.clear()
            ax.axis("off")
            
        # Update title
        rerun_marker = " [MARKED FOR RERUN]" if iter_idx in self.rerun_requests else ""
        self.fig.suptitle(
            f"Scene Decomposition - Iteration {self.current_idx + 1}/{len(self.iterations)} "
            f"(Index: {iter_idx}){rerun_marker}",
            fontsize=14,
            fontweight="bold",
            color="red" if iter_idx in self.rerun_requests else "black"
        )
        
        # Row 0: Scene images
        # Detected phrases / annotated - this is the clickable image for point selection
        if images.get("detected_phrases") is not None:
            self.axes[0, 0].imshow(images["detected_phrases"])
            title = "Detected Objects"
            if self.point_selection_mode:
                title += " [CLICK TO SELECT POINT]"
            self.axes[0, 0].set_title(title, fontsize=10, 
                                       color="blue" if self.point_selection_mode else "black")
            
            # Load point from rerun request if it exists for this iteration
            if iter_idx in self.rerun_requests:
                point = self.rerun_requests[iter_idx].get("point_prompt")
                if point:
                    self.selected_point = tuple(point)
            else:
                self.selected_point = None
            
            # Draw selected point marker if exists
            if self.selected_point is not None:
                x, y = self.selected_point
                # Draw crosshair marker
                self.axes[0, 0].plot(x, y, 'r+', markersize=20, markeredgewidth=3)
                self.axes[0, 0].plot(x, y, 'yo', markersize=12, markerfacecolor='none', markeredgewidth=2)
                self.axes[0, 0].annotate(f'({x:.0f}, {y:.0f})', (x, y), 
                                          xytext=(10, 10), textcoords='offset points',
                                          fontsize=9, color='red',
                                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Pre-removal
        if images.get("pre_removal") is not None:
            self.axes[0, 1].imshow(images["pre_removal"])
            self.axes[0, 1].set_title("Pre-Removal (with annotation)", fontsize=10)
        
        # Post-removal
        if images.get("post_removal") is not None:
            self.axes[0, 2].imshow(images["post_removal"])
            self.axes[0, 2].set_title("Post-Removal", fontsize=10)
            
        # Row 1: Object images
        # Masked object (no background)
        if images.get("masked_object") is not None:
            self.axes[1, 0].imshow(images["masked_object"])
            self.axes[1, 0].set_title("Segmented Object (isolated)", fontsize=10)
        
        # Masked object with background
        if images.get("masked_object_background") is not None:
            self.axes[1, 1].imshow(images["masked_object_background"])
            self.axes[1, 1].set_title("Object with Background", fontsize=10)
        
        # Removal mask
        if images.get("removal_mask") is not None:
            self.axes[1, 2].imshow(images["removal_mask"], cmap="gray")
            self.axes[1, 2].set_title("Removal Mask", fontsize=10)
        
        # Row 2: Metadata and focused view
        # Focused object view
        if images.get("masked_object_focus") is not None:
            self.axes[2, 0].imshow(images["masked_object_focus"])
            self.axes[2, 0].set_title("Object (Focused)", fontsize=10)
        
        # Metadata text
        metadata_text, validity_color = self._format_metadata_text(metadata, iter_idx)
        self.axes[2, 1].text(
            0.05, 0.95, metadata_text,
            transform=self.axes[2, 1].transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        )
        self.axes[2, 1].set_title("Iteration Info", fontsize=10)
        
        # Validity indicator
        is_valid = metadata.get("is_valid_removed_obj", None)
        validity_text = "VALID" if is_valid else "INVALID" if is_valid is not None else "UNKNOWN"
        self.axes[2, 2].text(
            0.5, 0.5, validity_text,
            transform=self.axes[2, 2].transAxes,
            fontsize=24,
            fontweight="bold",
            color=validity_color,
            ha="center",
            va="center"
        )
        removed_obj = metadata.get("removed_obj_phrase", "Unknown")
        self.axes[2, 2].text(
            0.5, 0.25, f'"{removed_obj}"',
            transform=self.axes[2, 2].transAxes,
            fontsize=12,
            ha="center",
            va="center",
            style="italic"
        )
        self.axes[2, 2].set_title("Validity Status", fontsize=10)
        
        # Update button labels
        self._update_button_labels()
        
        self.fig.canvas.draw_idle()
    
    def _update_button_labels(self):
        """Update button labels based on current state."""
        iter_data = self.iterations[self.current_idx]
        iter_idx = iter_data["iter_idx"]
        metadata = iter_data["metadata"]
        
        # Update validity button
        is_valid = metadata.get("is_valid_removed_obj", None)
        if is_valid:
            self.validity_btn.label.set_text("Mark Invalid")
        else:
            self.validity_btn.label.set_text("Mark Valid")
        
        # Update rerun button
        if iter_idx in self.rerun_requests:
            self.rerun_btn.label.set_text("Cancel Rerun")
            self.rerun_btn.color = "lightcoral"
            self.rerun_btn.hovercolor = "coral"
        else:
            self.rerun_btn.label.set_text("Mark for Rerun")
            self.rerun_btn.color = "lightblue"
            self.rerun_btn.hovercolor = "deepskyblue"
    
    def _on_prev(self, event):
        """Handle previous button click."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.slider.set_val(self.current_idx)
    
    def _on_next(self, event):
        """Handle next button click."""
        if self.current_idx < len(self.iterations) - 1:
            self.current_idx += 1
            self.slider.set_val(self.current_idx)
    
    def _on_slider_change(self, val):
        """Handle slider value change."""
        self.current_idx = int(val)
        self._update_display()
    
    def _on_toggle_validity(self, event):
        """Handle validity toggle button click."""
        iter_data = self.iterations[self.current_idx]
        iter_idx = iter_data["iter_idx"]
        metadata = iter_data["metadata"]
        json_path = iter_data["json_path"]
        
        # Toggle validity
        current_validity = metadata.get("is_valid_removed_obj", False)
        new_validity = not current_validity
        metadata["is_valid_removed_obj"] = new_validity
        
        # Save to JSON file
        try:
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Updated validity for iteration {iter_idx}: {new_validity}")
            print(f"[Saved] Iteration {iter_idx} validity -> {'Valid' if new_validity else 'Invalid'}")
        except Exception as e:
            logger.error(f"Failed to save validity update: {e}")
            print(f"[Error] Failed to save: {e}")
        
        self._update_display()
    
    def _on_toggle_rerun(self, event):
        """Handle rerun toggle button click."""
        iter_data = self.iterations[self.current_idx]
        iter_idx = iter_data["iter_idx"]
        
        if iter_idx in self.rerun_requests:
            # Remove rerun request
            del self.rerun_requests[iter_idx]
            self.selected_point = None
            logger.info(f"Removed rerun request for iteration {iter_idx}")
            print(f"[Removed] Rerun request for iteration {iter_idx}")
        else:
            # Add rerun request
            category = self.category_textbox.text.strip() if self.category_textbox.text.strip() else None
            rerun_downstream = self.downstream_var
            
            # Include point prompt if selected
            point_prompt = list(self.selected_point) if self.selected_point else None
            
            self.rerun_requests[iter_idx] = {
                "rerun": True,
                "category": category,
                "rerun_downstream": rerun_downstream,
                "point_prompt": point_prompt,
            }
            logger.info(f"Added rerun request for iteration {iter_idx}: category={category}, point={point_prompt}, downstream={rerun_downstream}")
            print(f"[Added] Rerun request for iteration {iter_idx}")
            if category:
                print(f"        Category override: {category}")
            if point_prompt:
                print(f"        Point prompt: ({point_prompt[0]:.0f}, {point_prompt[1]:.0f})")
            if rerun_downstream:
                print(f"        Will rerun all downstream iterations")
        
        self._update_display()
    
    def _on_downstream_toggle(self, label):
        """Handle downstream checkbox toggle."""
        self.downstream_var = not self.downstream_var
        
        # Update existing request if present
        iter_data = self.iterations[self.current_idx]
        iter_idx = iter_data["iter_idx"]
        if iter_idx in self.rerun_requests:
            self.rerun_requests[iter_idx]["rerun_downstream"] = self.downstream_var
    
    def _on_category_submit(self, text):
        """Handle category textbox submit."""
        iter_data = self.iterations[self.current_idx]
        iter_idx = iter_data["iter_idx"]
        
        # Update existing request if present
        if iter_idx in self.rerun_requests:
            category = text.strip() if text.strip() else None
            self.rerun_requests[iter_idx]["category"] = category
            print(f"[Updated] Category for iteration {iter_idx}: {category}")
    
    def _on_save_feedback(self, event):
        """Save all feedback to file."""
        feedback_path = self.decomposition_dir / FEEDBACK_FILENAME
        
        feedback = {
            "rerun_requests": {str(k): v for k, v in self.rerun_requests.items()},
            "total_iterations": len(self.iterations),
        }
        
        try:
            with open(feedback_path, "w") as f:
                json.dump(feedback, f, indent=4)
            logger.info(f"Saved feedback to {feedback_path}")
            print(f"\n{'='*50}")
            print(f"[Saved] Feedback saved to: {feedback_path}")
            print(f"        {len(self.rerun_requests)} rerun request(s)")
            print(f"{'='*50}\n")
            
            # Print summary
            if self.rerun_requests:
                print("Rerun requests:")
                for iter_idx, req in sorted(self.rerun_requests.items()):
                    iter_data = next((it for it in self.iterations if it["iter_idx"] == iter_idx), None)
                    if iter_data:
                        obj_name = iter_data["metadata"].get("removed_obj_phrase", "Unknown")
                        print(f"  Iter {iter_idx}: {obj_name}")
                        if req.get("category"):
                            print(f"           Category override: {req['category']}")
                        if req.get("point_prompt"):
                            pt = req["point_prompt"]
                            print(f"           Point prompt: ({pt[0]:.0f}, {pt[1]:.0f})")
                        if req.get("rerun_downstream"):
                            print(f"           + All downstream iterations")
                print()
                print("To apply these changes, run:")
                print("  python 5_decompose_scene.py s5_scene.use_feedback=true")
                print()
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
            print(f"[Error] Failed to save feedback: {e}")
    
    def _on_clear_feedback(self, event):
        """Clear all rerun requests."""
        self.rerun_requests.clear()
        self.selected_point = None
        self.point_selection_mode = False
        logger.info("Cleared all rerun requests")
        print("[Cleared] All rerun requests removed")
        self._update_display()
    
    def _on_toggle_point_mode(self, event):
        """Toggle point selection mode."""
        self.point_selection_mode = not self.point_selection_mode
        if self.point_selection_mode:
            print("[Point Mode] Click on the 'Detected Objects' image to select a point for SAM segmentation")
            self.point_btn.color = "plum"
            self.point_btn.hovercolor = "violet"
            self.point_btn.label.set_text("Cancel Select")
        else:
            print("[Point Mode] Point selection cancelled")
            self.point_btn.color = "lavender"
            self.point_btn.hovercolor = "plum"
            self.point_btn.label.set_text("Select Point")
        self._update_display()
    
    def _on_clear_point(self, event):
        """Clear the selected point for current iteration."""
        iter_data = self.iterations[self.current_idx]
        iter_idx = iter_data["iter_idx"]
        
        self.selected_point = None
        self.point_selection_mode = False
        self.point_btn.color = "lavender"
        self.point_btn.hovercolor = "plum"
        self.point_btn.label.set_text("Select Point")
        
        # Update rerun request if exists
        if iter_idx in self.rerun_requests:
            self.rerun_requests[iter_idx]["point_prompt"] = None
            print(f"[Cleared] Point prompt for iteration {iter_idx}")
        
        self._update_display()
    
    def _on_image_click(self, event):
        """Handle click on the figure for point selection."""
        if not self.point_selection_mode:
            return
        
        # Check if click is within the detected_phrases axes (row 0, col 0)
        if event.inaxes != self.axes[0, 0]:
            return
        
        # Get click coordinates
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # Store the selected point
        self.selected_point = (x, y)
        
        iter_data = self.iterations[self.current_idx]
        iter_idx = iter_data["iter_idx"]
        
        print(f"[Point Selected] ({x:.0f}, {y:.0f}) for iteration {iter_idx}")
        
        # Update existing rerun request if present
        if iter_idx in self.rerun_requests:
            self.rerun_requests[iter_idx]["point_prompt"] = [x, y]
            print(f"        Updated point in existing rerun request")
        else:
            print(f"        Note: Click 'Mark for Rerun' to include this point in a rerun request")
        
        # Exit point selection mode
        self.point_selection_mode = False
        self.point_btn.color = "lavender"
        self.point_btn.hovercolor = "plum"
        self.point_btn.label.set_text("Select Point")
        
        self._update_display()
    
    def show(self):
        """
        Display the visualization GUI.
        """
        # Create figure and axes
        self.fig, self.axes = plt.subplots(3, 3, figsize=(16, 14))
        plt.subplots_adjust(bottom=0.22, top=0.92, hspace=0.3, wspace=0.2)
        
        # Initialize downstream variable
        self.downstream_var = False
        
        # Create navigation controls
        ax_prev = plt.axes([0.15, 0.14, 0.08, 0.03])
        ax_next = plt.axes([0.77, 0.14, 0.08, 0.03])
        ax_slider = plt.axes([0.25, 0.14, 0.5, 0.02])
        
        btn_prev = Button(ax_prev, "◀ Previous")
        btn_prev.on_clicked(self._on_prev)
        
        btn_next = Button(ax_next, "Next ▶")
        btn_next.on_clicked(self._on_next)
        
        self.slider = Slider(
            ax_slider, "Iteration", 
            0, len(self.iterations) - 1, 
            valinit=0, 
            valstep=1,
            valfmt="%d"
        )
        self.slider.on_changed(self._on_slider_change)
        
        # Create validity toggle button
        ax_validity = plt.axes([0.15, 0.09, 0.12, 0.03])
        self.validity_btn = Button(ax_validity, "Toggle Validity", color="lightyellow", hovercolor="gold")
        self.validity_btn.on_clicked(self._on_toggle_validity)
        
        # Create rerun toggle button
        ax_rerun = plt.axes([0.29, 0.09, 0.12, 0.03])
        self.rerun_btn = Button(ax_rerun, "Mark for Rerun", color="lightblue", hovercolor="deepskyblue")
        self.rerun_btn.on_clicked(self._on_toggle_rerun)
        
        # Create downstream checkbox
        ax_downstream = plt.axes([0.43, 0.085, 0.15, 0.04])
        ax_downstream.set_facecolor("white")
        self.downstream_check = CheckButtons(ax_downstream, ["Rerun downstream"], [False])
        self.downstream_check.on_clicked(self._on_downstream_toggle)
        
        # Create category textbox
        ax_cat_label = plt.axes([0.59, 0.09, 0.08, 0.03])
        ax_cat_label.axis("off")
        ax_cat_label.text(0.5, 0.5, "Category:", ha="center", va="center", fontsize=10)
        
        ax_category = plt.axes([0.67, 0.09, 0.18, 0.03])
        self.category_textbox = TextBox(ax_category, "", initial="")
        self.category_textbox.on_submit(self._on_category_submit)
        
        # Create point selection controls
        ax_point = plt.axes([0.15, 0.045, 0.12, 0.03])
        self.point_btn = Button(ax_point, "Select Point", color="lavender", hovercolor="plum")
        self.point_btn.on_clicked(self._on_toggle_point_mode)
        
        ax_clear_point = plt.axes([0.28, 0.045, 0.1, 0.03])
        self.clear_point_btn = Button(ax_clear_point, "Clear Point", color="lightyellow", hovercolor="khaki")
        self.clear_point_btn.on_clicked(self._on_clear_point)
        
        # Create save and clear buttons
        ax_save = plt.axes([0.42, 0.01, 0.15, 0.04])
        self.save_btn = Button(ax_save, "Save Feedback", color="lightgreen", hovercolor="limegreen")
        self.save_btn.on_clicked(self._on_save_feedback)
        
        ax_clear = plt.axes([0.59, 0.01, 0.15, 0.04])
        self.clear_btn = Button(ax_clear, "Clear All Requests", color="mistyrose", hovercolor="lightcoral")
        self.clear_btn.on_clicked(self._on_clear_feedback)
        
        # Connect click event for point selection
        self.fig.canvas.mpl_connect('button_press_event', self._on_image_click)
        
        # Initial display
        self._update_display()
        
        # Show summary
        print("\n" + "=" * 60)
        print("DECOMPOSITION SUMMARY")
        print("=" * 60)
        for i, iter_data in enumerate(self.iterations):
            metadata = iter_data["metadata"]
            removed_obj = metadata.get("removed_obj_phrase", "Unknown")
            is_valid = metadata.get("is_valid_removed_obj", None)
            validity_str = "✓ Valid" if is_valid else "✗ Invalid" if is_valid is not None else "? Unknown"
            rerun_marker = " [RERUN]" if iter_data["iter_idx"] in self.rerun_requests else ""
            print(f"  Iter {iter_data['iter_idx']:2d}: {removed_obj:30s} [{validity_str}]{rerun_marker}")
        print("=" * 60)
        print(f"Total iterations: {len(self.iterations)}")
        valid_count = sum(1 for it in self.iterations if it["metadata"].get("is_valid_removed_obj", False))
        print(f"Valid objects: {valid_count}/{len(self.iterations)}")
        if self.rerun_requests:
            print(f"Pending rerun requests: {len(self.rerun_requests)}")
        print("=" * 60)
        print("\nControls:")
        print("  - Use slider or buttons to navigate iterations")
        print("  - 'Toggle Validity' - Mark current iteration as valid/invalid (saves immediately)")
        print("  - 'Mark for Rerun' - Mark current iteration for rerun")
        print("  - 'Rerun downstream' - Also rerun all iterations after this one")
        print("  - 'Category' - Optional category override for rerun")
        print("  - 'Select Point' - Enter point selection mode, then click on the image")
        print("                     to specify a SAM point prompt for segmentation")
        print("  - 'Clear Point' - Remove the selected point for current iteration")
        print("  - 'Save Feedback' - Save rerun requests to file")
        print("  - 'Clear All Requests' - Remove all rerun requests")
        print("=" * 60 + "\n")
        
        plt.show()


def create_tiled_summary(decomposition_dir: str, output_path: Optional[str] = None):
    """
    Create a static tiled summary image of all decomposition steps.
    
    Args:
        decomposition_dir: Path to the s5_scene output directory
        output_path: Optional path to save the summary image
    """
    viz = DecompositionVisualizer(decomposition_dir)
    n_iters = len(viz.iterations)
    
    if n_iters == 0:
        logger.error("No iterations to visualize")
        return
    
    # Create tiled figure
    fig, axes = plt.subplots(n_iters, 4, figsize=(16, 4 * n_iters))
    if n_iters == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle("Scene Decomposition Summary", fontsize=16, fontweight="bold")
    
    for i, iter_data in enumerate(viz.iterations):
        metadata = iter_data["metadata"]
        images = iter_data["images"]
        iter_idx = iter_data["iter_idx"]
        
        # Column 0: Pre-removal scene
        if images.get("pre_removal") is not None:
            axes[i, 0].imshow(images["pre_removal"])
        axes[i, 0].set_title(f"Iter {iter_idx}: Scene", fontsize=9)
        axes[i, 0].axis("off")
        
        # Column 1: Segmented object
        if images.get("masked_object") is not None:
            axes[i, 1].imshow(images["masked_object"])
        axes[i, 1].set_title("Segmented Object", fontsize=9)
        axes[i, 1].axis("off")
        
        # Column 2: Post-removal
        if images.get("post_removal") is not None:
            axes[i, 2].imshow(images["post_removal"])
        axes[i, 2].set_title("After Removal", fontsize=9)
        axes[i, 2].axis("off")
        
        # Column 3: Info
        removed_obj = metadata.get("removed_obj_phrase", "Unknown")
        is_valid = metadata.get("is_valid_removed_obj", None)
        validity_str = "VALID" if is_valid else "INVALID" if is_valid is not None else "UNKNOWN"
        validity_color = "green" if is_valid else "red" if is_valid is not None else "gray"
        
        axes[i, 3].text(
            0.5, 0.6, f'"{removed_obj}"',
            transform=axes[i, 3].transAxes,
            fontsize=10,
            ha="center",
            va="center",
            wrap=True
        )
        axes[i, 3].text(
            0.5, 0.3, validity_str,
            transform=axes[i, 3].transAxes,
            fontsize=14,
            fontweight="bold",
            color=validity_color,
            ha="center",
            va="center"
        )
        axes[i, 3].set_title("Category & Status", fontsize=9)
        axes[i, 3].axis("off")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Summary saved to {output_path}")
    
    plt.show()


@hydra.main(config_name="real2sim_cfg", config_path=CFG_DIR, version_base="1.3")
def main(cfg):
    """Main entry point for decomposition visualization."""
    decomposition_dir = cfg.s5_scene.out_dir
    
    logger.info("=" * 60)
    logger.info("Scene Decomposition Visualizer")
    logger.info(f"Decomposition directory: {decomposition_dir}")
    logger.info("=" * 60)
    
    if not Path(decomposition_dir).exists():
        logger.error(f"Decomposition directory not found: {decomposition_dir}")
        logger.error("Please run 5_decompose_scene.py first.")
        return
    
    # Check if there are any results
    obj_cat_dir = Path(decomposition_dir) / "obj_cat_list"
    if not obj_cat_dir.exists() or len(list(obj_cat_dir.glob("iter_*.json"))) == 0:
        logger.error("No decomposition results found. Please run 5_decompose_scene.py first.")
        return
    
    # Create and show visualizer
    try:
        visualizer = DecompositionVisualizer(decomposition_dir)
        visualizer.show()
    except Exception as e:
        logger.error(f"Error creating visualizer: {e}")
        raise


if __name__ == "__main__":
    main()
