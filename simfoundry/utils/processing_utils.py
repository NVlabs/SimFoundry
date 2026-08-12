# Copyright (c) 2024 the ACDC authors (Stanford Vision and Learning Lab)
# Licensed under the Apache License, Version 2.0.
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This file is derived from the ACDC / digital-cousins project
# (https://github.com/cremebrule/digital-cousins). NVIDIA modifications are licensed
# under Apache-2.0; the adapted upstream portions remain subject to the terms above.

import numpy as np
import torch as th
import cv2
import json
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPoint
import simfoundry.utils.transform_utils as T
from simfoundry.utils.python_utils import atomic_output_path
import torch
from typing import List
from torchvision.ops import box_convert
import supervision as sv
import multiprocessing
from typing import Optional, Tuple, Any
import re
from pathlib import Path

REMBG_CPU_PROVIDERS = ["CPUExecutionProvider"]


def annotate(image_source: np.ndarray, boxes_xyxy: np.ndarray, phrases: List[str], logits=None) -> np.ndarray:
    """
    This function annotates an image with bounding boxes and labels.

    Args:
    image_source (np.ndarray): The source image to be annotated.
    boxes_xyxy (np.ndarray): An array containing (xyxy)-ordered pixel bounding box coordinates.
    phrases (List[str] or None): A list of labels for each bounding box.
                            If None, phrases will not be drawn
    logits (List[str] or None): A list of logits for each phrase. If None, will not be shown

    Returns:
    np.ndarray: The annotated image.
    """
    import supervision as sv
    h, w, _ = image_source.shape
    detections = sv.Detections(xyxy=boxes_xyxy)

    labels = phrases if phrases else None
    if labels is not None and logits is not None:
        labels = [f"{phrase}:{logit:.2f}" for phrase, logit in zip(phrases, logits)]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame


def get_num_pixels_with_color(image, color, tolerance=0):
    assert isinstance(image, np.ndarray) and image.dtype == np.uint8
    return np.all(np.isclose(image, np.array(color).reshape(1, 1, 3), atol=tolerance), axis=-1).sum()


def select_best_generated_background_infill_image(original, mask, img_options, tiebreaker="in", blur_k=0, grad_threshold=10.0, erode_k=0):
    assert isinstance(original, np.ndarray) and original.dtype == np.uint8 and original.ndim == 3 and original.shape[-1] == 3
    assert isinstance(mask, np.ndarray) and mask.dtype == bool
    if blur_k > 0:
        original = cv2.GaussianBlur(original, (blur_k, blur_k), 0)
    results = dict()
    for i, option in enumerate(img_options):
        assert isinstance(option, np.ndarray) and option.dtype == np.uint8
        if blur_k > 0:
            option = cv2.GaussianBlur(option, (blur_k, blur_k), 0)

        # gx, gy = np.gradient(original)
        # original_grad_magnitudes = np.linalg.norm(gx, axis=-1) + np.linalg.norm(gy, axis=-1)
        original_grad_magnitudes = np.linalg.norm(np.linalg.norm(np.stack(np.gradient(original, axis=(0,1)), axis=0), axis=-1), axis=0)
        option_grad_magnitudes = np.linalg.norm(np.linalg.norm(np.stack(np.gradient(option, axis=(0,1)), axis=0), axis=-1), axis=0)
        original_lines = np.where(original_grad_magnitudes > grad_threshold, True, False)
        option_lines = np.where(option_grad_magnitudes > grad_threshold, True, False)
        # Image.fromarray(original).show()
        # Image.fromarray(option).show()
        # Image.fromarray(original_lines.astype(np.uint8) * 100).show()
        # Image.fromarray(option_lines.astype(np.uint8) * 100).show()
        # diff = np.linalg.norm((original - option) / 255.0, axis=-1)
        diff = original_lines ^ option_lines

        if erode_k > 0:
            diff = erode_mask(diff.astype(np.uint8) * 255, kernel_size=erode_k).astype(bool)

        # Image.fromarray(diff.astype(np.uint8) * 100).show()
        # diff = np.linalg.norm((original - option) / 255.0, axis=-1) > 0.5
        in_mask_diff_avg = np.mean(diff[mask])
        out_mask_diff_avg = np.mean(diff[~mask])
        results[i] = {
            "img": option,
            "diff": diff,
            "in_score": in_mask_diff_avg,           # higher is better, so score == diff
            "out_score": 1 / out_mask_diff_avg,         # lower is better, so score == 1/diff
        }
        print(f"in: {in_mask_diff_avg}, out: {out_mask_diff_avg}")
    # Evaluate scores -- point based system, N / ... 2 / 1 points assigned for 1st / 2nd / ... Nth place for both
    # in / out mask diff avg

    # TODO: Update scoring paradigm to properly handle in / out scores separately (use discrete ranking system, not continuous scores)
    in_score_ranking = sorted(results, key=lambda k: results[k]["in_score"])
    out_score_ranking = sorted(results, key=lambda k: results[k]["out_score"])
    scores = {idx: 0 for idx in results}
    for score, idx in enumerate(in_score_ranking):
        scores[idx] += score
    for score, idx in enumerate(out_score_ranking):
        scores[idx] += score
    sorted_score_ranking = sorted(scores, key=lambda k: scores[k])
    top_score_idxs = [idx for idx in sorted_score_ranking if scores[idx] == scores[sorted_score_ranking[-1]]]
    print(f"in score ranking: {in_score_ranking}")
    print(f"out score ranking: {out_score_ranking}")
    print(f"scores: {scores}")
    print(f"sorted score rankings: {sorted_score_ranking}")
    print(f"top score idxs: {top_score_idxs}")
    if tiebreaker == "in":
        best_idx = sorted(top_score_idxs, key=lambda k: in_score_ranking[k])[-1]
    else:
        best_idx = sorted(top_score_idxs, key=lambda k: out_score_ranking[k])[-1]
    # best_idx = None
    # best_in_score = None
    # best_out_score = None
    # for idx, result in results.items():
    #     if best_idx is None:
    #         best_idx = idx
    #         best_score = result["in_score"] + result["out_score"]
    #         continue
    #     option_score = result["in_score"] + result["out_score"]
    #     if (option_score > best_score) or ((option_score == best_score) and (result[f"{tiebreaker}_score"] > results[best_idx][f"{tiebreaker}_score"])):
    #         best_idx = idx
    #         best_score = option_score

    return best_idx, results[best_idx]["img"]


def pad_image_to_ratio(image, target_ratio, padding_color=(0, 0, 0), return_padding_size=False):
    """
    Pads an image to a given aspect ratio while maintaining the original aspect ratio.

    Args:
        image (np.array): The input image (OpenCV format).
        target_ratio (float): The desired aspect ratio (width / height).
        padding_color (tuple): The RGB color for the padding (e.g., (0, 0, 0) for black)
        return_padding_size (bool): Whether to return the padding size or not

    Returns:
            np.array or 2-tuple: The padded image if return_padding_size is False else (img, (delta_w, delta_h)) tuple
    """
    h, w = image.shape[:2]
    current_ratio = w / h

    if current_ratio == target_ratio:
        if return_padding_size:
            return image, (0, 0)
        else:
            return image  # No padding needed

    if current_ratio > target_ratio:  # Image is wider than target ratio
        new_w = w
        new_h = int(w / target_ratio)
    else:  # Image is taller than target ratio
        new_h = h
        new_w = int(h * target_ratio)

    # Calculate padding
    delta_w = new_w - w
    delta_h = new_h - h

    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=padding_color)
    if return_padding_size:
        return padded_image, (delta_w, delta_h)
    else:
        return padded_image


def unpad_image(image, delta_w, delta_h):
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left
    unpadded_image = image
    if delta_h > 0:
        if delta_h == 1:
            unpadded_image = unpadded_image[top:, :, ...]
        else:
            unpadded_image = unpadded_image[top:-bottom, :, ...]
    if delta_w > 0:
        if delta_w == 1:
            unpadded_image = unpadded_image[:, left:, ...]
        else:
            unpadded_image = unpadded_image[:, left:-right, ...]

    return unpadded_image


def crop_image_with_mask(
        image: np.ndarray,
        mask: np.ndarray,
        buffer_pixels: int = 10,
        target_aspect_ratio: Optional[float] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop an image based on a mask with buffer and aspect ratio constraints.

    Args:
        image: Input image (H, W) or (H, W, C)
        mask: Binary mask (H, W) where non-zero values indicate active pixels
        buffer_pixels: Minimum pixel buffer to add around the mask bounding box
        target_aspect_ratio: Desired aspect ratio (width/height). If None, computed from target_width/target_height
        target_width: Target width for aspect ratio (alternative to target_aspect_ratio)
        target_height: Target height for aspect ratio (alternative to target_aspect_ratio)

    Returns:
        cropped_image: The cropped image
        crop_coords: Tuple of (x1, y1, x2, y2) defining the crop region
    """

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Determine target aspect ratio
    if target_aspect_ratio is None:
        if target_width is not None and target_height is not None:
            target_aspect_ratio = target_width / target_height
        else:
            target_aspect_ratio = None  # No aspect ratio constraint

    # Step 1: Find bounding box of active pixels in mask
    active_pixels = np.where(mask > 0)

    if len(active_pixels[0]) == 0:
        raise ValueError("No active pixels found in mask")

    y_min, y_max = active_pixels[0].min(), active_pixels[0].max()
    x_min, x_max = active_pixels[1].min(), active_pixels[1].max()

    # Step 2: Add buffer (constrained by image boundaries)
    x1 = max(0, x_min - buffer_pixels)
    y1 = max(0, y_min - buffer_pixels)
    x2 = min(img_width - 1, x_max + buffer_pixels)
    y2 = min(img_height - 1, y_max + buffer_pixels)

    # Current dimensions after buffer
    current_width = x2 - x1 + 1
    current_height = y2 - y1 + 1

    # Step 3: Adjust for aspect ratio if specified
    if target_aspect_ratio is not None:
        current_aspect = current_width / current_height

        if current_aspect < target_aspect_ratio:
            # Need to expand width
            target_width_new = int(current_height * target_aspect_ratio)
            width_diff = target_width_new - current_width

            # Try to expand symmetrically
            expand_left = width_diff // 2
            expand_right = width_diff - expand_left

            # Apply expansion with boundary constraints
            new_x1 = x1 - expand_left
            new_x2 = x2 + expand_right

            # Handle boundary violations
            if new_x1 < 0:
                # Shift the excess to the right
                excess = -new_x1
                new_x1 = 0
                new_x2 = min(img_width - 1, new_x2 + excess)

            if new_x2 >= img_width:
                # Shift the excess to the left
                excess = new_x2 - (img_width - 1)
                new_x2 = img_width - 1
                new_x1 = max(0, new_x1 - excess)

            x1, x2 = new_x1, new_x2

        elif current_aspect > target_aspect_ratio:
            # Need to expand height
            target_height_new = int(current_width / target_aspect_ratio)
            height_diff = target_height_new - current_height

            # Try to expand symmetrically
            expand_top = height_diff // 2
            expand_bottom = height_diff - expand_top

            # Apply expansion with boundary constraints
            new_y1 = y1 - expand_top
            new_y2 = y2 + expand_bottom

            # Handle boundary violations
            if new_y1 < 0:
                # Shift the excess to the bottom
                excess = -new_y1
                new_y1 = 0
                new_y2 = min(img_height - 1, new_y2 + excess)

            if new_y2 >= img_height:
                # Shift the excess to the top
                excess = new_y2 - (img_height - 1)
                new_y2 = img_height - 1
                new_y1 = max(0, new_y1 - excess)

            y1, y2 = new_y1, new_y2

    # Ensure coordinates are integers
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Crop the image
    cropped_image = image[y1:y2 + 1, x1:x2 + 1]

    return cropped_image, (x1, y1, x2, y2)



def draw_mask_boundary(image, mask, color=(0, 255, 0), thickness=2):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)


def get_mask_centroid(mask):
    M = cv2.moments(mask)

    # Calculate the centroid coordinates
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

    return cx, cy


def erode_mask(mask, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    image = cv2.erode(mask, kernel, cv2.BORDER_REFLECT, iterations=1)
    return image

def dilate_mask(mask, kernel_width=3, iterations=1):
    # Define a structuring element (e.g., a 3x3 rectangular kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_width))

    # Dilate the mask
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)

    return dilated_mask


def geometric_median_mask(mask_image, num_iterations=200, tolerance=1e-8):
    """
    Computes the geometric median of a binary mask.

    Args:
        mask_image (numpy.ndarray): A 2D NumPy array representing the binary mask.
                                    Non-zero values indicate mask pixels.
        num_iterations (int): Maximum number of iterations for Weiszfeld's algorithm.
        tolerance (float): Tolerance for convergence.

    Returns:
        numpy.ndarray: The (h, w) coordinates of the geometric median pixel.
    """

    # Get the coordinates of all pixels in the mask
    mask_pixels = np.transpose(np.nonzero(mask_image))

    if len(mask_pixels) == 0:
        return None  # No mask pixels found

    # Initial guess: centroid (mean) of the mask pixels
    geometric_median = np.mean(mask_pixels, axis=0).astype(float)

    for i in range(num_iterations):
        distances = np.linalg.norm(mask_pixels - geometric_median.reshape(1, -1), axis=-1)
        zero_idxs = np.isclose(distances, 0)
        nonzero_idxs = ~zero_idxs
        if np.all(zero_idxs):
            return np.round(geometric_median).astype(int)

        numerator = np.sum(mask_pixels[nonzero_idxs] / distances[nonzero_idxs].reshape(-1, 1), axis=0)
        denominator = np.sum(1.0 / distances[nonzero_idxs])
        new_geometric_median = numerator / denominator

        if np.linalg.norm(new_geometric_median - geometric_median) < tolerance:
            break  # Convergence

        geometric_median = new_geometric_median

    return np.round(geometric_median).astype(int)


def get_largest_island(mask):
    """
    Keeps only the largest connected component in a binary mask using OpenCV.

    Args:
        mask (numpy.ndarray): A binary mask (e.g., 0 for background, 255 for foreground).

    Returns:
        numpy.ndarray: A new mask containing only the largest connected component.
    """

    # Connected component labeling
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Find the largest component (excluding background)
    largest_label = 0
    largest_area = 0
    for i in range(1, num_labels):  # Iterate from 1 to exclude the background label
        area = stats[i, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = i

    # Create a new mask for the largest component
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    new_mask[labels == largest_label] = 255  # Set pixels of the largest component to 255

    return new_mask


def draw_bounding_box(image, x_range, y_range, color, thickness=2):
    # Bounding box parameters (assuming these are min_x, min_y, max_x, max_y)
    min_x, max_x = x_range
    min_y, max_y = y_range

    # Define the top-left and bottom-right corners
    pt1 = (min_x, min_y)
    pt2 = (max_x, max_y)

    # Draw the bounding box
    cv2.rectangle(image, pt1, pt2, color, thickness)


def draw_numbered_circle(image, center, radius, number, circle_color=(0, 255, 0), circle_thickness=2, text_color=(255, 255, 255), font_scale=0.7, font_thickness=2):
    """
    Draws a circle on an image and places a number inside it.

    Args:
        image (numpy.ndarray): The image to draw on.
        center (tuple): The (x, y) coordinates of the circle's center.
        radius (int): The radius of the circle.
        number (int or str): The number to display inside the circle.
        circle_color (tuple): BGR color of the circle.
        circle_thickness (int): Thickness of the circle's outline. Use -1 for a filled circle.
        text_color (tuple): BGR color of the text.
        font_scale (float): Font scale factor for the text.
        font_thickness (int): Thickness of the text characters.
    """
    # Draw the circle
    cv2.circle(image, center, radius, circle_color, circle_thickness)

    # Convert the number to a string
    text = str(number)

    # Get text size to center it
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

    # Calculate text position to center it within the circle
    text_x = center[0] - text_width // 2
    text_y = center[1] + text_height // 2

    # Put the text on the image
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)


def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output={}):
  """
  @disp: np array (H,W)
  @invalid_thres: > thres is invalid
  """
  disp = disp.copy()
  H,W = disp.shape[:2]
  invalid_mask = disp>=invalid_thres
  if (invalid_mask==0).sum()==0:
    other_output['min_val'] = None
    other_output['max_val'] = None
    return np.zeros((H,W,3))
  if min_val is None:
    min_val = disp[invalid_mask==0].min()
  if max_val is None:
    max_val = disp[invalid_mask==0].max()
  other_output['min_val'] = min_val
  other_output['max_val'] = max_val
  vis = ((disp-min_val)/(max_val-min_val)).clip(0,1) * 255
  if cmap is None:
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[...,::-1]
  else:
    vis = cmap(vis.astype(np.uint8))[...,:3]*255
  if invalid_mask.any():
    vis[invalid_mask] = 0
  return vis.astype(np.uint8)


def extract_numbers_from_str(text):
    # Extract all integers and floats (including negative)
    numbers = re.findall(r'-?\d+\.?\d*', text)
    # Convert to appropriate types (int or float)
    extracted_numbers = []
    for num_str in numbers:
        if '.' in num_str:
            extracted_numbers.append(float(num_str))
        else:
            extracted_numbers.append(int(num_str))
    return extracted_numbers


def create_polygon_from_vertices(vertices):
    """
    Create a valid Polygon from a list of vertices projected onto the x-y plane.

    Args:
        vertices (list of tuples): A list of tuples representing the projected points on the x-y plane.

    Returns:
        None or Polygon: A Shapely Polygon object created from the vertices, or None if a valid polygon cannot be formed.
    """
    # Reduce to unique points, as the projection may result in duplicates
    unique_points = np.unique(vertices, axis=0)

    # Handle the special case of 4 unique points separately
    if len(unique_points) == 4:
        # Sorting points for a rectangle or a simple quadrilateral
        center = np.mean(unique_points, axis=0)
        angles = np.arctan2(unique_points[:,1] - center[1], unique_points[:,0] - center[0])
        sorted_points = unique_points[np.argsort(angles)]
        return Polygon(sorted_points)

    # For more than 4 points, use the convex hull to ensure a valid polygon
    # This works well for convex shapes but might not be ideal for concave ones
    if len(unique_points) > 4:
        convex_hull = MultiPoint(unique_points).convex_hull
        if isinstance(convex_hull, Polygon):
            return convex_hull
        else:
            raise AssertionError("Convex Hull Creation Fail")  # Convex hull is not a polygon (could be a line or a point)

    raise AssertionError("Not enough points for a polygon")  # Not enough points for a polygon

def project_vertices_to_plane(vertices, plane_coeff):
    """
    Project 3D vertices onto a plane using matrix operations and return 2D points (ignoring the z-axis).

    Args:
        vertices (np.array): Nx3 array of 3D vertices.
        plane_coeff (np.array): Plane equation coefficients [a, b, c, d].

    Returns:
        np.array: Nx2 array of 2D points (projected vertices) on the plane.
    """
    # Extract plane coefficients
    a, b, c, d = plane_coeff
    normal_vector = np.array([a, b, c])
    
    # Normalize the normal vector of the plane
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Project the vertices onto the plane
    distances = (np.dot(vertices, normal_vector) + d).reshape(-1, 1)
    projected_vertices = vertices - distances * normal_vector
    
    # Find two orthogonal vectors (v1 and v2) that form the local 2D coordinate system on the plane
    if np.abs(normal_vector[2]) > np.abs(normal_vector[0]):
        v1 = np.array([1, 0, -a/c])  # Create a vector orthogonal to the normal
    else:
        v1 = np.array([0, 1, -b/c])
    
    # Normalize v1 and compute v2
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal_vector, v1)
    
    # Construct a matrix that transforms the projected 3D points into 2D plane coordinates
    transformation_matrix = np.vstack([v1, v2]).T
    
    # Compute 2D coordinates by multiplying the projected vertices with the transformation matrix
    projected_2d_points = np.dot(projected_vertices, transformation_matrix)
    
    return projected_2d_points

def get_possible_obj_on_wall(wall_objs_info, is_lateral_wall=False, verbose=False, visualize=False):
    """
    Get all objects possible to be mounted on the wall.

    Args:
        vertices (List[dict]): A list of object info w.r.t. a wall, where each dictionary contains:
            object name, object aabb, the minimum and maximum distance between the object and the wall, 
            and a polygon formed by 8 vertices of the bounding box of the object projected on the wall plane.
        is_lateral_wall (bool): Whether the wall is a lateral wall, i.e., whose normal vector forms a large 
            angle with the camera's forward direction in the floor plane. When calling this function in ACDC,
            wall angle > 45 degree is regarded as a lateral wall.
        verbose (bool): Whether to print out progress.
        visualize (bool): Whether to visualize projected bounding boxes on a wall plane.

    Returns:
        Set: A set of all objects possible to be mounted on the wall.
    """
    all_obj_names = {info["name"] for info in wall_objs_info}
    invalid_obj_names = set()
    wall_objs_info.sort(key=lambda x: x["wall_dist_min"])    # Sort objs in terms of their minimum distance to the wall
    for i in range(len(wall_objs_info)-1):
        obj1_info = wall_objs_info[i]
        for j in range(i+1, len(wall_objs_info)):
            obj2_info = wall_objs_info[j]
            if obj2_info["name"] in invalid_obj_names:
                continue
            
            if visualize:
                def vis():
                    fig, ax = plt.subplots()

                    # Get the coordinates of the first polygon
                    x1, y1 = obj1_info["polygon"].exterior.xy
                    ax.plot(x1, y1, color='blue', label=obj1_info['name'])

                    # Get the coordinates of the second polygon
                    x2, y2 = obj2_info["polygon"].exterior.xy
                    ax.plot(x2, y2, color='green', label=obj2_info['name'])

                    # Add legend, labels, and title
                    ax.legend()
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_title('Visualization of Projected Bounding Box')

                    # Display the plot in a pop-up window
                    plt.show()
                vis_process = multiprocessing.Process(target=vis)
                vis_process.start()
                vis_process.join()

            if obj1_info["polygon"].intersects(obj2_info["polygon"]):
                intersect_area = obj1_info["polygon"].intersection(obj2_info["polygon"]).area
                # If the intersection area exceeds 20% area of either polygon, 
                # the obj with larger min dist is not possible to be mounted on the wall
                if intersect_area / obj1_info["polygon"].area >= 0.33 or intersect_area / obj2_info["polygon"].area >= 0.33:
                    if verbose:
                        print(f"{obj1_info['name']} intersect with {obj2_info['name']}. Add {obj2_info['name']} as invalid")
                    invalid_obj_names.add(obj2_info["name"])
                    continue
            
            if is_lateral_wall: 
                # For lateral wall, if resizing an object causes shrinkage bounding box of another object with smaller distance to a wall,
                # or unreasonably large bounding box of itself, it is impossible for the object to make physical contact with the wall.
                obj1_bottom_height = min(obj1_info["vertices"][0][-1], obj1_info["vertices"][1][-1])
                obj2_bottom_height = min(obj2_info["vertices"][0][-1], obj2_info["vertices"][1][-1])
                obj1_center_height = (obj1_info["vertices"][0][-1] + obj1_info["vertices"][1][-1]) / 2
                obj2_center_height = (obj2_info["vertices"][0][-1] + obj2_info["vertices"][1][-1]) / 2
                obj1_top_height = max(obj1_info["vertices"][0][-1], obj1_info["vertices"][1][-1])
                obj2_top_height = max(obj2_info["vertices"][0][-1], obj2_info["vertices"][1][-1])
                if ((obj1_bottom_height <= obj2_center_height and obj1_top_height >= obj2_bottom_height) or (obj2_bottom_height <= obj1_center_height and obj2_top_height >= obj1_bottom_height)) \
                    and obj1_info["wall_dist_max"] < obj2_info["wall_dist_min"]:
                    if verbose:
                        print(f"Resizing {obj2_info['name']} can cause shrinkage of bounding box of {obj1_info['name']}, or unreasonably large bounding box of {obj2_info['name']}. Add {obj2_info['name']} as invalid")
                    invalid_obj_names.add(obj2_info["name"])

    return all_obj_names - invalid_obj_names

def rescale_image(img, in_limits, out_limits):
    """
    Rescales image from having values in range @in_limits to range @out_limits

    Args:
        img (np.ndarray): Image to be rescaled
        in_limits (2-tuple): (min, max) range of the input image
        out_limits (2-tuple): (min, max) range of the rescaled image

    Returns:
        np.ndarray: Rescaled image
    """
    # Out shape if specified should be (H, W) 2-tuple
    # Keep absolute range to be from 0 to 10 --> renormalize to 0 - 1
    in_min, in_max = in_limits
    out_min, out_max = out_limits
    scale_factor = abs(out_max - out_min) / abs(in_max - in_min)
    out_tf = (out_max + out_min) / 2.0
    in_tf = (in_max + in_min) / 2.0
    img = (img - in_tf) * scale_factor + out_tf
    return img



def process_depth_linear(depth, in_limits=(0, 10), out_limits=(0, 1), out_shape=None, use_16bit=True):
    """
    Discretizes a linear depth map from range @in_limits to range @out_limits, and optionally resizes it to @out_shape

    Args:
        depth (np.ndarray): Input depth map with metric values
        in_limits (2-tuple): (min, max) range of the input image
        out_limits (2-tuple): (min, max) range of the processed image
        out_shape (None or 2-tuple): If specified, the (H, W) @depth should be resized to
        use_16bit (bool): Whether to use 16-bit or 8-bit uint encoding when rescaling the image

    Returns:
        np.ndarray: Processed depth image.
    """
    # Out shape if specified should be (H, W) 2-tuple
    # Keep absolute range to be from 0 to 10 --> renormalize to 0 - 1
    in_min, in_max = in_limits
    foreground = np.where(depth <= in_max)
    background = np.where(depth > in_max)
    depth = rescale_image(img=depth, in_limits=in_limits, out_limits=out_limits)

    # Zero out background
    bit_size = 16 if use_16bit else 8
    dtype = np.uint16 if use_16bit else np.uint8
    depth[background] = 0.0
    # Multiply by 2 ** 16
    depth = depth * (2 ** bit_size)
    depth = depth.astype(dtype)
    if out_shape is not None:
        depth = cv2.resize(depth, (out_shape[1], out_shape[0]))
    return depth


def unprocess_depth_linear(depth, in_limits=(0.0, 1.0), out_limits=(0.0, 10.0)):
    """
    Unnormalizes a linear depth map from range @in_limits to range @out_limits. This process is the inverse
    of @procedss_depth_linear)

    Args:
        depth (np.ndarray): Input depth map with normalized values (the output of @process_depth_linear)
        in_limits (2-tuple): (min, max) range of the input image
        out_limits (2-tuple): (min, max) range of the unprocessed image
    
    Returns:
        np.ndarray: Unnormalized depth image.
    """
    # Map to float, divide by 2**16, then invert transform
    depth = depth.astype(float)
    depth = depth / (2 ** 16)

    # Keep unnormalize from 0 - 1 --> 0 - 10
    in_min, in_max = in_limits
    out_min, out_max = out_limits
    scale_factor = abs(out_max - out_min) / abs(in_max - in_min)
    out_tf = (out_max + out_min) / 2.0
    in_tf = (in_max + in_min) / 2.0
    depth = (depth - in_tf) * scale_factor + out_tf
    return depth

def compute_point_cloud_from_depth(depth, K, cam_to_img_tf=None, world_to_cam_tf=None, visualize_every=0,
                                   grid_limits=None):
    """
    Computes point cloud from depth image.

    Args:
        depth (np.ndarray): Input depth map with normalized values (the output of @process_depth_linear)
        K (np.ndarray): 3x3 cam intrinsics matrix
        cam_to_img_tf (np.ndarray): 4x4 Camera to image coordinate transformation matrix.
                    omni cam_to_img_tf is T.pose2mat(([0, 0, 0], T.euler2quat([np.pi, 0, 0])))
        world_to_cam_tf (np.ndarray): 4x4 World to camera coordinate transformation matrix.
        visualize_every (int): Step size when uniformly sampling points in the resulting point cloud to visualize.
        grid_limits (float): Visualization plot grid limits.
    
    Returns:
        np.ndarray: Resulting point cloud.
    """
    
    h, w = depth.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij", sparse=False)
    assert depth.min() >= 0
    u = x
    v = y
    uv = np.dstack((u, v, np.ones_like(u)))

    Kinv = np.linalg.inv(K)

    pc = depth.reshape(-1, 1) * (uv.reshape(-1, 3) @ Kinv.T)
    pc = pc.reshape(h, w, 3)

    # If no tfs, use identity matrix
    cam_to_img_tf = np.eye(4) if cam_to_img_tf is None else cam_to_img_tf
    world_to_cam_tf = np.eye(4) if world_to_cam_tf is None else world_to_cam_tf

    pc = np.concatenate([pc.reshape(-1, 3), np.ones((h * w, 1))], axis=-1)  # shape (H*W, 4)

    # Convert using camera transform
    # Create (H * W, 4) vector from pc
    pc = (pc @ cam_to_img_tf.T @ world_to_cam_tf.T)[:, :3].reshape(h, w, 3)

    if visualize_every > 0:
        def vis():
            pc_flat = np.array(pc.reshape(-1, 3))
            pc_flat = np.where(np.linalg.norm(pc_flat, axis=-1, keepdims=True) > 1e4, 0.0, pc_flat)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(pc_flat[::visualize_every, 0], pc_flat[::visualize_every, 1], pc_flat[::visualize_every, 2], s=1)
            if grid_limits is not None:
                ax.set_xbound(lower=-grid_limits, upper=grid_limits)
                ax.set_ybound(lower=-grid_limits, upper=grid_limits)
                ax.set_zbound(lower=-grid_limits, upper=grid_limits)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()
        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

    return pc


def transform_point_cloud(pc, tf):
    assert len(pc.shape) == 2
    N, C = pc.shape
    assert C == 3
    pc = np.concatenate([pc, np.ones((N, 1))], axis=-1)  # shape (N, 4)

    # Convert using camera transform
    # Create (N, 4) vector from pc
    return (pc @ tf.T)[:, :3]


def compute_bbox_from_mask(mask_fpath):
    mask = Image.open(mask_fpath)
    W, H = mask.size
    segmentation = np.where(np.array(mask))
    x_min = int(np.min(segmentation[1]))
    x_max = int(np.max(segmentation[1]))
    y_min = int(np.min(segmentation[0]))
    y_max = int(np.max(segmentation[0]))
    bbox = x_min, y_min, x_max, y_max
    bbox = np.array(bbox) / np.array([W, H, W, H])
    return bbox


def load_mask(mask_dir):
    """Load a mask image and convert it to a binary numpy array."""
    mask = Image.open(mask_dir).convert('1')  # Convert to binary image
    return np.array(mask, dtype=np.uint8)


def mask_area(mask):
    """Calculate the area of the mask."""
    return np.sum(mask)


def mask_intersection_area(mask1, mask2):
    """Calculate the intersection area between two masks."""
    intersection = np.logical_and(mask1, mask2)
    return np.sum(intersection)


def filter_large_masks(all_mask_dirs, prop_area_threshold=0.7):
    """Filter out large masks that contain smaller masks."""
    masks = [(load_mask(mask_dir), mask_dir) for mask_dir in all_mask_dirs]
    remaining_masks = []

    for i, (mask1, dir1) in enumerate(masks):
        is_larger_mask = False
        for j, (mask2, dir2) in enumerate(masks):
            if i != j:
                inter_area = mask_intersection_area(mask1, mask2)
                min_area = min(mask_area(mask1), mask_area(mask2))

                if inter_area > prop_area_threshold * min_area:
                    # If mask1 is the larger one, we mark it for removal
                    if mask_area(mask1) > mask_area(mask2):
                        is_larger_mask = True
                        break

        if not is_larger_mask:
            remaining_masks.append(dir1)

    return remaining_masks


def shrink_mask(mask, kernel_size=(20, 20), iterations=10):
    """
    Shrinks a binary mask by performing morphological erosion.

    Args:
        mask (numpy.ndarray): A 2D numpy array with boolean values (True/False), where True indicates floor.
        kernel_size (tuple of int): Kernel size used to perform binary erosion. Larger kernel size will lead to more aggresize erosion
        iterations (int): Number of times the erosion operation is applied.

    Returns:
        numpy.ndarray: A shrunk version of the input mask.
    """
    from scipy.ndimage import binary_erosion

    # Define the structure for erosion (use a 3x3 square by default)
    structure = np.ones(kernel_size, dtype=bool)

    # Perform erosion
    shrunk_mask = binary_erosion(mask, structure=structure, iterations=iterations)

    return shrunk_mask


def display_inlier_outlier(cloud, ind):
    """
    Visualize inliers (gray) and outliers (red) of a point cloud with different colors.

    Args:
        cloud (Open3d.geometry.PointCloud): Object point cloud.
        ind (List): Indices of inliers in cloud.
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    vis_cloud_list = []
    if len(inlier_cloud.points) == 0:
        print(f"No inliers")
    else:
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        vis_cloud_list.append(inlier_cloud)

    if len(outlier_cloud.points) == 0:
        print(f"No outliers")
    else:
        outlier_cloud.paint_uniform_color([1, 0, 0])
        vis_cloud_list.append(outlier_cloud)

    if vis_cloud_list:
        def vis():
            o3d.visualization.draw_geometries(vis_cloud_list)
        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

def denoise_obj_point_cloud(
        pcd_obj, 
        visualize_process=False,
        visualize_result=False,
        thred_percentage=0.05,
        absolute_threshold=200,
        large_pc_threshold=12000,
        noise_thred_percentage = 0.65,
        noise_thred_percentage_large_pc = 0.4,
    ):
    """
    Removes noise from object point by a set of point cloud denoising methods.
    Default values for arguments are tuned for resized images with longer side equals 1600 pixels.

    Args:
        pcd_obj (open3d.geometry.PointCloud): Object point cloud.
        visualize_process (bool): Whether to visualize the denoising process
        visualize_result (bool): Whether to visualize denoising result of the downsampled point cloud
        thred_percentage (float): Clusters with points more than total points number * thred_percentage
                            will not be classified as noise clusters.
        absolute_threshold (float): Clusters with points more than absolute_threshold will not be 
                            classified as noise clusters. This check will be skipped for large point clouds.
        large_pc_threshold (int): Object point clouds with more than @large_pc_threshold points will be 
                            denoised as large object point cloud.
        noise_thred_percentage (float): If the noise cluster (the cluster with label -1) has more than 
                            total points number * thred_percentage, the noise cluster will not be removed
        noise_thred_percentage_large_pc (float): For large point cloud, if the noise cluster (the cluster with label -1) 
                            has more than total points number * thred_percentage, the noise cluster will not be removed

    Returns:
        open3d.geometry.PointCloud: Denoised object point cloud.
        list: Corresponding point indices in pcd_obj.
    """

    if visualize_process:
        def vis():
            o3d.visualization.draw_geometries([pcd_obj])
        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

    # 1. Downsample the object point cloud
    # Downsample more aggresively for object point clouds with more points
    downsample_factor = min(4, 1 + len(pcd_obj.points) // 1000)                                         
    downsample_factor = 5 if len(pcd_obj.points) > large_pc_threshold else downsample_factor
    downsampled_indices = np.arange(len(pcd_obj.points))[::downsample_factor]
    pcd_obj_downsampled = pcd_obj.select_by_index(downsampled_indices)
    if visualize_process:
        def vis():
            o3d.visualization.draw_geometries([pcd_obj_downsampled])
        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

    # 2. Apply out-of-the-shelf statistical outlier removal
    pcd_obj_downsampled, ind = pcd_obj_downsampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Retain only the valid downsampled indices after noise removal
    valid_downsampled_indices = downsampled_indices[ind]

    # 3. Perform DBSCAN clustering
    min_points_cluster = min(max(8, len(pcd_obj_downsampled.points) // 1500 * 5), 25)
    eps_cluster = min(0.05, max(0.012, (len(pcd_obj.points) / 4000 + 2) * 0.005)) # Use smaller eps for smaller point cloud
    labels = np.array(pcd_obj_downsampled.cluster_dbscan(eps=eps_cluster, min_points=min_points_cluster, print_progress=False))
    max_label = labels.max()
    original_color = np.asarray(pcd_obj_downsampled.colors).copy()

    if visualize_process:
        def vis():
            cmap = plt.get_cmap("tab20")
            colors = cmap(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            pcd_obj_downsampled.colors = o3d.utility.Vector3dVector(colors[:, :3])
            o3d.visualization.draw_geometries([pcd_obj_downsampled])
        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()
    
    # Identify unique clusters and count the number of points in each cluster
    total_points = len(labels)
    unique_clusters, counts = np.unique(labels, return_counts=True)

    # List to store the indices of clusters considered as noise
    noise_clusters = []

    # Threshold for determining if a cluster is significant (not noise)
    significant_threshold = thred_percentage * total_points
    
    # 4. Remove non-significant clusters where most of points are outside the bbox fitted by other clusters
    clean = False
    while not clean:
        clean = True
        for cluster_id, count in zip(unique_clusters, counts):
            if cluster_id == -1 and cluster_id not in noise_clusters and \
                ((total_points < large_pc_threshold and count < noise_thred_percentage * total_points) or \
                (total_points >= large_pc_threshold and noise_thred_percentage_large_pc)):
                noise_clusters.append(cluster_id)
                clean = False
                significant_threshold = thred_percentage * (total_points - count)
                continue
            if cluster_id in noise_clusters or count > significant_threshold or (total_points < large_pc_threshold and count > absolute_threshold):
                continue
            
            # For noise candidates, determine if they should be removed
            cluster_indices = np.where(labels == cluster_id)[0]

            # Exclude points with cluster_id in noise_clusters or the current cluster_id
            exclude_clusters = np.append(noise_clusters, cluster_id)
            non_noise_indices = np.where(~np.isin(labels, exclude_clusters))[0]
            
            if len(non_noise_indices) == 0:
                continue  # No non-noise data to compare against
            
            # Compute bounding box of the non-noise points
            non_noise_pcd = pcd_obj_downsampled.select_by_index(non_noise_indices)
            non_noise_pcd.colors = o3d.utility.Vector3dVector(np.repeat([[1, 0, 0]], len(non_noise_pcd.points), axis=0).astype(np.float64))
            bbox = non_noise_pcd.get_oriented_bounding_box()
            
            # Check points of the current cluster against the bounding box
            cluster_points = pcd_obj_downsampled.select_by_index(cluster_indices)
            bbox.color = (0,1,0)

            # Get indices of points that are inside the oriented bounding box
            cluster_points_array = np.asarray(cluster_points.points)
            cluster_points_o3d = o3d.utility.Vector3dVector(cluster_points_array)
            inside_indices = bbox.get_point_indices_within_bounding_box(cluster_points_o3d)

            # Count points outside the oriented bounding box
            outside_count = len(cluster_points.points) - len(inside_indices)
            
            # If more than 70% of the points are outside the bounding box, remove this cluster
            if outside_count > 0.7 * count:
                noise_clusters.append(cluster_id)
                clean = False

    # Removing noise clusters
    valid_clusters = [cluster_id for cluster_id in unique_clusters if cluster_id not in noise_clusters]
    valid_indices_in_downsampled = [i for i in range(total_points) if labels[i] in valid_clusters]

    # Get the final valid indices in the original point cloud by mapping back the downsampled indices
    valid_indices_in_original = valid_downsampled_indices[valid_indices_in_downsampled]
    
    # Create a cleaned point cloud from valid points
    cleaned_pcd = pcd_obj_downsampled.select_by_index(valid_indices_in_downsampled)
    
    # Restore the original colors for the cleaned point cloud
    cleaned_pcd.colors = o3d.utility.Vector3dVector(original_color[valid_indices_in_downsampled])

    if visualize_result:
        display_inlier_outlier(pcd_obj_downsampled, valid_indices_in_downsampled)
        def vis():
            aligned_bbox = cleaned_pcd.get_axis_aligned_bounding_box()
            aligned_bbox.color = (1, 0, 0)
            oriented_bbox = cleaned_pcd.get_oriented_bounding_box()
            oriented_bbox.color = (0, 1, 0)
            o3d.visualization.draw_geometries([cleaned_pcd, aligned_bbox, oriented_bbox])
        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

    # Return the cleaned point cloud
    return cleaned_pcd, valid_indices_in_original


def denoise_obj_point_cloud_v2(
        pcd_obj,
        visualize_process=False,
        visualize_result=False,
        target_density_cm2=5.0,
        outlier_removal_n_neighbors=None,
        outlier_removal_std_ratio=1.0,
        dbscan_eps=None,
        dbscan_min_pts=None,
        cluster_proportion_threshold=0.05,
):
    """
    Removes noise from object point by a set of point cloud denoising methods.
    Default values for arguments are tuned for resized images with longer side equals 1600 pixels.

    Args:
        pcd_obj (open3d.geometry.PointCloud): Object point cloud.
        visualize_process (bool): Whether to visualize the denoising process
        visualize_result (bool): Whether to visualize denoising result of the downsampled point cloud
        target_density_cm2 (float): Target surface density of points, specified in pts / cm^2
        outlier_removal_n_neighbors (None or int): If specified, the number of neighbors to take into consideration
            when removing statistical outliers from the downsampled point cloud.
            If None, will default to 4 * target_density_cm2
        outlier_removal_std_ratio (float): The standard deviation threshold when removing statistical outliers from
            the downsampled point cloud.
        dbscan_eps (None or float): If specified, the distance to use when clustering points. None defaults to 0.01
            (i.e.: using the target density cm2)
        dbscan_min_pts (None or int): If specified, the minimum number of points to consider when clustering points.
            None defaults to int(np.round(target_density_cm2 * dbscan_eps / 0.01)) (i.e.: normalized to the expected
            number of points within the unit area cm^2)
        cluster_proportion_threshold (float): The proportion of points to consider whether a given cluster is noise
            or not

    Returns:
        open3d.geometry.PointCloud: Denoised object point cloud.
        list: Corresponding point indices in pcd_obj.
    """
    n_points = len(pcd_obj.points)
    if n_points == 0:
        return pcd_obj, np.array([], dtype=int)
    if n_points < 4:
        return pcd_obj, np.arange(n_points)

    if visualize_process:
        def vis():
            o3d.visualization.draw_geometries([pcd_obj])

        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

    # 1. Downsample the object point cloud
    # We attempt to normalize the target density (assuming equally spaced points) according to its convex hull surface area
    ch, _ = pcd_obj.compute_convex_hull()
    ch_surface_area_cm2 = ch.get_surface_area() * (100 ** 2)     # m^2 -> cm^2
    # obb_extent = pcd_obj.get_oriented_bounding_box().extent
    # pcd_vol = np.prod(obb_extent) * (100 ** 3)       # in m^3 -> cm^3
    n_target_points = int(target_density_cm2 * ch_surface_area_cm2)
    if n_target_points < len(pcd_obj.points):
        downsampled_indices = np.random.choice(len(pcd_obj.points), n_target_points, replace=False)
        pcd_obj_downsampled = pcd_obj.select_by_index(downsampled_indices)
    else:
        downsampled_indices = np.arange(len(pcd_obj.points))
        pcd_obj_downsampled = pcd_obj

    if visualize_process:
        def vis():
            o3d.visualization.draw_geometries([pcd_obj_downsampled])

        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

    # 2. Apply out-of-the-shelf statistical outlier removal
    outlier_removal_n_neighbors = int(np.round(4 * target_density_cm2)) if outlier_removal_n_neighbors is None else outlier_removal_n_neighbors
    pcd_obj_downsampled, ind = pcd_obj_downsampled.remove_statistical_outlier(nb_neighbors=outlier_removal_n_neighbors, std_ratio=outlier_removal_std_ratio)

    # Retain only the valid downsampled indices after noise removal
    valid_downsampled_indices = downsampled_indices[ind]
    if len(pcd_obj_downsampled.points) == 0:
        return pcd_obj_downsampled, np.array([], dtype=int)

    # 3. Perform DBSCAN clustering
    dbscan_eps = 0.01 if dbscan_eps is None else dbscan_eps
    dbscan_min_pts = int(np.round(target_density_cm2 * dbscan_eps / 0.01)) if dbscan_min_pts is None else dbscan_min_pts
    labels = np.array(pcd_obj_downsampled.cluster_dbscan(eps=dbscan_eps, min_points=dbscan_min_pts, print_progress=False))
    if len(labels) == 0:
        return pcd_obj_downsampled, np.array([], dtype=int)
    max_label = labels.max()

    if visualize_process:
        def vis():
            cmap = plt.get_cmap("tab20")
            colors = cmap(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            pcd_obj_downsampled.colors = o3d.utility.Vector3dVector(colors[:, :3])
            o3d.visualization.draw_geometries([pcd_obj_downsampled])

        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

    # Identify unique clusters and count the number of points in each cluster
    total_points = len(labels)
    unique_clusters, counts = np.unique(labels, return_counts=True)

    # Remove any cluster that is smaller than the given overall proportion of the point cloud
    inlier_cluster_idxs = []
    for cluster_id in unique_clusters:
        cluster_pt_idxs = np.where(labels == cluster_id)[0]
        n_cluster_pts = len(cluster_pt_idxs)
        if (n_cluster_pts / total_points) >= cluster_proportion_threshold:
            inlier_cluster_idxs.append(cluster_pt_idxs)
    if len(inlier_cluster_idxs) == 0:
        return pcd_obj_downsampled, np.array([], dtype=int)
    inlier_cluster_idxs = np.concatenate(inlier_cluster_idxs)

    cleaned_pcd = pcd_obj_downsampled.select_by_index(inlier_cluster_idxs)
    valid_indices_in_original = valid_downsampled_indices[inlier_cluster_idxs]

    if visualize_result:
        display_inlier_outlier(pcd_obj_downsampled, inlier_cluster_idxs)

        def vis():
            aligned_bbox = cleaned_pcd.get_axis_aligned_bounding_box()
            aligned_bbox.color = (1, 0, 0)
            oriented_bbox = cleaned_pcd.get_oriented_bounding_box()
            oriented_bbox.color = (0, 1, 0)
            o3d.visualization.draw_geometries([cleaned_pcd, aligned_bbox, oriented_bbox])

        vis_process = multiprocessing.Process(target=vis)
        vis_process.start()
        vis_process.join()

    # Return the cleaned point cloud
    return cleaned_pcd, valid_indices_in_original


def get_aabb_vertices(aabb):
    """
    Get coordinates of 8 vertices of 3D axis-aligned bounding box.

    Args:
        aabb (open3d.geometry.AxisAlignedBoundingBox): Object point cloud.

    Returns:
        list: Coordinates of 8 vertices.
    """
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound
    
    # Generate the 8 vertices of the AABB
    vertices = [[min_bound[0], min_bound[1], min_bound[2]],
                [min_bound[0], min_bound[1], max_bound[2]],
                [min_bound[0], max_bound[1], min_bound[2]],
                [min_bound[0], max_bound[1], max_bound[2]],
                [max_bound[0], min_bound[1], min_bound[2]],
                [max_bound[0], min_bound[1], max_bound[2]],
                [max_bound[0], max_bound[1], min_bound[2]],
                [max_bound[0], max_bound[1], max_bound[2]]]
    return vertices


def remove_component(vec, dir_to_remove, normalize=False):
    """
    Removes component from @vec, optionally normalizing if requested

    Args:
        vec (np.ndarray): (x,y,z) vector to project onto subspace
        dir_to_remove (np.ndarray): (x,y,z) vector whose direction should be removed from @vec
        normalize (bool): Whether to normalize the resulting vector or not
    """
    assert np.isclose(np.linalg.norm(dir_to_remove), 1.0)
    vec_dir_proj = np.dot(vec, dir_to_remove) * dir_to_remove
    vec_dir_perp_proj = vec - vec_dir_proj
    return vec_dir_perp_proj / np.linalg.norm(vec_dir_perp_proj) if normalize else vec_dir_perp_proj


def points_in_same_direction(vec_a, vec_b):
    """
    Determines whether @vec_a and @vec_b point in the same direction or not

    Args:
        vec_a (np.ndarray): (x,y,z) vector A
        vec_b (np.ndarray): (x,y,z) vector B

    Returns:
        bool: Whether both vectors point in same direction or not
    """
    return np.dot(vec_a / np.linalg.norm(vec_a), vec_b / np.linalg.norm(vec_b)) > 0


def get_reproject_offset(
    pc_obj,
    z_dir,
    xy_dist,    # taken wrt z_dir direction
    z_dist     # taken wrt plane perpendicular from z_dir direction
):
    """
    Reprojects an image from a given dataset into its equivalent pan and tilt angle offset with respect to the
    camera pose

    Args:
        pc_obj (np.ndarray): (N, 3) object point cloud, specified in camera A's frame
        z_dir (np.ndarray): (x,y,z) direction of the global z-axis, specified in camera A's frame
        xy_dist (float): Horizontal distance from the object bbox center to camera B's frame
        z_dist (float): Vertical distance from the object bbox center to camera B's frame

    Returns:
        2-tuple:
            - float: Pan angle offset
            - float: Tilt angle offset
    """

    # Get bbox center of obj
    obj_min, obj_max = pc_obj.min(axis=0), pc_obj.max(axis=0)
    obj_pos = (obj_min + obj_max) / 2.0

    # Get biggest dimension
    scale_factor = np.max(obj_max - obj_min)

    # In OG, our relative position wrt the object's bbox is [0, -2.30, 0.65] given the maximum width is 1. So, we
    # similar scale our distance proportional to the corresponding scale factor
    # So want our camera facing this object pose, with relative pose 0.65 in the computed z_direction,
    # and -2.30 in the direction of parallel to the ground plane and pointing in the direction of the camera

    # compute which way is "up" with respect to the image frame
    cam_to_obj = obj_pos
    cam_to_obj_z_perp_dir = remove_component(vec=cam_to_obj, dir_to_remove=z_dir, normalize=True)
    new_cam_pos = obj_pos + z_dist * scale_factor * z_dir - xy_dist * scale_factor * cam_to_obj_z_perp_dir
    tilt_dir = np.cross(z_dir, cam_to_obj_z_perp_dir)

    # The resulting orientation tf will be the tf to get from direction vector [0, 0, 1] to direction
    # obj_pos - new_cam_pos, with the assumption there is zero roll (i.e.: camera is not tilted wrt the ground)
    # This is a compounding of two angles:
    # (a) pan angle, which is derived from cosine between projection of current cam direction [0, 0, 1] and desired
    #     cam direction onto the plane defined by z_dir, and
    # (b) tilt angle, which is derived from the cosine between projection of current cam direction [0, 0, 1] and desired
    #     cam direction onto the plane defined by tilt_dir
    cur_cam_dir = np.array([0, 0, 1.0])
    new_cam_dir = obj_pos - new_cam_pos
    cur_cam_pan_proj = remove_component(vec=cur_cam_dir, dir_to_remove=z_dir, normalize=True)
    new_cam_pan_proj = remove_component(vec=new_cam_dir, dir_to_remove=z_dir, normalize=True)
    pan_angle_dir = 1 if points_in_same_direction(np.cross(cur_cam_pan_proj, new_cam_pan_proj), z_dir) else -1
    pan_angle = np.arccos(np.dot(cur_cam_pan_proj, new_cam_pan_proj)) * pan_angle_dir
    pan_tf = T.axisangle2mat(pan_angle * z_dir)
    cur_cam_tilt_proj = remove_component(vec=cur_cam_dir, dir_to_remove=tilt_dir, normalize=True)
    new_cam_tilt_proj = remove_component(vec=new_cam_dir, dir_to_remove=tilt_dir, normalize=True)
    tilt_angle_dir = 1 if points_in_same_direction(np.cross(cur_cam_tilt_proj, new_cam_tilt_proj), z_dir) else -1
    tilt_angle = np.arccos(np.dot(cur_cam_tilt_proj, new_cam_tilt_proj)) * tilt_angle_dir
    tilt_tf = T.axisangle2mat(tilt_angle * tilt_dir)

    # Compose final 4x4 tf
    cam_tf = np.eye(4)
    cam_tf[:3, 3] = new_cam_pos
    cam_tf[:3, :3] = tilt_tf @ pan_tf

    return pan_angle, tilt_angle


def distance_to_plane(point, plane_params, keep_sign=False):
    x1, y1, z1 = point
    a, b, c, d = plane_params
    if keep_sign:
        # Allow distance between points behind the plane and the plane to be negative
        numerator = a*x1 + b*y1 + c*z1 + d
    else:
        numerator = abs(a*x1 + b*y1 + c*z1 + d)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    distance = numerator / denominator
    return distance


def is_overlapped_box(box1_bottom_left, box1_upper_right, box2_bottom_left, box2_upper_right):
    return not (box1_upper_right[0] < box2_bottom_left[0]
                or box1_bottom_left[0] > box2_upper_right[0]
                or box1_upper_right[1] < box2_bottom_left[1]
                or box1_bottom_left[1] > box2_upper_right[1])


def resize_image(img, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Rescales image @img, maintaining its aspect ratio.
    Source taken from: https://stackoverflow.com/a/44659589

    Args:
        img (np.ndarray): (H,W,3) Image to resize
        width (None or int): If specified, new width (in pixels) to resize image to. Only one of @width, @height should
            be specified
        height (None or int): If specified, new height (in pixels) to resize image to. Only one of @width, @height
            should be specified
        inter (int): Type of interpolation to use
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = img.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return img

    # check to see if the width is None
    assert width is None or height is None, "Only one of @width and @height should be specified!"

    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(img, dim, interpolation=inter)

    # return the resized image
    return resized

def resize_with_pad(image: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    """
    Resize image to target dimensions while preserving aspect ratio by padding with black.
    Replicates OpenPI's resize_with_pad behavior.
    
    Args:
        image: Input image as numpy array (H, W, C), uint8 or float32
        height: Target height (default 224)
        width: Target width (default 224)
        
    Returns:
        Resized and padded image as uint8 numpy array (height, width, C)
    """
    # Accept torch tensors and convert to numpy
    if isinstance(image, th.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = image.permute(1, 2, 0)
        image = image.numpy()
    elif not isinstance(image, np.ndarray):
        image = np.array(image)

    # Convert to uint8 if needed
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * np.clip(image, 0, 1)).astype(np.uint8)
    
    # Get current dimensions
    cur_height, cur_width = image.shape[:2]
    
    # Calculate scale to fit within target dimensions
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    
    # Resize using PIL for quality
    pil_image = Image.fromarray(image)
    pil_resized = pil_image.resize((resized_width, resized_height), Image.BILINEAR)
    resized = np.array(pil_resized)
    
    # Pad to target dimensions (centered)
    pad_top = (height - resized_height) // 2
    pad_bottom = height - resized_height - pad_top
    pad_left = (width - resized_width) // 2
    pad_right = width - resized_width - pad_left
    
    padded = np.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )
    
    return padded.astype(np.uint8)


class NumpyTorchEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, th.Tensor):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def make_json_serializable(obj):
    """
    Recursively convert tensors and numpy arrays to lists for JSON serialization.

    Args:
        obj: Object to convert (dict, list, tensor, array, or primitive)

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, th.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def _convert_omegaconf(obj):
    """Recursively convert OmegaConf objects to native Python types."""
    from omegaconf import OmegaConf, ListConfig, DictConfig
    
    if isinstance(obj, DictConfig):
        return {k: _convert_omegaconf(v) for k, v in obj.items()}
    elif isinstance(obj, ListConfig):
        return [_convert_omegaconf(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _convert_omegaconf(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_omegaconf(v) for v in obj]
    else:
        return obj


def dump_json(data, fpath):
    """
    Dump data to a JSON file.
    
    Handles OmegaConf ListConfig/DictConfig objects by converting them to
    native Python types before serialization.
    """
    # Recursively convert any OmegaConf objects to native Python types
    data = _convert_omegaconf(data)
    
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4)


def remove_background(
    input_path: Path | str,
    output_path: Path | str,
    session: Any = None,
    max_side: int | None = None,
) -> None:
    if isinstance(input_path, str):
        input_path = Path(input_path)
    if isinstance(output_path, str):
        output_path = Path(output_path)

    from rembg import remove, new_session
    if session is None:
        session = new_session("bria-rmbg", providers=REMBG_CPU_PROVIDERS)
    """Remove background from an image and save to output path."""
    print(f"Processing: {input_path}")

    input_image = Image.open(input_path).convert("RGBA")
    original_size = input_image.size

    if max_side is not None and max_side > 0:
        longest_side = max(original_size)
        if longest_side > max_side:
            scale = max_side / float(longest_side)
            resized_size = (
                max(1, int(round(original_size[0] * scale))),
                max(1, int(round(original_size[1] * scale))),
            )
            # Run rembg on a smaller image to cap ONNX memory use.
            inference_image = input_image.resize(resized_size, Image.Resampling.LANCZOS)
        else:
            inference_image = input_image
    else:
        inference_image = input_image

    output_image = remove(inference_image, session=session).convert("RGBA")

    if output_image.size != original_size:
        restored = input_image.copy()
        alpha = output_image.getchannel("A").resize(original_size, Image.Resampling.LANCZOS)
        restored.putalpha(alpha)
        output_image = restored

    output_path = output_path.with_suffix('.png')
    # Published atomically: stage 6 writes the file the stage-7 listener watches for.
    with atomic_output_path(output_path) as tmp_output_path:
        output_image.save(tmp_output_path)

    print(f"  -> Saved: {output_path}")
