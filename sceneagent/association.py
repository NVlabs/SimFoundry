"""Gaussian-to-object association constants from SceneAgent section 3.3.

Masks from different views form one object when their Gaussian centroids lie
within min(0.10, 0.6 e) reconstruction units and the Gaussian sets have
containment of at least 0.18 or Jaccard index of at least 0.08.

A Gaussian belongs to a mask when its center projects inside the mask and its
camera-frame depth is within the front-surface band

    z <= z_hat + max(eps_a, eps_r * z_hat)

with eps_a = 0.08 reconstruction units and eps_r = 0.03. The coarse image cell
used to compute z_hat is a square of round(s / 192) pixels, at most 16, where
s is the shorter image side.
"""

CENTROID_ABS = 0.10
CENTROID_EXTENT_SCALE = 0.6
CONTAINMENT_MIN = 0.18
JACCARD_MIN = 0.08
EPS_A = 0.08
EPS_R = 0.03
CELL_DIVISOR = 192
CELL_MAX = 16


def centroid_threshold(extent):
    return min(CENTROID_ABS, CENTROID_EXTENT_SCALE * float(extent))


def masks_same_object(centroid_distance, extent, containment, jaccard):
    close = float(centroid_distance) <= centroid_threshold(extent)
    overlap = float(containment) >= CONTAINMENT_MIN or float(jaccard) >= JACCARD_MIN
    return bool(close and overlap)


def front_surface_limit(cell_min_depth, eps_a=EPS_A, eps_r=EPS_R):
    cell_min_depth = float(cell_min_depth)
    return cell_min_depth + max(eps_a, eps_r * cell_min_depth)


def gaussian_belongs(projects_inside_mask, depth, cell_min_depth):
    if not projects_inside_mask:
        return False
    return float(depth) <= front_surface_limit(cell_min_depth)


def mask_cell_side_px(short_side_px):
    side = int(round(float(short_side_px) / CELL_DIVISOR))
    return min(CELL_MAX, side)
