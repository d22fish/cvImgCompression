# %%
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import sys
from scipy.ndimage import label, uniform_filter, binary_dilation, convolve, distance_transform_edt
from PIL import Image
np.set_printoptions(threshold=sys.maxsize)
from scipy.interpolate import splprep, splev
from io import StringIO
import struct
import heapq
from collections import Counter
import itertools
import math
import io
import random
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import warnings
from concurrent.futures import ProcessPoolExecutor
import time

# %%
CAST_LOG = {}
def load_random_n_images(folder, n=20, seed=42):
    random.seed(seed)
    filenames = sorted(os.listdir(folder))
    sample_size = min(n, len(filenames))
    random_filenames = random.sample(filenames, sample_size)
    return [os.path.join(folder, f) for f in random_filenames]

# %%
#Image-level complexity from local LAB standard deviation.
def local_lab_complexity_score(img_bgr, window_size=7, statistic="median"):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    # OpenCV LAB - standard-like LAB
    L = lab[:, :, 0] * (100.0 / 255.0)
    A = lab[:, :, 1] - 128.0
    B = lab[:, :, 2] - 128.0

    std_maps = []
    for ch in (L, A, B):
        mu = uniform_filter(ch, size=window_size, mode="reflect")
        mu2 = uniform_filter(ch * ch, size=window_size, mode="reflect")
        var = np.maximum(mu2 - mu * mu, 0.0)
        std_maps.append(np.sqrt(var))

    local_std = np.sqrt(std_maps[0]**2 + std_maps[1]**2 + std_maps[2]**2)

    if statistic == "mean":
        score = float(np.mean(local_std))
    else:
        score = float(np.median(local_std))

    return score, local_std

def adaptive_deltaE_threshold(complexity_score, low_complexity=2.0, high_complexity=12.0, high_thresh=10.0, low_thresh=4.0):
    # Low complexity - higher threshold
    # High complexity - lower threshold
    
    t = (complexity_score - low_complexity) / (high_complexity - low_complexity)
    t = np.clip(t, 0.0, 1.0)
    thresh = high_thresh * (1.0 - t) + low_thresh * t
    return float(thresh)

# %%
# Trace the outer edge of a cluster of pixels.
# Returns ordered X and Y coordinates along the contour.
def traceClusterEdges(cluster):
    if len(cluster) == 0:
        return [], []

    # Convert cluster to set for O(1) membership check
    pixels = set(cluster)

    # clockwise starting from north
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]

    # Find starting pixel: top-left-most
    start = min(pixels, key=lambda c: (c[1], c[0]))
    x, y = start
    contour = [(x, y)]
    current = start
    prev_dir = 6  # start looking from left neighbor (south-west)

    while True:
        found_next = False
        # Check 8 neighbors clockwise starting from (prev_dir + 1)
        for i in range(8):
            idx = (prev_dir + 1 + i) % 8
            dx, dy = directions[idx]
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in pixels:
                contour.append(neighbor)
                current = neighbor
                prev_dir = (idx + 4) % 8  # set new prev_dir opposite direction
                found_next = True
                break
        if not found_next or current == start:
            break

    # Separate X and Y
    X, Y = zip(*contour)
    return list(X), list(Y)

# %%
# Apply bilateral filter
def run_segmentation(filtered, thresh): 
    # Convert input image to CIELAB (OpenCV 8-bit encoding)
    img = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)

    # imgClusters[i,j] = cluster number pixel belongs to
    imgClusters = np.ones((img.shape[0], img.shape[1]), dtype=int)


    # List of perimeter pixels for each cluster
    clusterEdges = []

    # visited[i,j] = 1 - pixel not yet assigned to any cluster
    # visited[i,j] = 0 - pixel already assigned
    visited = np.ones((img.shape[0], img.shape[1]), dtype=int)

    clusterNum = 1              # current cluster ID

    # 8-connected neighborhood definition
    structure = np.ones((3, 3), dtype=np.uint8)

    # OpenCV LAB (8-bit) - approximate standard LAB:
    # L*: [0,100], a*: [-128,127], b*: [-128,127]
    img_f = img.astype(np.float64)
    L_std = img_f[:, :, 0] * (100.0 / 255.0)
    A_std = img_f[:, :, 1] - 128.0
    B_std = img_f[:, :, 2] - 128.0

    remaining = int(np.count_nonzero(visited))

    # Continue until all pixels have been assigned
    while remaining > 0:

        # Greedy assumption: all unvisited pixels could be in this cluster
        imgClusters[visited != 0] = clusterNum

        # ---- STEP 1: find the seed pixel and initialize cluster mean ----
        seed_positions = np.argwhere(visited == 1)
        if seed_positions.shape[0] == 0:
            break

        # ---- STEP 2: build the similarity mask (approx delta-E) ----
        startX, startY = seed_positions[0]
        seed_L = L_std[startX, startY]
        seed_A = A_std[startX, startY]
        seed_B = B_std[startX, startY]

        dL = L_std - seed_L
        dA = A_std - seed_A
        dB = B_std - seed_B

        # Euclidean distance in standard-like LAB coordinates
        dist = np.sqrt(dL**2 + dA**2 + dB**2)
        fitMask = (visited == 1) & (dist < thresh)

        # ---- STEP 3: connected component labeling (8-connected) ----
        labels, _ = label(fitMask, structure)

        # Label of the connected component containing the seed pixel
        seedLabel = labels[startX, startY]

        # region[i,j] = True ONLY for pixels reachable from the seed
        region = (labels == seedLabel)

        # ---- STEP 4: accept pixels in this connected component ----
        region_count = int(np.count_nonzero(region))
        if region_count == 0:
            # fallback so loop cannot stall forever
            visited[startX, startY] = 0
            imgClusters[startX, startY] = clusterNum
            region = np.zeros_like(visited, dtype=bool)
            region[startX, startY] = True
            region_count = 1

        visited[region] = 0
        imgClusters[region] = clusterNum
        remaining -= region_count

        # ---- STEP 5: border detection (vectorized 8-neighborhood) ----
        # Interior pixel has all 8-neighbors in region
        padded = np.pad(region, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        up        = padded[:-2, 1:-1]
        down      = padded[2:, 1:-1]
        left      = padded[1:-1, :-2]
        right     = padded[1:-1, 2:]
        up_left   = padded[:-2, :-2]
        up_right  = padded[:-2, 2:]
        down_left = padded[2:, :-2]
        down_right= padded[2:, 2:]

        interior8 = up & down & left & right & up_left & up_right & down_left & down_right
        border_mask = region & (~interior8)

        border_idx = np.argwhere(border_mask)
        clust = [tuple(rc) for rc in border_idx]

        # ---- STEP 6: finalize this cluster ----
        clusterEdges.append(clust)

        clusterNum += 1
    edgesSorted = []
    for cluster in clusterEdges:
        xList, yList = traceClusterEdges(cluster)
        edgesSorted.append([xList, yList])
    return imgClusters, edgesSorted

# %%
# Closed-contour B-spline helpers
def downsample_contour(contour_pts, stride=4):
    if len(contour_pts) <= 2:
        return contour_pts

    base = contour_pts[:-1] if contour_pts[0] == contour_pts[-1] else contour_pts
    stride = max(1, int(stride))
    sampled = [base[i] for i in range(0, len(base), stride)]

    if len(sampled) < 4:
        sampled = base

    if sampled and sampled[0] != sampled[-1]:
        sampled.append(sampled[0])
    return sampled

def fit_closed_bspline(contour_pts, smooth=0.0, degree=3):
    base = contour_pts[:-1]  # splprep(per=True) handles closure internally
    rows = np.array([p[0] for p in base], dtype=float)
    cols = np.array([p[1] for p in base], dtype=float)

    k = min(int(degree), len(base) - 1)
    if k < 1:
        return None

    try:
        tck, _ = splprep([rows, cols], s=float(smooth), per=True, k=k)
    except Exception:
        return None
    return tck

# Strict writer: write C only if geometry exists; Spline first, L fallback
def contour_area_rc(pts):
    # pts as [(row,col), ...], closed or open
    if len(pts) < 3:
        return 0.0
    base = pts[:-1] if pts[0] == pts[-1] else pts
    if len(base) < 3:
        return 0.0
    x = np.array([p[1] for p in base], dtype=float)
    y = np.array([p[0] for p in base], dtype=float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def enforce_closed_contour_rc(x, y):
    pts = [(int(r), int(c)) for r, c in zip(x, y)]
    if not pts:
        return []
    cleaned = [pts[0]]
    for p in pts[1:]:
        if p != cleaned[-1]:
            cleaned.append(p)
    if len(cleaned) < 3:
        return []
    if cleaned[0] != cleaned[-1]:
        cleaned.append(cleaned[0])
    return cleaned

def write_loop_L(f, contour_pts, max_pts=120):
    # L;r,c;r,c;...;
    base = contour_pts[:-1] if contour_pts[0] == contour_pts[-1] else contour_pts
    if len(base) < 3:
        return False, [], []
    step = max(1, math.ceil(len(base) / max_pts)) 
    sampled = base[::step]
    if len(sampled) < 3:
        return False, [], []
    if sampled[0] != sampled[-1]:
        sampled.append(sampled[0])
    rows_l = [p[0] for p in sampled]
    cols_l = [p[1] for p in sampled]
    dr = [rows_l[0]] + [rows_l[i] - rows_l[i-1] for i in range(1, len(rows_l))]
    dc = [cols_l[0]] + [cols_l[i] - cols_l[i-1] for i in range(1, len(cols_l))]
    payload = ';'.join(f"{dr[i]},{dc[i]}" for i in range(len(dr)))
    f.write("L;" + payload + ";\n")
    return True, dr, dc

# %%
# contour_rc: list/array of points (closed or open), returns geometric complexity metrics for adaptive spline smoothing.
def contour_complexity_metrics(contour_rc, complexity_weights=(0.6, 0.4)):
    pts = np.asarray(contour_rc, dtype=np.float64)

    if len(pts) < 5:
        return {
            "perimeter": 0.0,
            "mean_turn": 0.0,
            "max_turn": 0.0,
            "corner_density": 1.0,
            "complexity": 1.0,
        }

    # Ensure closed contour
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    diffs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    perimeter = float(np.sum(seg_lens))

    # Remove zero-length segments
    valid = seg_lens > 1e-8
    diffs = diffs[valid]
    seg_lens = seg_lens[valid]

    if len(diffs) < 3:
        return {
            "perimeter": perimeter,
            "mean_turn": 0.0,
            "max_turn": 0.0,
            "corner_density": 0.0,
            "complexity": 0.0,
        }

    unit = diffs / seg_lens[:, None]

    dots = np.sum(unit[:-1] * unit[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    turns = np.arccos(dots)  # radians

    mean_turn = float(np.mean(turns))
    max_turn = float(np.max(turns))

    corner_threshold = np.deg2rad(35.0)
    corner_density = float(np.mean(turns > corner_threshold))

    # Combined complexity score ~[0,1]
    complexity = complexity_weights[0] * min(mean_turn / np.pi, 1.0) + complexity_weights[1] * corner_density

    return {
        "perimeter": perimeter,
        "mean_turn": mean_turn,
        "max_turn": max_turn,
        "corner_density": corner_density,
        "complexity": complexity,
    }

# Returns contour-adaptive splprep smoothing parameter s, Higher s: more smoothing/fewer control points, Lower s: tighter fit.
def adaptive_spline_smooth(contour_rc, min_smooth=0.0, max_smooth=20.0, base_perimeter=200.0, complexity_weights=(0.6, 0.4)):
    m = contour_complexity_metrics(contour_rc, complexity_weights)

    perimeter = m["perimeter"]
    complexity = m["complexity"]

    # Larger contours can tolerate more total smoothing
    size_factor = max(perimeter / float(base_perimeter), 0.25)

    # More geometric complexity - less smoothing
    complexity_penalty = 1.0 - np.clip(complexity, 0.0, 1.0)

    s = max_smooth * size_factor * complexity_penalty
    s = float(np.clip(s, min_smooth, max_smooth))

    return s, m

# %%
# ---------- DCT helpers ----------
def jpeg_qtable_luma():
    return np.array([
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ], dtype=np.float32)

def quality_to_qtable(quality):
    q50 = jpeg_qtable_luma()
    q = int(np.clip(quality, 1, 100))
    scale = 5000 / q if q < 50 else 200 - 2*q
    qt = np.floor((q50 * scale + 50) / 100)
    qt = np.clip(qt, 1, 255).astype(np.float32)
    return qt

def pad_to_block(img, block=8):
    h, w, c = img.shape
    ph = (block - h % block) % block
    pw = (block - w % block) % block
    if ph == 0 and pw == 0:
        return img, (h, w)
    out = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="edge")
    return out, (h, w)

def psnr(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 99.0
    return 10*np.log10((255.0**2)/mse)

# ---------- Encode residual with DCT ----------
def encode_residual_dct(original_bgr, base_lab, quality=35, block=8):
    orig_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
    base_rgb = cv2.cvtColor(base_lab, cv2.COLOR_LAB2RGB).astype(np.int16)

    residual = (orig_rgb - base_rgb).astype(np.float32)
    residual_pad, (h0, w0) = pad_to_block(residual, block=block)
    hp, wp, _ = residual_pad.shape

    qt = quality_to_qtable(quality)
    qcoeff = np.zeros((hp, wp, 3), dtype=np.int16)

    for ch in range(3):
        for y in range(0, hp, block):
            for x in range(0, wp, block):
                blk = residual_pad[y:y+block, x:x+block, ch]
                d = cv2.dct(blk)
                qblk = np.round(d / qt).astype(np.int16)
                qcoeff[y:y+block, x:x+block, ch] = qblk

    return {
        "qcoeff": qcoeff,
        "h0": h0,
        "w0": w0,
        "quality": int(quality),
        "block": int(block)
    }

# ---------- Decode residual + add to base ----------
def decode_residual_dct(base_lab, payload):
    qcoeff = payload["qcoeff"].astype(np.float32)
    h0, w0 = payload["h0"], payload["w0"]
    quality, block = payload["quality"], payload["block"]

    base_rgb = cv2.cvtColor(base_lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    base_pad, _ = pad_to_block(base_rgb, block=block)

    hp, wp, _ = qcoeff.shape
    qt = quality_to_qtable(quality)
    residual_hat = np.zeros((hp, wp, 3), dtype=np.float32)

    for ch in range(3):
        for y in range(0, hp, block):
            for x in range(0, wp, block):
                qblk = qcoeff[y:y+block, x:x+block, ch]
                d_hat = qblk * qt
                blk_hat = cv2.idct(d_hat)
                residual_hat[y:y+block, x:x+block, ch] = blk_hat

    recon_pad = np.clip(np.round(base_pad + residual_hat), 0, 255).astype(np.uint8)
    return recon_pad[:h0, :w0, :]

# %%
def fit_region_planar_model(img_color, region_mask):
    ys, xs = np.where(region_mask)
    if len(xs) == 0:
        return np.zeros((3, 3), dtype=float), 0.0, 0.0
    
    xc, yc = float(xs.mean()), float(ys.mean())
    A = None
    if len(xs) >= 3:
        A = np.column_stack([np.ones_like(xs), xs - xc, ys - yc]).astype(np.float64)
    if A is None or np.linalg.matrix_rank(A) < 3:
        # fallback: constant model as degenerate plane
        mean_col = img_color[ys, xs].mean(axis=0) if len(xs) > 0 else np.array([0, 0, 0], dtype=float)
        coef = np.zeros((3, 3), dtype=float)
        coef[:, 0] = mean_col
        xc, yc = float(xs.mean()) if len(xs) > 0 else 0.0, float(ys.mean()) if len(ys) > 0 else 0.0
        return coef, xc, yc

    vals = img_color[ys, xs].astype(np.float64)  # Nx3

    coef = np.zeros((3, 3), dtype=np.float64)
    for ch in range(3):
        coef[ch], *_ = np.linalg.lstsq(A, vals[:, ch], rcond=None)
    return coef, xc, yc

def fit_region_quadratic_model(img_color, region_mask):
    ys, xs = np.where(region_mask)
    if len(xs) < 6:
        mean_col = img_color[ys, xs].mean(axis=0) if len(xs) > 0 else np.zeros(3)
        coef = np.zeros((3, 6), dtype=float)
        coef[:, 0] = mean_col
        xc = float(xs.mean()) if len(xs) > 0 else 0.0
        yc = float(ys.mean()) if len(ys) > 0 else 0.0
        return coef, xc, yc

    xc, yc = float(xs.mean()), float(ys.mean())
    dx = xs - xc
    dy = ys - yc
    A = np.column_stack([np.ones_like(xs), dx, dy, dx**2, dy**2, dx*dy]).astype(np.float64)
    vals = img_color[ys, xs].astype(np.float64)

    coef = np.zeros((3, 6), dtype=np.float64)
    for ch in range(3):
        coef[ch], *_ = np.linalg.lstsq(A, vals[:, ch], rcond=None)
    return coef, xc, yc

# %%
def check_int_cast(arr, scale, dtype, label, log=CAST_LOG):
    info = np.iinfo(dtype)
    scaled = np.round(np.asarray(arr, dtype=np.float64) * scale)

    max_abs = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    over = (scaled < info.min) | (scaled > info.max)
    n_over = int(np.count_nonzero(over))

    e = log.setdefault(label, {"max_abs": 0.0, "n_clipped": 0, "n_total": 0, "limit": int(info.max)})
    e["max_abs"] = max(e["max_abs"], max_abs)
    e["n_clipped"] += n_over
    e["n_total"] += int(scaled.size)

    scaled = np.clip(scaled, info.min, info.max)
    return scaled.astype(dtype)

# %%
# Evaluates linear (coef 3x3) or quadratic (coef 3x6) model at region pixels.
def reconstruct_region(coef, region_mask, xc, yc):
    H, W = region_mask.shape
    ys, xs = np.where(region_mask)
    recon = np.zeros((H, W, 3), dtype=np.float32)
    dx = xs - xc
    dy = ys - yc

    if coef.shape[1] == 3:
        for ch in range(3):
            recon[ys, xs, ch] = coef[ch,0] + coef[ch,1]*dx + coef[ch,2]*dy
    else:
        for ch in range(3):
            recon[ys, xs, ch] = (coef[ch,0] + coef[ch,1]*dx + coef[ch,2]*dy +
                                 coef[ch,3]*dx**2 + coef[ch,4]*dy**2 + coef[ch,5]*dx*dy)

    recon[ys, xs] = np.clip(recon[ys, xs], 0, 255)
    return recon

# %%
# Quantizes DCT residual between original and planar reconstruction for one region.
def encode_region_dct_residual(img_lab, planar_recon, region_mask, quality=35, block=8):
    ys, xs = np.where(region_mask)
    if len(xs) == 0:
        return None

    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1

    if (r1 - r0) < block or (c1 - c0) < block:
        return None

    H, W = r1 - r0, c1 - c0
    orig_crop = img_lab[r0:r1, c0:c1].astype(np.float32)
    plan_crop = planar_recon[r0:r1, c0:c1].astype(np.float32)
    mask_crop = region_mask[r0:r1, c0:c1]

    residual = orig_crop - plan_crop
    residual[~mask_crop] = 0.0

    ph = H + (block - H % block) % block
    pw = W + (block - W % block) % block
    res_pad = np.zeros((ph, pw, 3), dtype=np.float32)
    res_pad[:H, :W] = residual

    qt = quality_to_qtable(quality)
    qcoeff = np.zeros((ph, pw, 3), dtype=np.int16)

    for ch in range(3):
        for y in range(0, ph, block):
            for x in range(0, pw, block):
                blk = res_pad[y:y+block, x:x+block, ch]
                d = cv2.dct(blk)
                qcoeff[y:y+block, x:x+block, ch] = np.round(d / qt).astype(np.int16)

    # texture score: fraction of DCT energy in AC coefficients
    dc_energy, total_energy = 0.0, 0.0
    for ch in range(3):
        for y in range(0, ph, block):
            for x in range(0, pw, block):
                blk = qcoeff[y:y+block, x:x+block, ch].astype(float)
                total_energy += float(np.sum(blk**2))
                dc_energy += float(blk[0, 0]**2)
    ac_energy = total_energy - dc_energy
    texture_score = ac_energy / total_energy if total_energy > 0 else 0.0

    nz = int(np.count_nonzero(qcoeff))
    byte_cost = 10 + nz * 6

    return {'qcoeff': qcoeff, 'bbox': (r0, c0, H, W), 'byte_cost': byte_cost, 'texture_score': texture_score}

# %%
# Lagrangian R-D decision (linear, quadratic, linear+DCT)
# J = D + lambda * R, choose tier with minimum J
def rd_three_tier(img_lab, region_mask, coef_lin, xc, yc, coef_quad, dct_payload, R_geom, lam, quality=35):
    ys, xs = np.where(region_mask)
    orig = img_lab[ys, xs].astype(np.float64)

    # tier 0: linear planar only
    recon_lin = reconstruct_region(coef_lin, region_mask, xc, yc)
    D0 = float(np.mean((orig - recon_lin[ys, xs]) ** 2))
    R0 = R_geom + 1 + 18
    J0 = D0 + lam * R0

    # tier 1: quadratic only
    recon_quad = reconstruct_region(coef_quad, region_mask, xc, yc)
    D1 = float(np.mean((orig - recon_quad[ys, xs]) ** 2))
    R1 = R_geom + 1 + 54
    J1 = D1 + lam * R1

    # tier 2: linear + DCT residual
    if dct_payload is None:
        best = min([(J0, 0), (J1, 1)], key=lambda x: x[0])
        return best[1], best[0]

    r0, c0, H, W = dct_payload['bbox']
    qcoeff = dct_payload['qcoeff'].astype(np.float32)
    qt = quality_to_qtable(quality)
    block = 8
    ph, pw = qcoeff.shape[:2]

    residual_hat = np.zeros((ph, pw, 3), dtype=np.float32)
    for ch in range(3):
        for y in range(0, ph, block):
            for x in range(0, pw, block):
                residual_hat[y:y+block, x:x+block, ch] = cv2.idct(qcoeff[y:y+block, x:x+block, ch] * qt)

    recon_dct = recon_lin.copy()
    recon_dct[r0:r0+H, c0:c0+W] += residual_hat[:H, :W]
    recon_dct = np.clip(recon_dct, 0, 255)

    D2 = float(np.mean((orig - recon_dct[ys, xs]) ** 2))
    R2 = R_geom + 1 + 18 + dct_payload['byte_cost']
    J2 = D2 + lam * R2

    best = min([(J0, 0), (J1, 1), (J2, 2)], key=lambda x: x[0])
    return best[1], best[0]

# %%
# Extract boundary band from labeled image, returns boolean masks for band and interior
def boundary_band_from_labels(labels, radius=2):
    H, W = labels.shape
    b = np.zeros((H, W), dtype=np.uint8)
    b[:, :-1] |= (labels[:, :-1] != labels[:, 1:]).astype(np.uint8)
    b[:-1, :] |= (labels[:-1, :] != labels[1:, :]).astype(np.uint8)
    if radius > 0:
        k = np.ones((2*radius+1, 2*radius+1), dtype=np.uint8)
        b = cv2.dilate(b, k, iterations=1)
    band = (b == 1)
    interior = ~band
    return band, interior

# Metrics (MAE, PSNR, SSIM) computed only over masked region
def masked_mae(orig_rgb, rec_rgb, mask2d):
    # d[mask2d] - shape (N,3), mean over all selected channel values
    d = np.abs(orig_rgb.astype(np.float32) - rec_rgb.astype(np.float32))
    if mask2d.sum() == 0:
        return np.nan
    return float(d[mask2d].mean())

def masked_psnr(orig_rgb, rec_rgb, mask2d):
    e2 = (orig_rgb.astype(np.float32) - rec_rgb.astype(np.float32)) ** 2
    if mask2d.sum() == 0:
        return np.nan
    mse = float(e2[mask2d].mean())
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0**2) / mse))

# compute per-channel SSIM maps, then average map over mask
def masked_ssim(orig_rgb, rec_rgb, mask2d):
    vals = []
    for ch in range(3):
        _, s_map = ssim(
            orig_rgb[:, :, ch], rec_rgb[:, :, ch],
            data_range=255, full=True
        )
        vals.append(s_map)
    s_map_rgb = np.mean(np.stack(vals, axis=2), axis=2)
    if mask2d.sum() == 0:
        return np.nan
    return float(s_map_rgb[mask2d].mean())

# %%
# Vectorized repair for unpainted pixels after reconstruction.
# Fills unpainted pixels that touch already-painted pixels by averaging their painted neighbors
# Repeats outward for up to max_iters.
def repair_coverage(imRecover, painted, max_iters=5, min_coverage=0.98):
    imRecover = imRecover.copy()
    painted = painted.astype(bool).copy()

    # 8-neighbor kernel, excluding center pixel
    kernel = np.ones((3, 3), dtype=np.float64)
    kernel[1, 1] = 0.0

    coverage = float(np.mean(painted))
    if coverage >= min_coverage:
        return imRecover, painted.astype(np.uint8), coverage

    for _ in range(max_iters):
        painted_float = painted.astype(np.float64)

        # Find unpainted pixels adjacent to painted pixels
        adjacent_to_painted = (binary_dilation(painted, structure=np.ones((3, 3), dtype=bool)) & (~painted))

        if not np.any(adjacent_to_painted):
            break

        # Count painted neighbors around every pixel
        neighbor_count = convolve(painted_float, kernel, mode="constant", cval=0.0)
        fillable = adjacent_to_painted & (neighbor_count > 0)

        if not np.any(fillable):
            break

        repaired = imRecover.astype(np.float64)

        # Average painted-neighbor colors for each channel
        for ch in range(imRecover.shape[2]):
            channel = imRecover[:, :, ch].astype(np.float64)
            neighbor_sum = convolve(channel * painted_float, kernel, mode="constant", cval=0.0)
            repaired[:, :, ch][fillable] = (neighbor_sum[fillable] / neighbor_count[fillable])

        imRecover = np.clip(repaired, 0, 255).astype(np.uint8)
        painted[fillable] = True

        coverage = float(np.mean(painted))
        if coverage >= min_coverage:
            break

    return imRecover, painted.astype(np.uint8), coverage

# Fallback: every unpainted pixel gets the color of the nearest painted pixel to remove all remaining unfilled spots
def repair_coverage_nearest(imRecover, painted):
    imRecover = imRecover.copy()
    painted_bool = painted.astype(bool)

    if np.all(painted_bool):
        return imRecover, painted_bool.astype(np.uint8), 1.0

    # distance_transform_edt returns indices of nearest zero pixel.
    nearest_indices = distance_transform_edt(~painted_bool, return_distances=False, return_indices=True)

    nearest_rows = nearest_indices[0]
    nearest_cols = nearest_indices[1]

    unpainted = ~painted_bool
    imRecover[unpainted] = imRecover[nearest_rows[unpainted], nearest_cols[unpainted]]
    painted_bool[unpainted] = True

    coverage = float(np.mean(painted_bool))
    return imRecover, painted_bool.astype(np.uint8), coverage

# %%
def run_encode_decode(
        img, imgClusters, edgesSorted, chosenImage,
        spline_min_smooth=0.0,
        spline_max_smooth=20.0,
        spline_base_perim=200.0,
        complexity_weights=(0.6, 0.4),
        corner_thresh_deg=35.0,
        rd_lambda=.001,
        rd_dct_quality=35
    ):
    min_points = 6
    min_area = 4.0
    encoded_ok = 0
    fallback_ok = 0
    dropped = 0
    tier_counts = [0, 0, 0]
    # overflow logging
    CAST_LOG = {}
    CAST_LOG.clear()
    pid = os.getpid()
    txt_path = f"/tmp/imComp_{pid}.txt"
    bin_path = f"/tmp/imFullComp_{pid}.bin"
    huff_path = f"/tmp/imFullComp_{pid}.huff"
    with open(txt_path, "w") as f, open(bin_path, "wb") as fb:
        header_pos = fb.tell()
        fb.write(struct.pack('<HHI', img.shape[0], img.shape[1], 0))
        for i in range(len(edgesSorted)):
            x = edgesSorted[i][0]
            y = edgesSorted[i][1]

            contour = enforce_closed_contour_rc(x, y)
            if len(contour) < min_points or contour_area_rc(contour) < min_area:
                dropped += 1
                continue

            # Try spline first
            #tck = fit_closed_bspline(contour, smooth=SPLINE_SMOOTH, degree=3)
            s_val, contour_metrics = adaptive_spline_smooth(contour, min_smooth=spline_min_smooth, max_smooth=spline_max_smooth, base_perimeter=spline_base_perim, complexity_weights=complexity_weights)
            tck = fit_closed_bspline(contour, smooth=s_val, degree=3)
            #print(f"s={s_val:.2f}, perim={contour_metrics['perimeter']:.1f}, complexity={contour_metrics['complexity']:.3f}")

            wrote_geom = False
            geom_lines = []
            geom_bin = None

            if tck is not None:
                t, c, k = tck
                cx, cy = c[0], c[1]
                if len(t) > 0 and len(cx) > 0 and len(cy) > 0:
                    # Delta coding
                    tQ = np.round(t * 10000).astype(np.int32)
                    tD = np.r_[tQ[0], np.diff(tQ)].astype(np.int16)
                    cxQ = np.round(cx * 10000).astype(np.int32)
                    cxD = np.r_[cxQ[0], np.diff(cxQ)]
                    cyQ = np.round(cy * 10000).astype(np.int32)
                    cyD = np.r_[cyQ[0], np.diff(cyQ)]
                    t_str = ",".join(str(v) for v in tD)
                    cx_str = ",".join(str(v) for v in cxD)
                    cy_str = ",".join(str(v) for v in cyD)
                    geom_lines.append("S;" + str(int(k)) + ";" + t_str + ";" + cx_str + ";" + cy_str + ";")
                    geom_bin = (
                        struct.pack('<BBH', 0, int(k), len(tD)) +
                        tD.astype('<i2').tobytes() +
                        struct.pack('<H', len(cxD)) +
                        cxD.astype('<i4').tobytes() +
                        cyD.astype('<i4').tobytes()
                    )
                    wrote_geom = True

            # Fallback if spline failed
            if not wrote_geom:
                temp = StringIO()
                ok, dr, dc = write_loop_L(temp, contour, max_pts=120)
                if ok:
                    geom_lines.append(temp.getvalue().strip())
                    geom_bin = (
                        struct.pack('<BH', 1, len(dr)) +
                        np.array(dr, dtype=np.int8).tobytes() +
                        np.array(dc, dtype=np.int8).tobytes()
                    )
                    wrote_geom = True
                    fallback_ok += 1

            if wrote_geom:
                # --- planar model fit ---
                cluster_id = i + 1
                region_mask = (imgClusters == cluster_id)
                coef, xc, yc = fit_region_planar_model(img, region_mask)
                coef_quad, _, _ = fit_region_quadratic_model(img, region_mask)
                recon_lin = reconstruct_region(coef, region_mask, xc, yc)
                dct_payload = encode_region_dct_residual(img, recon_lin, region_mask, quality=rd_dct_quality)
                tier, _ = rd_three_tier(img, region_mask, coef, xc, yc, coef_quad, dct_payload, len(geom_bin), rd_lambda, quality=rd_dct_quality)
                tier_counts[tier] += 1
                xcQ, ycQ = int(round(xc)), int(round(yc))

                if tier == 0:
                    coefQ = check_int_cast(coef, 100, np.int16, "M.linear.tier0")
                    m0 = ",".join(str(v) for v in coefQ[0])
                    m1 = ",".join(str(v) for v in coefQ[1])
                    m2 = ",".join(str(v) for v in coefQ[2])
                    f.write(f"M;0;{xcQ},{ycQ};{m0};{m1};{m2};\n")
                    fb.write(struct.pack('<hh', xcQ, ycQ))
                    fb.write(struct.pack('<B', 0))
                    fb.write(coefQ.astype('<i2').tobytes())

                elif tier == 1:
                    coefQ_lin = check_int_cast(coef_quad[:, :3], 100,   np.int16, "M.quad.linear")
                    coefQ_sec = check_int_cast(coef_quad[:, 3:], 10000, np.int32, "M.quad.second")
                    m0 = ",".join(str(v) for v in np.concatenate([coefQ_lin[0], coefQ_sec[0]]))
                    m1 = ",".join(str(v) for v in np.concatenate([coefQ_lin[1], coefQ_sec[1]]))
                    m2 = ",".join(str(v) for v in np.concatenate([coefQ_lin[2], coefQ_sec[2]]))
                    f.write(f"M;1;{xcQ},{ycQ};{m0};{m1};{m2};\n")
                    fb.write(struct.pack('<hh', xcQ, ycQ))
                    fb.write(struct.pack('<B', 1))
                    fb.write(coefQ_lin.astype('<i2').tobytes())   # 3x3 int16 = 18 bytes
                    fb.write(coefQ_sec.astype('<i4').tobytes())   # 3x3 int32 = 12 bytes

                else:
                    coefQ = check_int_cast(coef, 100, np.int16, "M.linear.tier2")
                    m0 = ",".join(str(v) for v in coefQ[0])
                    m1 = ",".join(str(v) for v in coefQ[1])
                    m2 = ",".join(str(v) for v in coefQ[2])
                    f.write(f"M;2;{xcQ},{ycQ};{m0};{m1};{m2};\n")
                    fb.write(struct.pack('<hh', xcQ, ycQ))
                    fb.write(struct.pack('<B', 2))
                    fb.write(coefQ.astype('<i2').tobytes())

                for line in geom_lines:
                    f.write(line + "\n")
                f.write("\n")
                fb.write(geom_bin)

                if tier == 2:
                    r0b, c0b, H_b, W_b = dct_payload['bbox']
                    qcoeff = dct_payload['qcoeff']
                    ph, pw = qcoeff.shape[:2]
                    nz_idx = np.argwhere(qcoeff != 0)
                    fb.write(struct.pack('<HHHH', r0b, c0b, H_b, W_b))
                    fb.write(struct.pack('<H', len(nz_idx)))
                    for idx in nz_idx:
                        y_i, x_i, ch_i = idx
                        pos = int(ch_i) * ph * pw + int(y_i) * pw + int(x_i)
                        fb.write(struct.pack('<I', pos))
                        fb.write(struct.pack('<h', int(qcoeff[y_i, x_i, ch_i])))

                encoded_ok += 1
            else:
                dropped += 1
    with open(bin_path, "r+b") as fb:
        fb.seek(header_pos + 4)
        fb.write(struct.pack('<I', encoded_ok))
    spline_count = encoded_ok - fallback_ok
    total = encoded_ok
    print(f"S (spline):   {spline_count}/{total} = {spline_count/total*100:.1f}%")
    print(f"L (fallback): {fallback_ok}/{total} = {fallback_ok/total*100:.1f}%")
    print(f"Dropped:      {dropped}")

    print("\nFixed-width cast report:")
    for label, e in sorted(CAST_LOG.items()):
        pct = 100 * e["n_clipped"] / max(e["n_total"], 1)
        print(f"  {label:16s} max_abs    ={e['max_abs']:10.1f} / limit {e['limit']:6d}"
            f"   clipped {e['n_clipped']}/{e['n_total']} ({pct:.3f}%)")
        
    # 1. Count byte frequencies across the full binary file
    data = open(bin_path, "rb").read()
    freq = Counter(data)

    # 2. Build Huffman tree with tie-breaking counter to avoid heapq comparison errors
    counter = itertools.count()
    heap = [[f, next(counter), s, None, None] for s, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        heapq.heappush(heap, [lo[0] + hi[0], next(counter), None, lo, hi])

    root = heap[0]

    # 3. Generate code table by walking the tree
    codes = {}

    def build_codes(node, prefix=""):
        if node[2] is not None:
            codes[node[2]] = prefix or "0"
        else:
            build_codes(node[3], prefix + "0")
            build_codes(node[4], prefix + "1")

    build_codes(root)

    # 4. Encode byte stream and pad to byte boundary
    bitstring = "".join(codes[b] for b in data)
    padding = (8 - len(bitstring) % 8) % 8
    bitstring += "0" * padding

    encoded = bytearray()
    for i in range(0, len(bitstring), 8):
        encoded.append(int(bitstring[i:i+8], 2))

    # 5. Write header (frequency table + padding) then encoded data
    with open(huff_path, "wb") as fh:
        fh.write(struct.pack('<H', len(freq)))
        for sym, cnt in freq.items():
            fh.write(struct.pack('<BI', sym, cnt))
        fh.write(struct.pack('<B', padding))
        fh.write(bytes(encoded))

    print(f"[PID {pid}] Huffman: {os.path.getsize(huff_path)} Binary: {os.path.getsize(bin_path)} Text: {os.path.getsize(txt_path)}")

    print(f"Tier 0 (linear):    {tier_counts[0]}/{encoded_ok} = {tier_counts[0]/encoded_ok*100:.1f}%")
    print(f"Tier 1 (quadratic): {tier_counts[1]}/{encoded_ok} = {tier_counts[1]/encoded_ok*100:.1f}%")
    print(f"Tier 2 (DCT):       {tier_counts[2]}/{encoded_ok} = {tier_counts[2]/encoded_ok*100:.1f}%")

    # Conversion of huffman to binary stream

    # 1. Read header to recover frequency table
    with open(huff_path, "rb") as fh:
        n_syms = struct.unpack('<H', fh.read(2))[0]
        freq_dec = {}
        for _ in range(n_syms):
            sym, cnt = struct.unpack('<BI', fh.read(5))
            freq_dec[sym] = cnt
        padding = struct.unpack('<B', fh.read(1))[0]
        encoded = fh.read()

    # 2. Rebuild tree identically from stored frequencies
    counter = itertools.count()
    heap = [[f, next(counter), s, None, None] for s, f in freq_dec.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        heapq.heappush(heap, [lo[0] + hi[0], next(counter), None, lo, hi])

    root = heap[0]

    # 3. Decode bitstream back to original bytes
    bitstring = "".join(f"{b:08b}" for b in encoded)
    if padding > 0:
        bitstring = bitstring[:-padding]

    # Converted huffman to binary stream for round-trip verification, saved in decoded
    decoded = bytearray()
    node = root
    for bit in bitstring:
        node = node[3] if bit == "0" else node[4]
        if node[2] is not None:
            decoded.append(node[2])
            node = root

    # Check that the decoded bytes match the original data
    assert bytes(decoded) == data, "Huffman round-trip mismatch"

    # 1. Initialize recovery image and list to track cluster sizes for sorting
    imRecoverBin = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    imRecoverBin[:] = (255, 128, 128)
    clusters_to_draw = []

    # 2. Read binary file once to collect all shapes and their data
    with io.BytesIO(bytes(decoded)) as fb:
        H, W, n_regions = struct.unpack('<HHI', fb.read(8))

        for _ in range(n_regions):
            xcQ, ycQ = struct.unpack('<hh', fb.read(4))
            model_type = struct.unpack('<B', fb.read(1))[0]

            if model_type == 1:
                #raw = np.frombuffer(fb.read(36), dtype='<i2').reshape(3, 6).astype(float)
                lin = np.frombuffer(fb.read(18), dtype='<i2').reshape(3, 3).astype(float)
                sec = np.frombuffer(fb.read(36), dtype='<i4').reshape(3, 3).astype(float)
                coef = np.zeros((3, 6), dtype=float)
                coef[:, :3] = lin / 100.0
                coef[:, 3:] = sec / 10000.0
                #coef = np.zeros((3, 6), dtype=float)
                #coef[:, :3] = raw[:, :3] / 100.0
                #coef[:, 3:] = raw[:, 3:] / 10000.0
            else:
                coef = np.frombuffer(fb.read(18), dtype='<i2').reshape(3, 3).astype(float) / 100.0        

            geom_type = struct.unpack('<B', fb.read(1))[0]

            pts = []
            if geom_type == 0:
                k = struct.unpack('<B', fb.read(1))[0]
                n_knots = struct.unpack('<H', fb.read(2))[0]
                tD = np.frombuffer(fb.read(n_knots * 2), dtype='<i2')
                t = np.cumsum(tD).astype(float) / 10000.0
                n_ctrl = struct.unpack('<H', fb.read(2))[0]
                cxD = np.frombuffer(fb.read(n_ctrl * 4), dtype='<i4')
                cyD = np.frombuffer(fb.read(n_ctrl * 4), dtype='<i4')
                cx = np.cumsum(cxD).astype(float) / 10000.0
                cy = np.cumsum(cyD).astype(float) / 10000.0

                tck = (t, [cx, cy], k)
                coarse_n = max(128, int(4 * n_ctrl))
                coarse_rows, coarse_cols = splev(np.linspace(0.0, 1.0, coarse_n, endpoint=False), tck)
                drows = np.diff(np.r_[coarse_rows, coarse_rows[0]])
                dcols = np.diff(np.r_[coarse_cols, coarse_cols[0]])
                perimeter_estimate = np.sum(np.sqrt(drows**2 + dcols**2))
                n_samples = max(int(np.ceil(perimeter_estimate)), int(4 * n_ctrl), 64)
                rows, cols = splev(np.linspace(0.0, 1.0, n_samples, endpoint=False), tck)

                for r, c in zip(rows, cols):
                    pts.append([int(round(c)), int(round(r))])

            else:
                n_pts = struct.unpack('<H', fb.read(2))[0]
                dr = np.frombuffer(fb.read(n_pts), dtype=np.int8)
                dc = np.frombuffer(fb.read(n_pts), dtype=np.int8)
                rs = np.cumsum(dr)
                cs = np.cumsum(dc)
                for i in range(n_pts):
                    pts.append([int(np.clip(cs[i], 0, W-1)), int(np.clip(rs[i], 0, H-1))])

            dct_info = None
            if model_type == 2:
                r0d, c0d, H_d, W_d = struct.unpack('<HHHH', fb.read(8))
                block = 8
                ph = H_d + (block - H_d % block) % block
                pw = W_d + (block - W_d % block) % block
                n_nz = struct.unpack('<H', fb.read(2))[0]
                qcoeff = np.zeros((ph, pw, 3), dtype=np.int16)
                for _ in range(n_nz):
                    pos = struct.unpack('<I', fb.read(4))[0]
                    val = struct.unpack('<h', fb.read(2))[0]
                    ch_i = pos // (ph * pw)
                    rem = pos % (ph * pw)
                    y_i = rem // pw
                    x_i = rem % pw
                    qcoeff[y_i, x_i, ch_i] = val

                qt = quality_to_qtable(rd_dct_quality)
                residual_hat = np.zeros((ph, pw, 3), dtype=np.float32)
                for ch in range(3):
                    for y in range(0, ph, block):
                        for x in range(0, pw, block):
                            residual_hat[y:y+block, x:x+block, ch] = cv2.idct(qcoeff[y:y+block, x:x+block, ch].astype(np.float32) * qt)

                dct_info = {'bbox': (r0d, c0d, H_d, W_d), 'residual': residual_hat}

            clusters_to_draw.append({'pts': pts, 'model': coef, 'area': 0.0, 'xc': float(xcQ), 'yc': float(ycQ), 'dct': dct_info})

    # 3. Painter's algorithm sorted by polygon area
    for cluster in clusters_to_draw:
        if len(cluster['pts']) > 2:
            polygon_points = np.array(cluster['pts'], dtype=np.int32).reshape((-1, 1, 2))
            cluster['area'] = float(abs(cv2.contourArea(polygon_points)))
        else:
            cluster['area'] = 0.0

    clusters_to_draw.sort(key=lambda x: x['area'], reverse=True)

    # 4. Draw + track coverage (single pass)
    painted = np.zeros((H, W), dtype=np.uint8)

    drawn_clusters = 0
    for cluster in clusters_to_draw:
        if len(cluster['pts']) > 2:
            polygon_points = np.array(cluster['pts'], dtype=np.int32).reshape((-1, 1, 2))

            region = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(region, [polygon_points], color=1)
            ys, xs = np.where(region == 1)

            if len(xs) == 0:
                continue

            coef = cluster['model']
            xc, yc = cluster['xc'], cluster['yc']
            recon = reconstruct_region(coef, region.astype(bool), xc, yc)
            imRecoverBin[ys, xs] = recon[ys, xs].astype(np.uint8)

            if cluster['dct'] is not None:
                r0d, c0d, H_d, W_d = cluster['dct']['bbox']
                res = cluster['dct']['residual']
                ry = np.clip(ys - r0d, 0, res.shape[0] - 1)
                rx = np.clip(xs - c0d, 0, res.shape[1] - 1)
                corrected = recon[ys, xs].astype(np.float32) + res[ry, rx]
                imRecoverBin[ys, xs] = np.clip(corrected, 0, 255).astype(np.uint8)

            painted[ys, xs] = 1
            drawn_clusters += 1

    # 5. Coverage + repair
    imRecoverBin, painted, coverage = repair_coverage(imRecoverBin, painted, max_iters=5, min_coverage=0.98)
    if coverage < 1.0:
        imRecoverBin, painted, coverage = repair_coverage_nearest(imRecoverBin, painted)
    imRecoverBin = cv2.cvtColor(imRecoverBin, cv2.COLOR_LAB2RGB)
    return imRecoverBin, os.path.getsize(huff_path)

# %%
# Hyperparameter tuning on validation set
def run_pipeline(image_path, sigmaColor=35, sigmaSpace=None,
                 high_thresh=8.0, low_thresh=4.0,
                 low_complexity=2.0, high_complexity=12.0, window_size=7,
                 spline_min_smooth=0.0, spline_max_smooth=20.0, spline_base_perim=200.0,
                 complexity_weights=(0.6, 0.4), corner_thresh_deg=35.0,
                 rd_lambda=0.001, rd_dct_quality=35):
    
    # Run full encode-decode pipeline on one image, return metrics dict.
    if sigmaSpace is None:
        sigmaSpace = sigmaColor

    chosenImage = cv2.imread(image_path)
    img = cv2.cvtColor(chosenImage, cv2.COLOR_BGR2LAB).astype(float)
    orig_rgb = cv2.cvtColor(chosenImage, cv2.COLOR_BGR2RGB)

    # Stage 1: Segmentation
    complexity_score, _ = local_lab_complexity_score(chosenImage, window_size=window_size)
    thresh = adaptive_deltaE_threshold(complexity_score,
                                        low_complexity=low_complexity,
                                        high_complexity=high_complexity,
                                        high_thresh=high_thresh,
                                        low_thresh=low_thresh)
    filtered = cv2.bilateralFilter(chosenImage, d=7, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

    imgClusters, edgesSorted = run_segmentation(filtered, thresh) 

    # Stage 2 & 3: Encode with given parameters
    rec_rgb, file_size = run_encode_decode(
        img, imgClusters, edgesSorted, chosenImage,
        spline_min_smooth=spline_min_smooth,
        spline_max_smooth=spline_max_smooth,
        spline_base_perim=spline_base_perim,
        complexity_weights=complexity_weights,
        corner_thresh_deg=corner_thresh_deg,
        rd_lambda=rd_lambda,
        rd_dct_quality=rd_dct_quality
    )

    # Metrics
    band_mask, interior_mask = boundary_band_from_labels(imgClusters, radius=2)
    return {
        'psnr': masked_psnr(orig_rgb, rec_rgb, np.ones(orig_rgb.shape[:2], dtype=bool)),
        'ssim': masked_ssim(orig_rgb, rec_rgb, np.ones(orig_rgb.shape[:2], dtype=bool)),
        'bnd_psnr': masked_psnr(orig_rgb, rec_rgb, band_mask),
        'bnd_ssim': masked_ssim(orig_rgb, rec_rgb, band_mask),
        'int_psnr': masked_psnr(orig_rgb, rec_rgb, interior_mask),
        'int_ssim': masked_ssim(orig_rgb, rec_rgb, interior_mask),
        'file_size': file_size,
        'bpp': file_size * 8 / (orig_rgb.shape[0] * orig_rgb.shape[1]),
        'thresh': thresh,
        'n_regions': len(edgesSorted),
    }

def _run_one(args):
    path, kwargs = args
    return run_pipeline(path, **kwargs)

# %% [markdown]
# Hyperparameter Tuning

# %%
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    val_images = load_random_n_images('BSDS500/val', n=20, seed=42)
    results = []
    for sc in [190]:
        for ss in [145]:
            for ht in [28.0]:
                for lt in [10.5]:
                    with ProcessPoolExecutor(max_workers=10) as pool:
                        scores = list(pool.map(_run_one, [(p, {'sigmaColor': sc, 'sigmaSpace': ss, 'high_thresh': ht, 'low_thresh': lt}) for p in val_images]))
                    avg_psnr = np.mean([s['psnr'] for s in scores])
                    avg_ssim = np.mean([s['ssim'] for s in scores])
                    avg_bpp = np.mean([s['bpp'] for s in scores])
                    results.append({'sigmaColor': sc, 'sigmaSpace': ss, 'high_thresh': ht, 'low_thresh': lt,
                                   'psnr': avg_psnr, 'ssim': avg_ssim, 'bpp': avg_bpp})
                    print(f"[{time.strftime('%H:%M:%S')}] sc={sc} ss={ss} ht={ht} lt={lt}: PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} bpp={avg_bpp:.3f}")

    best = max(results, key=lambda r: r['psnr'])
    print(f"\nBest S1: sigmaColor={best['sigmaColor']}, sigmaSpace={best['sigmaSpace']}, high_thresh={best['high_thresh']}, low_thresh={best['low_thresh']}")

    results_s2 = []
    for smin in [1.0]:
        for smax in [4.5]:
            for bp in [300.0]:
                for cw in [(0.40, 0.60)]:
                    with ProcessPoolExecutor(max_workers=10) as pool:
                        scores = list(pool.map(_run_one, [(p, {'sigmaColor': best['sigmaColor'],
                            'sigmaSpace': best['sigmaSpace'],
                            'high_thresh': best['high_thresh'], 'low_thresh': best['low_thresh'],
                            'spline_min_smooth': smin, 'spline_max_smooth': smax,
                            'spline_base_perim': bp, 'complexity_weights': cw}) for p in val_images]))
                    avg_psnr = np.mean([s['psnr'] for s in scores])
                    avg_ssim = np.mean([s['ssim'] for s in scores])
                    avg_bpp = np.mean([s['bpp'] for s in scores])
                    results_s2.append({'smin': smin, 'smax': smax, 'bp': bp, 'cw': cw,
                                      'psnr': avg_psnr, 'ssim': avg_ssim, 'bpp': avg_bpp})
                    print(f"[{time.strftime('%H:%M:%S')}] smin={smin} smax={smax} bp={bp} cw={cw}: PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} bpp={avg_bpp:.3f}")

    best_s2 = max(results_s2, key=lambda r: r['psnr'])
    print(f"\nBest S2: smin={best_s2['smin']}, smax={best_s2['smax']}, bp={best_s2['bp']}, cw={best_s2['cw']}")

    val_images = load_random_n_images('svgs', n=20, seed = 42)

    best_svg = []
    for sc in [85]:
        for ss in [120]:
            for ht in [18.0]:
                for lt in [0.0]:
                    with ProcessPoolExecutor(max_workers=10) as pool:
                        scores = list(pool.map(_run_one, [(p, {'sigmaColor': sc, 'sigmaSpace': ss, 'high_thresh': ht, 'low_thresh': lt}) for p in val_images]))
                    avg_psnr = np.mean([s['psnr'] for s in scores])
                    avg_ssim = np.mean([s['ssim'] for s in scores])
                    avg_bpp = np.mean([s['bpp'] for s in scores])
                    best_svg.append({'sigmaColor': sc, 'sigmaSpace': ss, 'high_thresh': ht, 'low_thresh': lt,
                                   'psnr': avg_psnr, 'ssim': avg_ssim, 'bpp': avg_bpp})
                    print(f"[{time.strftime('%H:%M:%S')}] sc={sc} ss={ss} ht={ht} lt={lt}: PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} bpp={avg_bpp:.3f}")

    best_svg = max(best_svg, key=lambda r: r['psnr'])
    print(f"\nBest SVG: sigmaColor={best_svg['sigmaColor']}, sigmaSpace={best_svg['sigmaSpace']}, high_thresh={best_svg['high_thresh']}, low_thresh={best_svg['low_thresh']}")

    best_s2_svg = []
    for smin in [0.0]:
        for smax in [9.0]:
            for bp in [225.0]:
                for cw in [(0.40, 0.60)]:
                    with ProcessPoolExecutor(max_workers=10) as pool:
                        scores = list(pool.map(_run_one, [(p, {'sigmaColor': best_svg['sigmaColor'],
                            'sigmaSpace': best_svg['sigmaSpace'],
                            'high_thresh': best_svg['high_thresh'], 'low_thresh': best_svg['low_thresh'],
                            'spline_min_smooth': smin, 'spline_max_smooth': smax,
                            'spline_base_perim': bp, 'complexity_weights': cw}) for p in val_images]))
                    avg_psnr = np.mean([s['psnr'] for s in scores])
                    avg_ssim = np.mean([s['ssim'] for s in scores])
                    avg_bpp = np.mean([s['bpp'] for s in scores])
                    best_s2_svg.append({'smin': smin, 'smax': smax, 'bp': bp, 'cw': cw,
                                      'psnr': avg_psnr, 'ssim': avg_ssim, 'bpp': avg_bpp})
                    print(f"[{time.strftime('%H:%M:%S')}] smin={smin} smax={smax} bp={bp} cw={cw}: PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} bpp={avg_bpp:.3f}")

    best_s2_svg = max(best_s2_svg, key=lambda r: r['psnr'])
    print(f"\nBest S2 SVG: smin={best_s2_svg['smin']}, smax={best_s2_svg['smax']}, bp={best_s2_svg['bp']}, cw={best_s2_svg['cw']}")

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print("\nBSDS500:")
    print(f"  S1: sigmaColor={best['sigmaColor']}, sigmaSpace={best['sigmaSpace']}, high_thresh={best['high_thresh']}, low_thresh={best['low_thresh']} (PSNR={best['psnr']:.2f})")
    print(f"  S2: smin={best_s2['smin']}, smax={best_s2['smax']}, bp={best_s2['bp']}, cw={best_s2['cw']} (PSNR={best_s2['psnr']:.2f})")

    print("\nSVG:")
    print(f"  S1: sigmaColor={best_svg['sigmaColor']}, sigmaSpace={best_svg['sigmaSpace']}, high_thresh={best_svg['high_thresh']}, low_thresh={best_svg['low_thresh']} (PSNR={best_svg['psnr']:.2f})")
    print(f"  S2: smin={best_s2_svg['smin']}, smax={best_s2_svg['smax']}, bp={best_s2_svg['bp']}, cw={best_s2_svg['cw']} (PSNR={best_s2_svg['psnr']:.2f})")
    print("="*60)