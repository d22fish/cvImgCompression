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
from numpy.polynomial import Chebyshev
from scipy.spatial import cKDTree
import torch
import warnings
from concurrent.futures import ProcessPoolExecutor
import time
import json

# %%
########### v7.2 ablation study #############
def encode_for_decomposition(img_lab, imgClusters, edgesSorted, clustMeans, path):
    with open(path, "w") as f:
        for i in range(len(edgesSorted)):
            x, y = edgesSorted[i][0], edgesSorted[i][1]
            contour = enforce_closed_contour_rc(x, y)
            if len(contour) < 6 or contour_area_rc(contour) < 4.0:
                continue
            s_val, _ = adaptive_spline_smooth(contour, min_smooth=0.0, max_smooth=20.0, base_perimeter=200.0)
            tck = fit_closed_bspline(contour, smooth=s_val, degree=3)
            wrote_geom = False
            geom_lines = []
            if tck is not None:
                t, c, k = tck
                cx, cy = c[0], c[1]
                if len(t) > 0 and len(cx) > 0 and len(cy) > 0:
                    t_str = ",".join(f"{float(v):.6f}" for v in t)
                    cx_str = ",".join(f"{float(v):.6f}" for v in cx)
                    cy_str = ",".join(f"{float(v):.6f}" for v in cy)
                    geom_lines.append("S;" + str(int(k)) + ";" + t_str + ";" + cx_str + ";" + cy_str + ";")
                    wrote_geom = True
            if not wrote_geom:
                temp = StringIO()
                if write_loop_L(temp, contour, max_pts=120):
                    geom_lines.append(temp.getvalue().strip())
                    wrote_geom = True
            if wrote_geom:
                col = clustMeans[i]
                cluster_id = i + 1
                region_mask = (imgClusters == cluster_id)
                coef, xc, yc = fit_region_planar_model(img_lab, region_mask)
                m0 = ",".join(f"{v:.6f}" for v in coef[0])
                m1 = ",".join(f"{v:.6f}" for v in coef[1])
                m2 = ",".join(f"{v:.6f}" for v in coef[2])
                f.write(f"C;{int(round(col[0]))},{int(round(col[1]))},{int(round(col[2]))};\n")
                f.write(f"M;{xc:.4f},{yc:.4f};{m0};{m1};{m2};\n")
                for line in geom_lines:
                    f.write(line + "\n")
                f.write("\n")

def run_decomposition(image_path, config):
    comp_path = f"imComp_{os.getpid()}.txt"
    try:
        chosenImage = cv2.imread(image_path)
        complexity_score, _ = local_lab_complexity_score(chosenImage, window_size=7)
        thresh = adaptive_deltaE_threshold(complexity_score, low_complexity=2.0, high_complexity=12.0,
                                            high_thresh=config['high_thresh'], low_thresh=config['low_thresh'])
        filtered = cv2.bilateralFilter(chosenImage, d=7, sigmaColor=config['sigmaColor'], sigmaSpace=config['sigmaSpace'])
        imgClusters, edgesSorted = run_segmentation(filtered, thresh)
        img_lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB).astype(float)

        clustMeans = []
        for i in range(len(edgesSorted)):
            region_mask = (imgClusters == i + 1)
            clustMeans.append(img_lab[region_mask].mean(axis=0) if region_mask.sum() > 0 else np.zeros(3))

        encode_for_decomposition(img_lab, imgClusters, edgesSorted, clustMeans, comp_path)

        orig_rgb = cv2.cvtColor(chosenImage, cv2.COLOR_BGR2RGB)
        orig_lab = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2LAB)

        A_rgb = cv2.cvtColor(recon_exact_mean(orig_lab, imgClusters, clustMeans), cv2.COLOR_LAB2RGB)
        B_rgb = cv2.cvtColor(recon_spline_mean_from_file(comp_path, imgClusters.shape), cv2.COLOR_LAB2RGB)
        C_rgb = cv2.cvtColor(recon_exact_original(orig_lab, imgClusters), cv2.COLOR_LAB2RGB)
        D_rgb = cv2.cvtColor(recon_exact_planar(orig_lab, imgClusters), cv2.COLOR_LAB2RGB)
        E_rgb = cv2.cvtColor(recon_spline_planar_actual(comp_path, imgClusters.shape), cv2.COLOR_LAB2RGB)
        F_rgb = cv2.cvtColor(recon_spline_original_from_file(comp_path, imgClusters.shape, orig_lab), cv2.COLOR_LAB2RGB)

        band_mask, interior_mask = boundary_band_from_labels(imgClusters, radius=2)

        dC = report("C exact+original", C_rgb, orig_rgb, band_mask, interior_mask)
        dD = report("D exact+planar", D_rgb, orig_rgb, band_mask, interior_mask)
        dE = report("E spline+planar", E_rgb, orig_rgb, band_mask, interior_mask)
        dF = report("F spline+original", F_rgb, orig_rgb, band_mask, interior_mask)

        dseg = dC
        dbnd = dF - dC
        dapp = dD - dC
        dpred = dseg + dbnd + dapp
        dint = dE - dpred
        return {'dseg': dseg, 'dbnd': dbnd, 'dapp': dapp, 'dint': dint, 'dE': dE}
    finally:
        if os.path.exists(comp_path):
            os.remove(comp_path)

# Spline reconstruction using planar model fill for each cluster
def recon_spline_planar_actual(imComp_path, img_shape_hw):
    # 1. Initialize recovery image and list to track cluster sizes for sorting
    H, W = img_shape_hw
    imRecover = np.zeros((H, W, 3), dtype=np.uint8)
    imRecover[:] = (255, 128, 128)
    clusters_to_draw = []

    # 2. Read the file once to collect all shapes and their data
    with open(imComp_path, "r") as f:
        current_cluster = None

        for line in f:
            l = line.strip().split(';')
            if len(l) == 0 or l[0] == '':
                continue
            
            # C is start of new cluster
            if l[0] == 'C':
                color = np.array([int(i) for i in l[1].split(',')], dtype=np.uint8)
                current_cluster = {'color': color, 'pts': [], 'model': None}
                clusters_to_draw.append(current_cluster)

            elif l[0] == 'M' and current_cluster is not None:
                xc, yc = [float(v) for v in l[1].split(',')]
                ch0 = np.array([float(v) for v in l[2].split(',')], dtype=float)  # [a,b,c]
                ch1 = np.array([float(v) for v in l[3].split(',')], dtype=float)
                ch2 = np.array([float(v) for v in l[4].split(',')], dtype=float)
                current_cluster['model'] = np.vstack([ch0, ch1, ch2])  # shape (3,3)
                current_cluster['xc'] = xc
                current_cluster['yc'] = yc

            elif l[0] == 'S' and current_cluster is not None:
                # S;degree;knots;ctrl_row;ctrl_col;
                k = int(l[1])
                t = np.array([float(v) for v in l[2].split(',')], dtype=float)
                cx = np.array([float(v) for v in l[3].split(',')], dtype=float)
                cy = np.array([float(v) for v in l[4].split(',')], dtype=float)

                tck = (t, [cx, cy], k)

                coarse_n = max(128, int(4 * len(cx)))
                coarse_rows, coarse_cols = splev(np.linspace(0.0, 1.0, coarse_n, endpoint=False), tck)

                drows = np.diff(np.r_[coarse_rows, coarse_rows[0]])
                dcols = np.diff(np.r_[coarse_cols, coarse_cols[0]])
                perimeter_estimate = np.sum(np.sqrt(drows**2 + dcols**2))

                n_samples = max(int(np.ceil(perimeter_estimate)), int(4 * len(cx)), 64)
                rows, cols = splev(np.linspace(0.0, 1.0, n_samples, endpoint=False), tck)

                for r, c in zip(rows, cols):
                    rr = int(round(r))
                    cc = int(round(c))
                    current_cluster['pts'].append([cc, rr])
            
            elif l[0] == 'L' and current_cluster is not None:
                for tok in l[1:]:
                    if not tok:
                        continue
                    rc = tok.split(',')
                    if len(rc) != 2:
                        continue
                    r = int(np.clip(round(float(rc[0])), 0, H-1))
                    c = int(np.clip(round(float(rc[1])), 0, W-1))
                    current_cluster['pts'].append([c, r])  # x,y

    # 3. Painter's algorithm sorted by polygon area
    for cluster in clusters_to_draw:
        if len(cluster['pts']) > 2:
            polygon_points = np.array(cluster['pts'], dtype=np.int32).reshape((-1, 1, 2))
            cluster['area'] = float(abs(cv2.contourArea(polygon_points)))
        else:
            cluster['area'] = 0.0
    
    clusters_to_draw.sort(key=lambda x: x['area'], reverse=True)

    # 4. Draw + track coverage (single pass)
    H, W = imRecover.shape[:2]
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

            if cluster.get('model') is None:
                raise ValueError("Missing M record for cluster")

            coef = cluster['model']  # shape (3,3)
            dx = xs - cluster['xc']
            dy = ys - cluster['yc']
            v0 = coef[0, 0] + coef[0, 1] * dx + coef[0, 2] * dy
            v1 = coef[1, 0] + coef[1, 1] * dx + coef[1, 2] * dy
            v2 = coef[2, 0] + coef[2, 1] * dx + coef[2, 2] * dy
            vals = np.stack([v0, v1, v2], axis=1)
            imRecover[ys, xs] = np.clip(vals, 0, 255).astype(np.uint8)

            painted[ys, xs] = 1
            drawn_clusters += 1

    # 5. Coverage + repair
    imRecover, painted, coverage = repair_coverage(imRecover, painted, max_iters=5, min_coverage=0.98)
    if coverage < 1.0:
        imRecover, painted, coverage = repair_coverage_nearest(imRecover, painted)

    return imRecover

#Spline reconstruction using mean color fill for each cluster
def recon_spline_mean_from_file(imComp_path, shape_hw):
    # 1. Initialize recovery image and list to track cluster sizes for sorting
    H, W = shape_hw
    imRecover = np.zeros((H, W, 3), dtype=np.uint8)
    imRecover[:] = (255, 128, 128)

    clusters_to_draw = []
    current_cluster = None

    # 2. Read the file once to collect all shapes and their data
    with open(imComp_path, "r") as f:
        for line in f:
            l = line.strip().split(';')
            if len(l) == 0 or l[0] == '':
                continue

            if l[0] == 'C':
                if len(l) >= 3 and ',' in l[2]:   # C;id;r,g,b;
                    color = np.array([int(v) for v in l[2].split(',')], dtype=np.uint8)
                else:                               # C;r,g,b;
                    color = np.array([int(v) for v in l[1].split(',')], dtype=np.uint8)
                current_cluster = {'color': color, 'pts': []}
                clusters_to_draw.append(current_cluster)

            elif l[0] == 'S' and current_cluster is not None:
                k = int(l[1])
                t = np.array([float(v) for v in l[2].split(',')], dtype=float)
                cx = np.array([float(v) for v in l[3].split(',')], dtype=float)
                cy = np.array([float(v) for v in l[4].split(',')], dtype=float)

                tck = (t, [cx, cy], k)

                coarse_n = max(128, int(4 * len(cx)))
                coarse_rows, coarse_cols = splev(np.linspace(0.0, 1.0, coarse_n, endpoint=False), tck)

                drows = np.diff(np.r_[coarse_rows, coarse_rows[0]])
                dcols = np.diff(np.r_[coarse_cols, coarse_cols[0]])
                perimeter_estimate = np.sum(np.sqrt(drows**2 + dcols**2))

                n_samples = max(int(np.ceil(perimeter_estimate)), int(4 * len(cx)), 64)
                rows, cols = splev(np.linspace(0.0, 1.0, n_samples, endpoint=False), tck)

                for r, c in zip(rows, cols):
                    rr, cc = int(round(r)), int(round(c))
                    if 0 <= rr < H and 0 <= cc < W:
                        current_cluster['pts'].append([cc, rr])

            elif l[0] == 'L' and current_cluster is not None:
                for tok in l[1:]:
                    if not tok:
                        continue
                    rc = tok.split(',')
                    if len(rc) != 2:
                        continue
                    r = int(np.clip(round(float(rc[0])), 0, H-1))
                    c = int(np.clip(round(float(rc[1])), 0, W-1))
                    current_cluster['pts'].append([c, r])

    # 3. Painter's algorithm sorted by polygon area
    for cluster in clusters_to_draw:
        if len(cluster['pts']) > 2:
            polygon_points = np.array(cluster['pts'], dtype=np.int32).reshape((-1, 1, 2))
            cluster['area'] = float(abs(cv2.contourArea(polygon_points)))
        else:
            cluster['area'] = 0.0

    clusters_to_draw.sort(key=lambda x: x['area'], reverse=True)

    # 4. Draw + track coverage (single pass)
    H, W = imRecover.shape[:2]
    painted = np.zeros((H, W), dtype=np.uint8)

    for cluster in clusters_to_draw:
        if len(cluster['pts']) > 2:
            polygon_points = np.array(cluster['pts'], dtype=np.int32).reshape((-1, 1, 2))
            region = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(region, [polygon_points], color=1)
            ys, xs = np.where(region == 1)

            if len(xs) == 0:
                continue

            imRecover[ys, xs] = cluster['color']
            painted[ys, xs] = 1

    # 5. Coverage + repair
    imRecover, painted, coverage = repair_coverage(imRecover, painted, max_iters=5, min_coverage=0.98)
    if coverage < 1.0:
        imRecover, painted, coverage = repair_coverage_nearest(imRecover, painted)

    return imRecover

# Spline reconstruction using original colors for each cluster
def recon_spline_original_from_file(imComp_path, img_shape_hw, orig_lab):
    # 1. Initialize recovery image and list to track cluster sizes for sorting
    H, W = img_shape_hw
    imRecover = np.zeros((H, W, 3), dtype=np.uint8)
    imRecover[:] = (255, 128, 128) 
    clusters_to_draw = []
    current_cluster = None
    next_id = 1

    # 2. Read the file once to collect all shapes and their data
    with open(imComp_path, "r") as f:
        for line in f:
            l = line.strip().split(';')
            if len(l) == 0 or l[0] == '':
                continue

            if l[0] == 'C':
                # supports C;id;r,g,b; and C;r,g,b;
                if len(l) >= 3 and ',' in l[2]:
                    color = np.array([int(v) for v in l[2].split(',')], dtype=np.uint8)
                else:
                    color = np.array([int(v) for v in l[1].split(',')], dtype=np.uint8)
                    next_id += 1

                current_cluster = {'color': color, 'pts': []}
                clusters_to_draw.append(current_cluster)

            elif l[0] == 'S' and current_cluster is not None:
                k = int(l[1])
                t = np.array([float(v) for v in l[2].split(',')], dtype=float)
                cx = np.array([float(v) for v in l[3].split(',')], dtype=float)
                cy = np.array([float(v) for v in l[4].split(',')], dtype=float)
                tck = (t, [cx, cy], k)

                coarse_n = max(128, int(4 * len(cx)))
                coarse_rows, coarse_cols = splev(np.linspace(0.0, 1.0, coarse_n, endpoint=False), tck)
                
                drows = np.diff(np.r_[coarse_rows, coarse_rows[0]])
                dcols = np.diff(np.r_[coarse_cols, coarse_cols[0]])
                perimeter_estimate = np.sum(np.sqrt(drows**2 + dcols**2))
                
                n_samples = max(int(np.ceil(perimeter_estimate)), int(4 * len(cx)), 64)
                rows, cols = splev(np.linspace(0.0, 1.0, n_samples, endpoint=False), tck)

                for r, c in zip(rows, cols):
                    rr, cc = int(round(r)), int(round(c))
                    if 0 <= rr < H and 0 <= cc < W:
                        current_cluster['pts'].append([cc, rr])  # x,y

            elif l[0] == 'L' and current_cluster is not None:
                for tok in l[1:]:
                    if not tok:
                        continue
                    rc = tok.split(',')
                    if len(rc) != 2:
                        continue
                    r = int(np.clip(round(float(rc[0])), 0, H - 1))
                    c = int(np.clip(round(float(rc[1])), 0, W - 1))
                    current_cluster['pts'].append([c, r])  # x,y

    for cluster in clusters_to_draw:
        if len(cluster['pts']) > 2:
            polygon_points = np.array(cluster['pts'], dtype=np.int32).reshape((-1, 1, 2))
            cluster['area'] = float(abs(cv2.contourArea(polygon_points)))
        else:
            cluster['area'] = 0.0
    clusters_to_draw.sort(key=lambda x: x['area'], reverse=True)

    # 4. Draw + track coverage (single pass)
    painted = np.zeros((H, W), dtype=np.uint8)
    H, W = imRecover.shape[:2]

    for cluster in clusters_to_draw:
        if len(cluster['pts']) > 2:
            polygon_points = np.array(cluster['pts'], dtype=np.int32).reshape((-1, 1, 2))
            
            region = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(region, [polygon_points], color=1)
            ys, xs = np.where(region == 1)
            
            if len(xs) == 0:
                continue

            # ORIGINAL per-pixel interior
            imRecover[ys, xs] = orig_lab[ys, xs]
            painted[ys, xs] = 1

    # 5. Coverage + repair
    imRecover, painted, coverage = repair_coverage(imRecover, painted, max_iters=5, min_coverage=0.98)
    if coverage < 1.0:
        imRecover, painted, coverage = repair_coverage_nearest(imRecover, painted)

    return imRecover

# Exact reconstruction using original colors for each cluster
def recon_exact_original(img_lab, imgClusters):
    imRecover = np.zeros_like(img_lab, dtype=np.uint8)
    cluster_ids = sorted([int(i) for i in np.unique(imgClusters) if int(i) > 0])

    for cluster_id in cluster_ids:
        region_mask = (imgClusters == cluster_id)
        imRecover[region_mask] = img_lab[region_mask]

    return imRecover

# Exact reconstruction using mean color for each cluster
def recon_exact_mean(img_lab, imgClusters, clustMeans):
    imRecover = np.zeros_like(img_lab, dtype=np.uint8)
    cluster_ids = sorted([int(i) for i in np.unique(imgClusters) if int(i) > 0])

    for cluster_id in cluster_ids:
        region_mask = (imgClusters == cluster_id)
        imRecover[region_mask] = np.array(clustMeans[cluster_id - 1], dtype=np.uint8)

    return imRecover

# Exact reconstruction using planar model fit for each cluster
def recon_exact_planar(img_lab, imgClusters):
    imRecover = np.zeros_like(img_lab, dtype=np.uint8)
    cluster_ids = sorted([int(i) for i in np.unique(imgClusters) if int(i) > 0])

    for cluster_id in cluster_ids:
        region_mask = (imgClusters == cluster_id)
        ys, xs = np.where(region_mask)
        if len(xs) == 0:
            continue

        coef, xc, yc = fit_region_planar_model(img_lab, region_mask)  # (3,3), [a,b,c] per channel
        dx = xs - xc
        dy = ys - yc
        v0 = coef[0, 0] + coef[0, 1] * dx + coef[0, 2] * dy
        v1 = coef[1, 0] + coef[1, 1] * dx + coef[1, 2] * dy
        v2 = coef[2, 0] + coef[2, 1] * dx + coef[2, 2] * dy
        vals = np.stack([v0, v1, v2], axis=1)
        imRecover[ys, xs] = np.clip(vals, 0, 255).astype(np.uint8)

    return imRecover

# Return metrics for each reconstruction, overall and split by boundary/interior regions
def report(name, rec_rgb, orig_rgb, band_mask, interior_mask):
    full_mse   = compute_mse(orig_rgb, rec_rgb)
    full_mae = masked_mae(orig_rgb, rec_rgb, np.ones(orig_rgb.shape[:2], dtype=bool))
    full_psnr = masked_psnr(orig_rgb, rec_rgb, np.ones(orig_rgb.shape[:2], dtype=bool))
    full_ssim = ssim(orig_rgb, rec_rgb, channel_axis=2, data_range=255)
    full_lpips = compute_lpips(orig_rgb, rec_rgb)

    b_mae = masked_mae(orig_rgb, rec_rgb, band_mask)
    i_mae = masked_mae(orig_rgb, rec_rgb, interior_mask)

    b_psnr = masked_psnr(orig_rgb, rec_rgb, band_mask)
    i_psnr = masked_psnr(orig_rgb, rec_rgb, interior_mask)

    b_ssim = masked_ssim(orig_rgb, rec_rgb, band_mask)
    i_ssim = masked_ssim(orig_rgb, rec_rgb, interior_mask)

    print(f"{name}:")
    print(f"  full      MSE={full_mse:.4f}, MAE={full_mae:.3f}, PSNR={full_psnr:.3f}, SSIM={full_ssim:.4f}, LPIPS={full_lpips:.4f}")
    print(f"  boundary  MAE={b_mae:.3f}, PSNR={b_psnr:.3f}, SSIM={b_ssim:.4f}")
    print(f"  interior  MAE={i_mae:.3f}, PSNR={i_psnr:.3f}, SSIM={i_ssim:.4f}")

    return full_mse

def aggregate_decomposition(images, config, label):
    with ProcessPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(_run_decomposition_one, [(p, config) for p in images]))
    avg = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    print(f"\n{label} (n={len(images)}): {avg}")
    return avg

########### v7.2 ablation study #############

########### v7.3 boundary encoding study #############
def run_boundary_comparison(image_path, config):
    comp_path = f"imComp_{os.getpid()}.txt"
    try:
        chosenImage = cv2.imread(image_path)
        complexity_score, _ = local_lab_complexity_score(chosenImage, window_size=7)
        thresh = adaptive_deltaE_threshold(complexity_score, low_complexity=2.0, high_complexity=12.0,
                                            high_thresh=config['high_thresh'], low_thresh=config['low_thresh'])
        filtered = cv2.bilateralFilter(chosenImage, d=7, sigmaColor=config['sigmaColor'], sigmaSpace=config['sigmaSpace'])
        imgClusters, edgesSorted = run_segmentation(filtered, thresh)
        img = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB).astype(float)
        H, W = img.shape[:2]
        min_area = 50.0

        orig_rgb = cv2.cvtColor(chosenImage, cv2.COLOR_BGR2RGB)
        band_mask, interior_mask = boundary_band_from_labels(imgClusters, radius=2)

        budgets = [8, 12, 16, 20, 24]
        byte_budgets = [120, 160, 200, 250, 300, 400]
        methods = {'B-spline': bspline_at_n, 'Bezier': bezier_at_n, 'Chebyshev': chebyshev_at_n, 'Polynomial': poly_at_n}

        result = {'param_matched': {m: {n: {} for n in budgets} for m in methods},
                'byte_matched': {m: {B: {} for B in byte_budgets} for m in methods}}

        iou_scores = {m: {n: [] for n in budgets} for m in methods}
        chamfer_scores = {m: {n: [] for n in budgets} for m in methods}
        for x_list, y_list in edgesSorted:
            contour = enforce_closed_contour_rc(x_list, y_list)
            if len(contour) < 8 or contour_area_rc(contour) < min_area or (len(contour) - 1) < max(budgets):
                continue
            cache, ok = {}, True
            for n in budgets:
                for name, fn in methods.items():
                    pts = fn(contour, n, H, W)
                    if pts is None or len(pts) < 3:
                        ok = False
                    cache[(n, name)] = pts
            if ok:
                for n in budgets:
                    for name in methods:
                        iou_scores[name][n].append(contour_iou(H, W, cache[(n, name)], contour))
                        chamfer_scores[name][n].append(symmetric_chamfer(cache[(n, name)], contour))

        for m in methods:
            for n in budgets:
                result['param_matched'][m][n]['iou'] = float(np.mean(iou_scores[m][n])) if iou_scores[m][n] else np.nan
                result['param_matched'][m][n]['chamfer'] = float(np.mean(chamfer_scores[m][n])) if chamfer_scores[m][n] else np.nan

        for m, fn in methods.items():
            for n in budgets:
                rec_rgb = compress_fixed_n(fn, n, img, imgClusters, edgesSorted, min_area, comp_path)
                result['param_matched'][m][n]['mae'] = masked_mae(orig_rgb, rec_rgb, band_mask)
                result['param_matched'][m][n]['psnr'] = masked_psnr(orig_rgb, rec_rgb, band_mask)
                result['param_matched'][m][n]['ssim'] = masked_ssim(orig_rgb, rec_rgb, band_mask)

        n_at = {B: {m: max_n_for_bytes(m, B) for m in methods} for B in byte_budgets}
        for B in byte_budgets:
            for m, fn in methods.items():
                n = n_at[B][m]
                rec_rgb = compress_fixed_n(fn, n, img, imgClusters, edgesSorted, min_area, comp_path)
                result['byte_matched'][m][B]['n'] = n
                result['byte_matched'][m][B]['mae'] = masked_mae(orig_rgb, rec_rgb, band_mask)
                result['byte_matched'][m][B]['psnr'] = masked_psnr(orig_rgb, rec_rgb, band_mask)
                result['byte_matched'][m][B]['ssim'] = masked_ssim(orig_rgb, rec_rgb, band_mask)
        return result
    finally:
        if os.path.exists(comp_path):
            os.remove(comp_path)


def compress_fixed_n(fit_fn, n, img, imgClusters, edgesSorted, min_area, path):
    with open(path, 'w') as f:
        for i, (x_list, y_list) in enumerate(edgesSorted):
            contour = enforce_closed_contour_rc(x_list, y_list)
            if len(contour) < 3:
                continue
            cluster_id = i + 1
            region_mask = (imgClusters == cluster_id)
            coef, xc, yc = fit_region_planar_model(img, region_mask)
            m0 = ','.join(f'{v:.6f}' for v in coef[0])
            m1 = ','.join(f'{v:.6f}' for v in coef[1])
            m2 = ','.join(f'{v:.6f}' for v in coef[2])
            f.write(f'M;{xc:.4f},{yc:.4f};{m0};{m1};{m2};\n')
            pts = None
            if len(contour) >= 8 and contour_area_rc(contour) >= min_area and (len(contour) - 1) >= n:
                pts = fit_fn(contour, n, img.shape[0], img.shape[1])
            if pts is None or len(pts) < 3:
                pts = subsample_to_n(contour, n)
            payload = ';'.join(f'{float(r):.4f},{float(c):.4f}' for r, c in pts)
            f.write(f'L;{payload};\n\n')
    rec_lab = decode_planar_any_geometry(path, img.shape[:2])
    return cv2.cvtColor(rec_lab, cv2.COLOR_LAB2RGB)

def geom_bytes(method, n, k=3):
    if method == 'B-spline':
        n_knots = n + k + 1
        return 4 + n_knots * 2 + 2 + n * 4 * 2
    elif method == 'Bezier':
        return 3 + n * 4 * 2
    elif method in ('Chebyshev', 'Polynomial'):
        n_segs = max(1, n // (k + 1))
        return 4 + n_segs * 2 * (k + 1) * 4

def max_n_for_bytes(method, budget):
    for n in range(60, 3, -1):
        if geom_bytes(method, n) <= budget:
            return n
    return None

def decode_planar_any_geometry(imComp_path, shape_hw):
    # 1. Initialize recovery image and list to track cluster sizes for sorting
    H, W = shape_hw
    imRecover = np.zeros((H, W, 3), dtype=np.uint8)
    imRecover[:] = (255, 128, 128) 
    clusters_to_draw = []

    # 2. Read the file once to collect all shapes and their data
    with open(imComp_path, 'r') as f:
        current_cluster = None

        for line in f:
            l = line.strip().split(';')
            if len(l) == 0 or l[0] == '':
                continue

            tag = l[0]
            if tag == 'M':
                xc, yc = [float(v) for v in l[1].split(',')]
                ch0 = np.array([float(v) for v in l[2].split(',')], dtype=float)
                ch1 = np.array([float(v) for v in l[3].split(',')], dtype=float)
                ch2 = np.array([float(v) for v in l[4].split(',')], dtype=float)
                
                current_cluster = {'model': np.vstack([ch0, ch1, ch2]), 'xc': xc, 'yc': yc, 'pts': []}
                clusters_to_draw.append(current_cluster)

            elif tag == 'S' and current_cluster is not None:
                k = int(l[1])
                t = np.array([float(v) for v in l[2].split(',')], dtype=float)
                cx = np.array([float(v) for v in l[3].split(',')], dtype=float)
                cy = np.array([float(v) for v in l[4].split(',')], dtype=float)
                
                tck = (t, [cx, cy], k)

                coarse_n = max(128, int(4 * len(cx)))
                coarse_rows, coarse_cols = splev(np.linspace(0.0, 1.0, coarse_n, endpoint=False), tck)

                drows = np.diff(np.r_[coarse_rows, coarse_rows[0]])
                dcols = np.diff(np.r_[coarse_cols, coarse_cols[0]])
                perimeter_estimate = np.sum(np.sqrt(drows**2 + dcols**2))

                n_samples = max(int(np.ceil(perimeter_estimate)), int(4 * len(cx)), 64)
                rows, cols = splev(np.linspace(0.0, 1.0, n_samples, endpoint=False), tck)
                
                for r, c in zip(rows, cols):
                    rr = int(round(r))
                    cc = int(round(c))
                    current_cluster['pts'].append([cc, rr])

            elif tag == 'B' and current_cluster is not None:
                p0 = np.array([float(v) for v in l[1].split(',')], dtype=float)
                p1 = np.array([float(v) for v in l[2].split(',')], dtype=float)
                p2 = np.array([float(v) for v in l[3].split(',')], dtype=float)
                p3 = np.array([float(v) for v in l[4].split(',')], dtype=float)
                chord = np.linalg.norm(p3 - p0)
                n_samples = max(12, int(chord * 2.0))
                seg_pts = sample_bezier_rc(p0, p1, p2, p3, H, W, n_samples)
                for r, c in seg_pts:
                    current_cluster['pts'].append([c, r])  # x,y


            elif tag == 'P' and current_cluster is not None:
                startPt = int(l[1])
                poly_coeffs = [float(i) for i in l[2].split(',')]
                p = np.poly1d(poly_coeffs)
                endPt = int(l[3])
                step = 1 if endPt >= startPt else -1
                for r in range(startPt, endPt + step, step):
                    rr = int(np.clip(r, 0, H - 1))
                    cc = int(np.clip(round(p(r)), 0, W - 1))
                    current_cluster['pts'].append([cc, rr])

            elif tag == 'V' and current_cluster is not None:
                r_start, c_start = [int(k) for k in l[1].split(',')]
                r_end, c_end = [int(k) for k in l[2].split(',')]
                steps = max(abs(r_end - r_start), abs(c_end - c_start))
                if steps > 0:
                    for s in range(steps + 1):
                        rr = int(r_start + s * (r_end - r_start) / steps)
                        cc = int(c_start + s * (c_end - c_start) / steps)
                        rr = int(np.clip(rr, 0, H - 1))
                        cc = int(np.clip(cc, 0, W - 1))
                        current_cluster['pts'].append([cc, rr])

            elif tag == 'L' and current_cluster is not None:
                for tok in l[1:]:
                    if not tok:
                        continue
                    rc = tok.split(',')
                    if len(rc) != 2:
                        continue
                    r = int(np.clip(round(float(rc[0])), 0, H - 1))
                    c = int(np.clip(round(float(rc[1])), 0, W - 1))
                    current_cluster['pts'].append([c, r])

    # 3. Painter's algorithm sorted by polygon area
    for cluster in clusters_to_draw:
        if len(cluster['pts']) > 2:
            polygon_points = np.array(cluster['pts'], dtype=np.int32).reshape((-1, 1, 2))
            cluster['area'] = float(abs(cv2.contourArea(polygon_points)))
        else:
            cluster['area'] = 0.0
        
    clusters_to_draw.sort(key=lambda c: c['area'], reverse=True)

    # 4. Draw + track coverage (single pass)
    H, W = imRecover.shape[:2]
    painted = np.zeros((H, W), dtype=np.uint8)

    for cluster in clusters_to_draw:
        if len(cluster['pts']) <= 2:
            continue
        polygon_points = np.array(cluster['pts'], dtype=np.int32).reshape((-1, 1, 2))
        
        region = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(region, [polygon_points], color=1)
        ys, xs = np.where(region == 1)
        
        if len(xs) == 0:
            continue

        coef = cluster['model']
        dx = xs - cluster['xc']
        dy = ys - cluster['yc']
        v0 = coef[0, 0] + coef[0, 1] * dx + coef[0, 2] * dy
        v1 = coef[1, 0] + coef[1, 1] * dx + coef[1, 2] * dy
        v2 = coef[2, 0] + coef[2, 1] * dx + coef[2, 2] * dy
        vals = np.stack([v0, v1, v2], axis=1)
        imRecover[ys, xs] = np.clip(vals, 0, 255).astype(np.uint8)

        painted[ys, xs] = 1

    #  5. Coverage + repair
    imRecover, painted, coverage = repair_coverage(imRecover, painted, max_iters=5, min_coverage=0.98)
    if coverage < 1.0:
        imRecover, painted, coverage = repair_coverage_nearest(imRecover, painted)

    return imRecover

def fit_param_segment(seg_pts, H, W, deg=3, n_out=24):
    rows = np.array([p[0] for p in seg_pts], dtype=float)
    cols = np.array([p[1] for p in seg_pts], dtype=float)
    t = np.linspace(0.0, 1.0, len(seg_pts))
    tt = np.linspace(0.0, 1.0, max(8, int(n_out)))
    fr = Chebyshev.fit(t, rows, deg)
    fc = Chebyshev.fit(t, cols, deg)
    rr = fr(tt)
    cc = fc(tt)
    out = []
    for r, c in zip(rr, cc):
        rr_i = int(np.clip(round(r), 0, H-1))
        cc_i = int(np.clip(round(c), 0, W-1))
        if not out or out[-1] != (rr_i, cc_i):
            out.append((rr_i, cc_i))
    return out

# Piecewise cubic Bezier helpers
def contour_polygon_area(contour_pts):
    if len(contour_pts) < 4:
        return 0.0
    pts = contour_pts[:-1] if contour_pts[0] == contour_pts[-1] else contour_pts
    if len(pts) < 3:
        return 0.0
    x = np.array([p[1] for p in pts], dtype=float)
    y = np.array([p[0] for p in pts], dtype=float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def simplify_contour_dp(contour_pts, eps_ratio=0.01):
    # Simplify contour with Douglas-Peucker in (x,y) = (col,row)
    if len(contour_pts) < 6:
        return contour_pts

    base = contour_pts[:-1] if contour_pts[0] == contour_pts[-1] else contour_pts
    if len(base) < 4:
        return contour_pts

    arr = np.array([[p[1], p[0]] for p in base], dtype=np.float32).reshape((-1, 1, 2))
    peri = cv2.arcLength(arr, True)
    eps = max(0.5, eps_ratio * peri)
    approx = cv2.approxPolyDP(arr, eps, True).reshape((-1, 2))

    simp = [(int(round(y)), int(round(x))) for x, y in approx]
    if len(simp) < 4:
        return contour_pts

    if simp[0] != simp[-1]:
        simp.append(simp[0])
    return simp

def choose_adaptive_stride(contour_pts):
    # Larger stride for longer/smoother contours to reduce micro-segments
    n = len(contour_pts)
    area = contour_polygon_area(contour_pts)
    if n < 20 or area < 20:
        return 2
    if n < 40 or area < 60:
        return 3
    if n < 80 or area < 150:
        return 4
    return 6

def build_bezier_segments(contour_rc, stride=4, alpha=0.10, min_chord=2.0):
    base = contour_rc[:-1] if contour_rc[0] == contour_rc[-1] else contour_rc
    n = len(base)
    if n < 4:
        return []

    stride = max(1, int(stride))
    anchors = [base[i] for i in range(0, n, stride)]
    if len(anchors) < 4:
        anchors = base[:]

    k = len(anchors)
    segs = []
    for i in range(k):
        p0 = np.array(anchors[(i - 1) % k], dtype=float)
        p1 = np.array(anchors[i], dtype=float)
        p2 = np.array(anchors[(i + 1) % k], dtype=float)
        p3 = np.array(anchors[(i + 2) % k], dtype=float)

        b0 = p1
        b1 = p1 + alpha * (p2 - p0)
        b2 = p2 - alpha * (p3 - p1)
        b3 = p2

        if np.linalg.norm(b3 - b0) >= min_chord:
            segs.append((b0, b1, b2, b3))
    return segs

def sample_bezier_rc(p0, p1, p2, p3, H, W, n_samples):
    pts = []
    for t in np.linspace(0.0, 1.0, max(8, int(n_samples))):
        a = (1 - t) ** 3
        b = 3 * (1 - t) ** 2 * t
        c = 3 * (1 - t) * t ** 2
        d = t ** 3
        p = a * p0 + b * p1 + c * p2 + d * p3  # p = [row, col]
        r = int(np.clip(round(p[0]), 0, H - 1))
        c_ = int(np.clip(round(p[1]), 0, W - 1))
        if not pts or pts[-1] != (r, c_):
            pts.append((r, c_))
    return pts

# Sub sampling methods for comparison
def subsample_to_n(contour, n):
    if len(contour) == 0: return contour
    base = contour[:-1] if contour[0] == contour[-1] else contour
    if len(base) <= n:
        return contour
    idx = np.round(np.linspace(0, len(base) - 1, n)).astype(int)
    sampled = [base[i] for i in idx]
    sampled.append(sampled[0])
    return sampled

def bspline_at_n(contour, n_target, H, W, max_iter=40):
    base = contour[:-1] if contour[0] == contour[-1] else contour
    if len(base) < n_target:
        return None
    rows = np.array([p[0] for p in base], dtype=float)
    cols = np.array([p[1] for p in base], dtype=float)
    s_lo, s_hi = 0.0, float(len(base)) * 100.0
    best_tck = None
    best_gap = 1e9
    for _ in range(max_iter):
        s_mid = 0.5 * (s_lo + s_hi)
        try:
            tck, _ = splprep([rows, cols], s=s_mid, per=True, k=3)
        except Exception:
            s_hi = s_mid
            continue
        n_ctrl = len(tck[1][0])
        gap = abs(n_ctrl - n_target)
        if gap < best_gap:
            best_tck, best_gap = tck, gap
        if n_ctrl > n_target:
            s_lo = s_mid
        elif n_ctrl < n_target:
            s_hi = s_mid
        else:
            break
    if best_tck is None or best_gap > 2:
        return None
    n_ctrl = len(best_tck[1][0])
    r, c = splev(np.linspace(0, 1, max(200, 4 * n_ctrl), endpoint=False), best_tck)
    return list(zip(r, c))

def bezier_at_n(contour, n, H, W):
    sub = subsample_to_n(contour, n)
    if len(sub) < 5: return None
    segs = build_bezier_segments(sub, stride=1, min_chord=0.0)
    if not segs: return None
    pts = []
    for b0, b1, b2, b3 in segs:
        pts.extend(sample_bezier_rc(b0, b1, b2, b3, H, W, max(8, int(np.linalg.norm(b3-b0)*2))))
    return pts if len(pts) > 2 else None

def chebyshev_at_n(contour, n, H, W, deg=3):
    base = contour[:-1] if contour[0] == contour[-1] else contour
    if len(base) < 4: return None
    n_segs  = max(1, n // (deg + 1))
    seg_len = max(4, len(base) // n_segs)
    fitted  = []
    for s in range(0, len(base), seg_len):
        seg = base[s : s + seg_len + 1]
        if len(seg) < 4: continue
        fitted.extend(fit_param_segment(seg, H, W, deg=deg, n_out=max(12, len(seg)*2)))
    return fitted if len(fitted) >= 3 else None

def poly_at_n(contour, n, H, W, deg=3):
    base = contour[:-1] if contour[0] == contour[-1] else contour
    if len(base) < 4: return None
    n_segs  = max(1, n // (deg + 1))
    seg_len = max(4, len(base) // n_segs)
    fitted  = []
    for s in range(0, len(base), seg_len):
        seg = base[s : s + seg_len + 1]
        if len(seg) < 4: continue
        rows = np.array([p[0] for p in seg], dtype=float)
        cols = np.array([p[1] for p in seg], dtype=float)
        t  = np.linspace(0.0, 1.0, len(seg))
        tt = np.linspace(0.0, 1.0, max(12, len(seg) * 2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.exceptions.RankWarning)
            pr = np.poly1d(np.polyfit(t, rows, deg))
            pc = np.poly1d(np.polyfit(t, cols, deg))
        for r, c in zip(pr(tt), pc(tt)):
            ri = int(np.clip(round(r), 0, H - 1))
            ci = int(np.clip(round(c), 0, W - 1))
            if not fitted or fitted[-1] != (ri, ci):
                fitted.append((ri, ci))
    return fitted if len(fitted) >= 3 else None

# Rasterize both contours as filled polygons and compute pixel-level IoU
def rasterize(pts_rc, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    arr = np.array([[c, r] for r, c in pts_rc], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [arr], 1)
    return mask

def contour_iou(H, W, pts_fit_rc, pts_orig_rc):
    m_fit  = rasterize(pts_fit_rc, H, W)
    m_orig = rasterize(pts_orig_rc, H, W)
    inter = int(np.sum((m_fit == 1) & (m_orig == 1)))
    union = int(np.sum((m_fit == 1) | (m_orig == 1)))
    return inter / union if union > 0 else 1.0

def symmetric_chamfer(pts_a, pts_b):
    a = np.array(pts_a, dtype=np.float64)
    b = np.array(pts_b, dtype=np.float64)
    
    # Distance from A to B
    tree_b = cKDTree(b)
    d_a2b, _ = tree_b.query(a)
    term_a = np.mean(d_a2b ** 2)  # Mean of squared distances
    
    # Distance from B to A
    tree_a = cKDTree(a)
    d_b2a, _ = tree_a.query(b)
    term_b = np.mean(d_b2a ** 2)  # Mean of squared distances
    
    return term_a + term_b

def aggregate_boundary_comparison(images, config, label):
    with ProcessPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(_run_boundary_one, [(p, config) for p in images]))
    methods = list(results[0]['param_matched'].keys())
    budgets = list(results[0]['param_matched'][methods[0]].keys())
    byte_budgets = list(results[0]['byte_matched'][methods[0]].keys())

    avg = {'param_matched': {m: {n: {} for n in budgets} for m in methods},
           'byte_matched': {m: {B: {} for B in byte_budgets} for m in methods}}
    for m in methods:
        for n in budgets:
            for metric in results[0]['param_matched'][m][n]:
                avg['param_matched'][m][n][metric] = float(np.nanmean([r['param_matched'][m][n][metric] for r in results]))
        for B in byte_budgets:
            for metric in results[0]['byte_matched'][m][B]:
                avg['byte_matched'][m][B][metric] = float(np.nanmean([r['byte_matched'][m][B][metric] for r in results]))

    print(f"\n{label} boundary comparison (n={len(images)}):")
    return avg

########### v7.3 boundary encoding study #############

########### v7.4 end-to-end functions #############
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
def check_int_cast(arr, scale, dtype, label, log):
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
    D0 = float(np.sum((orig - recon_lin[ys, xs]) ** 2))
    R0 = R_geom + 1 + 18
    J0 = D0 + lam * R0

    # tier 1: quadratic only
    recon_quad = reconstruct_region(coef_quad, region_mask, xc, yc)
    D1 = float(np.sum((orig - recon_quad[ys, xs]) ** 2))
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

    D2 = float(np.sum((orig - recon_dct[ys, xs]) ** 2))
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
        rd_lambda=100,
        rd_dct_quality=35,
        verbose=False
    ):
    min_points = 6
    min_area = 4.0
    encoded_ok = 0
    fallback_ok = 0
    dropped = 0
    tier_counts = [0, 0, 0]
    tier_pixel_counts = [0, 0, 0]
    tier_byte_counts = [0, 0, 0]
    tier_ctrl_pts = [[], [], []]
    # overflow logging
    CAST_LOG = {}
    CAST_LOG.clear()
    pid = os.getpid()
    txt_path = f"/tmp/imComp_{pid}.txt"
    bin_path = f"/tmp/imFullComp_{pid}.bin"
    huff_path = f"/tmp/imFullComp_{pid}.huff"
    with open(txt_path, "w") as f, open(bin_path, "wb") as fb:
        header_pos = fb.tell()
        FORMAT_VERSION = 1
        fb.write(struct.pack('<BBHHI', FORMAT_VERSION, rd_dct_quality, img.shape[0], img.shape[1], 0))
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
            n_ctrl = 0

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
                    n_ctrl = len(cxD)

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
                    n_ctrl = len(dr)

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
                ys_r, _ = np.where(region_mask)
                tier_pixel_counts[tier] += len(ys_r)
                tier_ctrl_pts[tier].append(n_ctrl)
                if tier == 0:
                    tier_byte_counts[tier] += 23 + len(geom_bin)
                elif tier == 1:
                    tier_byte_counts[tier] += 59 + len(geom_bin)
                else:
                    tier_byte_counts[tier] += 33 + len(geom_bin) + (dct_payload['byte_cost'] if dct_payload else 0)
                xcQ, ycQ = int(round(xc)), int(round(yc))

                if tier == 0:
                    coefQ = check_int_cast(coef, 100, np.int16, "M.linear.tier0", CAST_LOG)
                    m0 = ",".join(str(v) for v in coefQ[0])
                    m1 = ",".join(str(v) for v in coefQ[1])
                    m2 = ",".join(str(v) for v in coefQ[2])
                    f.write(f"M;0;{xcQ},{ycQ};{m0};{m1};{m2};\n")
                    fb.write(struct.pack('<hh', xcQ, ycQ))
                    fb.write(struct.pack('<B', 0))
                    fb.write(coefQ.astype('<i2').tobytes())

                elif tier == 1:
                    coefQ_lin = check_int_cast(coef_quad[:, :3], 100,   np.int16, "M.quad.linear", CAST_LOG)
                    coefQ_sec = check_int_cast(coef_quad[:, 3:], 10000, np.int32, "M.quad.second", CAST_LOG)
                    m0 = ",".join(str(v) for v in np.concatenate([coefQ_lin[0], coefQ_sec[0]]))
                    m1 = ",".join(str(v) for v in np.concatenate([coefQ_lin[1], coefQ_sec[1]]))
                    m2 = ",".join(str(v) for v in np.concatenate([coefQ_lin[2], coefQ_sec[2]]))
                    f.write(f"M;1;{xcQ},{ycQ};{m0};{m1};{m2};\n")
                    fb.write(struct.pack('<hh', xcQ, ycQ))
                    fb.write(struct.pack('<B', 1))
                    fb.write(coefQ_lin.astype('<i2').tobytes())   # 3x3 int16 = 18 bytes
                    fb.write(coefQ_sec.astype('<i4').tobytes())   # 3x3 int32 = 12 bytes

                else:
                    coefQ = check_int_cast(coef, 100, np.int16, "M.linear.tier2", CAST_LOG)
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
                    fb.write(struct.pack('<I', len(nz_idx)))
                    for idx in nz_idx:
                        y_i, x_i, ch_i = idx
                        pos = int(ch_i) * ph * pw + int(y_i) * pw + int(x_i)
                        fb.write(struct.pack('<I', pos))
                        fb.write(struct.pack('<h', int(qcoeff[y_i, x_i, ch_i])))

                encoded_ok += 1
            else:
                dropped += 1
    with open(bin_path, "r+b") as fb:
        fb.seek(header_pos + 6)
        fb.write(struct.pack('<I', encoded_ok))
    spline_count = encoded_ok - fallback_ok
    total = encoded_ok
    if verbose:
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

    if verbose:
        print(f"[PID {pid}] Huffman: {os.path.getsize(huff_path)} Binary: {os.path.getsize(bin_path)} Text: {os.path.getsize(txt_path)}")

    total_px = max(sum(tier_pixel_counts), 1)
    total_b = max(sum(tier_byte_counts), 1)
    names = ['linear', 'quadratic', 'DCT']
    if verbose:
        for i, name in enumerate(names):
            print(f"Tier {i} ({name}): regions {tier_counts[i]}/{encoded_ok} ({tier_counts[i]/encoded_ok*100:.1f}%))"
                f"  pixels {tier_pixel_counts[i]/total_px*100:.1f}%"
                f"  bytes {tier_byte_counts[i]/total_b*100:.1f}%")

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
        _, dct_quality_hdr, H, W, n_regions = struct.unpack('<BBHHI', fb.read(10))

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
                n_nz = struct.unpack('<I', fb.read(4))[0]
                qcoeff = np.zeros((ph, pw, 3), dtype=np.int16)
                for _ in range(n_nz):
                    pos = struct.unpack('<I', fb.read(4))[0]
                    val = struct.unpack('<h', fb.read(2))[0]
                    ch_i = pos // (ph * pw)
                    rem = pos % (ph * pw)
                    y_i = rem // pw
                    x_i = rem % pw
                    qcoeff[y_i, x_i, ch_i] = val

                qt = quality_to_qtable(dct_quality_hdr)
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
    cov_raster = float(np.mean(painted))
    imRecoverBin, painted, cov_repair = repair_coverage(imRecoverBin, painted, max_iters=5, min_coverage=0.98)
    cov_nearest = cov_repair
    if cov_repair < 1.0:
        imRecoverBin, painted, cov_nearest = repair_coverage_nearest(imRecoverBin, painted)
    imRecoverBin = cv2.cvtColor(imRecoverBin, cv2.COLOR_LAB2RGB)
    return imRecoverBin, os.path.getsize(huff_path), CAST_LOG, {
        'tier_counts': tier_counts,
        'tier_pixel_counts': tier_pixel_counts,
        'tier_byte_counts': tier_byte_counts,
        'tier_ctrl_pts': tier_ctrl_pts,
        'spline_count': spline_count,
        'fallback_ok': fallback_ok,
        'dropped': dropped,
        'encoded_ok': encoded_ok,
        'cov_raster': cov_raster,
        'cov_repair': cov_repair,
        'cov_nearest': cov_nearest,
    }

# %%
# Hyperparameter tuning on validation set
def run_pipeline(image_path, sigmaColor=35, sigmaSpace=None,
                 high_thresh=8.0, low_thresh=4.0,
                 low_complexity=2.0, high_complexity=12.0, window_size=7,
                 spline_min_smooth=0.0, spline_max_smooth=20.0, spline_base_perim=200.0,
                 complexity_weights=(0.6, 0.4), corner_thresh_deg=35.0,
                 rd_lambda=100, rd_dct_quality=35, lpips_fn=None, verbose=False):
    
    # Run full encode-decode pipeline on one image, return metrics dict.
    if sigmaSpace is None:
        sigmaSpace = sigmaColor

    chosenImage = cv2.imread(image_path)
    if chosenImage is None:
        raise ValueError(f"could not read image: {image_path}")
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
    rec_rgb, file_size, CAST_LOG, diag = run_encode_decode(
        img, imgClusters, edgesSorted, chosenImage,
        spline_min_smooth=spline_min_smooth,
        spline_max_smooth=spline_max_smooth,
        spline_base_perim=spline_base_perim,
        complexity_weights=complexity_weights,
        corner_thresh_deg=corner_thresh_deg,
        rd_lambda=rd_lambda,
        rd_dct_quality=rd_dct_quality,
        verbose=verbose
    )

    band_mask, interior_mask = boundary_band_from_labels(imgClusters, radius=2)
    if lpips_fn is not None:
        orig_t = torch.from_numpy(orig_rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        rec_t  = torch.from_numpy(rec_rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            lpips_val = float(lpips_fn(orig_t, rec_t))
    else:
        lpips_val = np.nan
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
        'lpips': lpips_val,
        'diag': diag,
    }

def _run_one(args):
    path, kwargs = args
    return run_pipeline(path, **kwargs)

def _run_decomposition_one(args):
    path, config = args
    return run_decomposition(path, config)

def _run_boundary_one(args):
    path, config = args
    return run_boundary_comparison(path, config)

########### v7.4 end-to-end functions #############

########### Plotting and outputs ###########
# Convert HxWx3 uint8 RGB to BCHW float
def img_to_tensor(img_rgb):
    t = torch.from_numpy(img_rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    return (t / 127.5) - 1.0

def compute_lpips(orig_rgb, rec_rgb):
    with torch.no_grad():
        return float(lpips_fn(img_to_tensor(orig_rgb), img_to_tensor(rec_rgb)))

def compute_mse(orig_rgb, rec_rgb):
    return float(np.mean((orig_rgb.astype(np.float32) - rec_rgb.astype(np.float32)) ** 2))

def load_disjoint_splits(folder, sizes, seed=42):
    random.seed(seed)
    _img_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    filenames = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in _img_exts)
    if sum(sizes) > len(filenames):
        raise ValueError(f"Requested {sum(sizes)} images but folder has {len(filenames)}")
    random.shuffle(filenames)
    splits, idx = [], 0
    for size in sizes:
        splits.append([os.path.join(folder, f) for f in filenames[idx:idx + size]])
        idx += size
    return splits

def run_final_test(test_images, config):
    with ProcessPoolExecutor(max_workers=10) as pool:
        scores = list(pool.map(_run_one, [(p, config) for p in test_images]))
    avg_psnr = np.mean([s['psnr'] for s in scores])
    avg_ssim = np.mean([s['ssim'] for s in scores])
    avg_bpp  = np.mean([s['bpp']  for s in scores])
    print(f"\nFINAL TEST (n={len(test_images)}): PSNR={avg_psnr:.2f} SSIM={avg_ssim:.4f} bpp={avg_bpp:.3f}")
    return {'psnr': avg_psnr, 'ssim': avg_ssim, 'bpp': avg_bpp, 'per_image': scores}

def report_diagnostics(images, config, rd_lambda, label):
    cfg = dict(config)
    cfg['rd_lambda'] = rd_lambda
    cfg['verbose'] = False

    initial_total = 0
    encoded_total = 0
    spline_total = 0
    fallback_total = 0
    cov_raster_list = []
    cov_repair_list = []
    cov_nearest_list = []
    overflow_pcts = []

    for path in images:
        result = run_pipeline(path, **cfg)
        diag = result['diag']
        initial_total += result['n_regions']
        encoded_total += diag['encoded_ok']
        spline_total += diag['spline_count']
        fallback_total += diag['fallback_ok']
        cov_raster_list.append(diag['cov_raster'])
        cov_repair_list.append(diag['cov_repair'])
        cov_nearest_list.append(diag['cov_nearest'])

    print(f"\n{label} (lambda={rd_lambda}):")
    print(f"  Region count: initial={initial_total} encoded={encoded_total}")
    sl_ratio = spline_total / max(fallback_total, 1)
    print(f"  S/L ratio: {sl_ratio:.2f} (spline={spline_total}, fallback={fallback_total})")
    print(f"  Coverage repair: raster={100*np.mean(cov_raster_list):.2f}% "
          f"final_after_dilation_convolution={100*np.mean(cov_repair_list):.2f}% "
          f"final_after_nearest={100*np.mean(cov_nearest_list):.2f}%")

    return {
        'initial': initial_total, 'encoded': encoded_total, 'sl_ratio': sl_ratio,
        'cov_raster': np.mean(cov_raster_list), 'cov_repair': np.mean(cov_repair_list),
        'cov_nearest': np.mean(cov_nearest_list),
    }

def report_mode_allocation(images, config, rd_lambda, label):
    cfg = dict(config)
    cfg['rd_lambda'] = rd_lambda
    cfg['verbose'] = False

    mode_names = ['Linear', 'Quadratic', 'DCT']
    totals = {m: {'regions': 0, 'pixels': 0, 'bytes': 0, 'ctrl_pts': [], 'byte_costs': []} for m in mode_names}
    total_regions = 0
    total_pixels = 0
    total_bytes = 0

    for path in images:
        result = run_pipeline(path, **cfg)
        diag = result['diag']
        tier_counts = diag['tier_counts']
        tier_pixel_counts = diag['tier_pixel_counts']
        tier_byte_counts = diag['tier_byte_counts']

        total_regions += sum(tier_counts)
        total_pixels += sum(tier_pixel_counts)
        total_bytes += sum(tier_byte_counts)

        for i, m in enumerate(mode_names):
            totals[m]['regions'] += tier_counts[i]
            totals[m]['pixels'] += tier_pixel_counts[i]
            totals[m]['bytes'] += tier_byte_counts[i]
            totals[m]['ctrl_pts'].extend(diag['tier_ctrl_pts'][i])

    print(f"\n{label} (lambda={rd_lambda}):")
    print(f"{'Mode':<10} {'%Regions':>10} {'%Pixels':>10} {'%Bytes':>10} {'Avg|P|':>8} {'AvgBytes':>10}")
    for m in mode_names:
        pct_r = 100 * totals[m]['regions'] / max(total_regions, 1)
        pct_p = 100 * totals[m]['pixels'] / max(total_pixels, 1)
        pct_b = 100 * totals[m]['bytes'] / max(total_bytes, 1)
        avg_ctrl = np.mean(totals[m]['ctrl_pts']) if totals[m]['ctrl_pts'] else 0.0
        avg_bytes = totals[m]['bytes'] / max(totals[m]['regions'], 1)
        print(f"{m:<10} {pct_r:>9.1f}% {pct_p:>9.1f}% {pct_b:>9.1f}% {avg_ctrl:>8.1f} {avg_bytes:>10.1f}")
    return totals

def run_rd_sweep(images, config, lambdas, dct_qualities, lpips_fn=None):
    results = []
    for lam in lambdas:
        for dct_q in dct_qualities:
            cfg = dict(config)
            cfg.update({'rd_lambda': lam, 'rd_dct_quality': dct_q, 'lpips_fn': lpips_fn})
            scores = [run_pipeline(p, **cfg) for p in images]
            entry = {
                'lam': lam, 'dct_q': dct_q,
                'bpp':   np.mean([s['bpp']   for s in scores]),
                'psnr':  np.mean([s['psnr']  for s in scores]),
                'ssim':  np.mean([s['ssim']  for s in scores]),
                'lpips': np.nanmean([s['lpips'] for s in scores]),
            }
            results.append(entry)
            print(f"[{time.strftime('%H:%M:%S')}] lam={lam} dct_q={dct_q}: "
                  f"PSNR={entry['psnr']:.2f} SSIM={entry['ssim']:.4f} bpp={entry['bpp']:.3f}")
    return results

def encode_raster_baseline(images, codec, qualities):
    flag = cv2.IMWRITE_JPEG_QUALITY if codec == 'jpeg' else cv2.IMWRITE_WEBP_QUALITY
    ext = '.jpg' if codec == 'jpeg' else '.webp'
    results = []
    for q in qualities:
        scores = []
        for p in images:
            img_bgr = cv2.imread(p)
            orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            H, W = orig_rgb.shape[:2]
            _, buf = cv2.imencode(ext, img_bgr, [flag, q])
            rec_rgb = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            full = np.ones((H, W), dtype=bool)
            scores.append({
                'psnr': masked_psnr(orig_rgb, rec_rgb, full),
                'ssim': masked_ssim(orig_rgb, rec_rgb, full),
                'bpp':  len(buf) * 8 / (H * W),
            })
        results.append({
            'quality': q,
            'bpp':  np.mean([s['bpp']  for s in scores]),
            'psnr': np.mean([s['psnr'] for s in scores]),
            'ssim': np.mean([s['ssim'] for s in scores]),
        })
    return results

def plot_rd_curves(ours, baselines, metric='psnr', savepath=None):
    ylabel = {'psnr': 'PSNR (dB)', 'ssim': 'SSIM', 'lpips': 'LPIPS'}[metric]
    fig, ax = plt.subplots(figsize=(7, 5))
    pts = sorted(ours, key=lambda r: r['bpp'])
    ax.plot([r['bpp'] for r in pts], [r[metric] for r in pts], marker='o', label='Ours', linewidth=2)
    styles = {'jpeg': ('darkorange', '--'), 'webp': ('forestgreen', '--'), 'png': ('steelblue', ':')}
    for label, data in baselines.items():
        pts_b = sorted(data, key=lambda r: r['bpp'])
        color, ls = styles.get(label, ('gray', '--'))
        ax.plot([r['bpp'] for r in pts_b], [r[metric] for r in pts_b],
                marker='s', label=label.upper(), color=color, linestyle=ls, linewidth=2)
    ax.set_xlabel('Bits per pixel (bpp)')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    if savepath:
        plt.savefig(savepath, format='svg', bbox_inches='tight')
    return fig, ax


def run_threshold_sweep(images, sc_vals, tau_vals, base_config):
    grid = {}
    for sc in sc_vals:
        for tau in tau_vals:
            cfg = dict(base_config)
            cfg.update({'sigmaColor': sc, 'high_thresh': tau})
            scores = [run_pipeline(p, **cfg) for p in images]
            grid[(sc, tau)] = {
                'bpp':       np.mean([s['bpp']       for s in scores]),
                'psnr':      np.mean([s['psnr']      for s in scores]),
                'n_regions': np.mean([s['n_regions'] for s in scores]),
            }
            print(f"[{time.strftime('%H:%M:%S')}] sc={sc} tau={tau}: "
                  f"PSNR={grid[(sc,tau)]['psnr']:.2f} bpp={grid[(sc,tau)]['bpp']:.3f} "
                  f"regions={grid[(sc,tau)]['n_regions']:.0f}")
    return grid

def plot_threshold_sweep(grid, sc_vals, tau_vals, savepath=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sc_vals)))
    for sc, color in zip(sc_vals, colors):
        psnrs = [grid[(sc, tau)]['psnr']      for tau in tau_vals]
        bpps  = [grid[(sc, tau)]['bpp']       for tau in tau_vals]
        regs  = [grid[(sc, tau)]['n_regions'] for tau in tau_vals]
        axes[0].plot(tau_vals, psnrs, marker='o', label=f'σ={sc}', color=color, linewidth=2)
        axes[1].plot(tau_vals, bpps,  marker='o', label=f'σ={sc}', color=color, linewidth=2)
        axes[2].plot(tau_vals, regs,  marker='o', label=f'σ={sc}', color=color, linewidth=2)
    for ax, ylabel in zip(axes, ['PSNR (dB)', 'bpp', 'Region count']):
        ax.set_xlabel('τ (color threshold)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
    plt.show()
    if savepath:
        plt.savefig(savepath, format='svg', bbox_inches='tight')

def plot_boundary_fig1(avg, savepath=None):
    budgets = sorted(avg['param_matched'][next(iter(avg['param_matched']))].keys())
    byte_budgets = sorted(avg['byte_matched'][next(iter(avg['byte_matched']))].keys())
    colors = {'B-spline': 'steelblue', 'Bezier': 'darkorange', 'Chebyshev': 'forestgreen', 'Polynomial': 'crimson'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for m in avg['param_matched']:
        psnr_by_n = [avg['param_matched'][m][n]['psnr'] for n in budgets]
        ax1.plot(budgets, psnr_by_n, marker='o', label=m, color=colors[m], linewidth=2)

        psnr_by_byte = [avg['byte_matched'][m][B]['psnr'] for B in byte_budgets]
        ax2.plot(byte_budgets, psnr_by_byte, marker='o', label=m, color=colors[m], linewidth=2)

    ax1.set(xlabel='Control points N', ylabel='Boundary-band PSNR', title='Matched parameter budget')
    ax2.set(xlabel='Byte budget', ylabel='Boundary-band PSNR', title='Matched byte budget')
    ax1.legend(); ax2.legend(); ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    if savepath:
        plt.savefig(savepath, format='svg', bbox_inches='tight')

def run_timing_experiment(images, scales, config):
    tmp = '/tmp/_timing_img.png'
    results = []
    for scale in scales:
        runtimes, n_pix_list = [], []
        for p in images:
            img_bgr = cv2.imread(p)
            H, W = img_bgr.shape[:2]
            scaled = cv2.resize(img_bgr, (max(1, int(W * scale)), max(1, int(H * scale))))
            cv2.imwrite(tmp, scaled)
            t0 = time.perf_counter()
            run_pipeline(tmp, **config)
            runtimes.append(time.perf_counter() - t0)
            n_pix_list.append(scaled.shape[0] * scaled.shape[1])
        results.append({'scale': scale, 'n_pixels': np.mean(n_pix_list), 'runtime_s': np.mean(runtimes)})
        print(f"  scale={scale:.2f}x: {np.mean(n_pix_list)/1e6:.2f}Mpx  {np.mean(runtimes):.2f}s")
    return results

def plot_timing(results, savepath=None):
    pts = sorted(results, key=lambda r: r['n_pixels'])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r['n_pixels'] / 1e6 for r in pts], [r['runtime_s'] for r in pts], marker='o', linewidth=2)
    ax.set_xlabel('Megapixels')
    ax.set_ylabel('Runtime (s)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    if savepath:
        plt.savefig(savepath, format='svg', bbox_inches='tight')

def write_final_report(decomp_natural, decomp_structured, bnd_natural, bnd_structured,
                        mode_natural, mode_structured, diag_natural, diag_structured,
                        filepath='final_results.txt'):
    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\nTABLE V - Error decomposition\n" + "=" * 70 + "\n")
        for label, d in [('Natural', decomp_natural), ('Structured', decomp_structured)]:
            f.write(f"\n{label}:\n")
            for k, v in d.items():
                f.write(f"  {k:8s} = {v:.4f}\n")

        f.write("\n" + "=" * 70 + "\nTABLE VI - Matched parameter budget\n" + "=" * 70 + "\n")
        for label, d in [('Natural', bnd_natural), ('Structured', bnd_structured)]:
            f.write(f"\n{label}:\n")
            methods = list(d['param_matched'].keys())
            budgets = sorted(d['param_matched'][methods[0]].keys())
            for metric in ['mae', 'psnr', 'ssim', 'iou', 'chamfer']:
                f.write(f"\n  {metric.upper()}\n")
                f.write(f"  {'N':>6}" + "".join(f"{m:>14}" for m in methods) + "\n")
                for n in budgets:
                    row = f"  {n:>6}"
                    for m in methods:
                        row += f"{d['param_matched'][m][n][metric]:>14.4f}"
                    f.write(row + "\n")

        f.write("\n" + "=" * 70 + "\nTABLE VII - Matched byte budget\n" + "=" * 70 + "\n")
        for label, d in [('Natural', bnd_natural), ('Structured', bnd_structured)]:
            f.write(f"\n{label}:\n")
            methods = list(d['byte_matched'].keys())
            byte_budgets = sorted(d['byte_matched'][methods[0]].keys())
            for metric in ['n', 'mae', 'psnr', 'ssim']:
                f.write(f"\n  {metric.upper()}\n")
                f.write(f"  {'Budget':>8}" + "".join(f"{m:>14}" for m in methods) + "\n")
                for B in byte_budgets:
                    row = f"  {B:>8}"
                    for m in methods:
                        val = d['byte_matched'][m][B][metric]
                        row += f"{val:>14.4f}" if metric != 'n' else f"{val:>14d}"
                    f.write(row + "\n")

        f.write("\n" + "=" * 70 + "\nTABLE IX - Mode allocation\n" + "=" * 70 + "\n")
        for label, totals in [('Natural', mode_natural), ('Structured', mode_structured)]:
            f.write(f"\n{label}:\n")
            total_regions = sum(totals[m]['regions'] for m in totals)
            total_pixels = sum(totals[m]['pixels'] for m in totals)
            total_bytes = sum(totals[m]['bytes'] for m in totals)
            f.write(f"  {'Mode':<10} {'%Regions':>10} {'%Pixels':>10} {'%Bytes':>10} {'Avg|P|':>8} {'AvgBytes':>10}\n")
            for m in totals:
                pct_r = 100 * totals[m]['regions'] / max(total_regions, 1)
                pct_p = 100 * totals[m]['pixels'] / max(total_pixels, 1)
                pct_b = 100 * totals[m]['bytes'] / max(total_bytes, 1)
                avg_ctrl = np.mean(totals[m]['ctrl_pts']) if totals[m]['ctrl_pts'] else 0.0
                avg_bytes = totals[m]['bytes'] / max(totals[m]['regions'], 1)
                f.write(f"  {m:<10} {pct_r:>9.1f}% {pct_p:>9.1f}% {pct_b:>9.1f}% {avg_ctrl:>8.1f} {avg_bytes:>10.1f}\n")

        f.write("\n" + "=" * 70 + "\nTABLE X - Reconstruction diagnostics\n" + "=" * 70 + "\n")
        for label, d in [('Natural', diag_natural), ('Structured', diag_structured)]:
            f.write(f"\n{label}:\n")
            f.write(f"  Region count: initial={d['initial']} encoded={d['encoded']}\n")
            f.write(f"  S/L ratio: {d['sl_ratio']:.2f}\n")
            f.write(f"  Coverage repair: raster={100*d['cov_raster']:.2f}% "
                    f"dilation/convolution={100*d['cov_repair']:.2f}% "
                    f"nearest={100*d['cov_nearest']:.2f}%\n")

    print(f"\nFinal results written to {filepath}")

########### Plotting and outputs ###########s

# %% [markdown]
# Hyperparameter Tuning

# %%
with warnings.catch_warnings():
    warnings.simplefilter('ignore', np.exceptions.RankWarning)
    warnings.simplefilter('ignore', UserWarning)
    lpips_fn = lpips.LPIPS(net='alex', verbose=False)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Load all images
    tune_images, stability_images = load_disjoint_splits('BSDS500/val', [15, 30], seed=42)
    val_svgs, test_svgs = load_disjoint_splits('svgs', [15, 30], seed=42)
    kodak_images = [os.path.join('kodak', f) for f in sorted(os.listdir('kodak'))]

    # Read in hyperparameters
    config_bsds = json.load(open('tuned_config.json'))['natural']
    config_svg = json.load(open('tuned_config.json'))['structured']

    # Set up figure formatting
    FIGURE_DIR = 'paper_figures'
    os.makedirs(FIGURE_DIR, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'figure.dpi': 150})

    # ===== Run final end-to-end pipeline ===== 
    print("\nBSDS final:")
    run_final_test(kodak_images, config_bsds)
    print("\nStructured final:")
    run_final_test(test_svgs, config_svg)

    # ===== v7.2 Testing ===== 
    decomp_natural = aggregate_decomposition(kodak_images, config_bsds, 'Natural')
    decomp_structured = aggregate_decomposition(test_svgs, config_svg, 'Structured')

    # ===== v7.3 Testing ===== 
    bnd_natural = aggregate_boundary_comparison(kodak_images, config_bsds, 'Natural')
    bnd_structured = aggregate_boundary_comparison(test_svgs, config_svg, 'Structured')
    plot_boundary_fig1(bnd_natural, savepath='paper_figures/fig1_boundary_natural.svg')
    plot_boundary_fig1(bnd_structured, savepath='paper_figures/fig1_boundary_structured.svg')

    # ===== v7.4 Testing ===== 
    # R-D sweep — sweep lambda with fixed dct_quality; adjust ranges after lambda calibration
    lp_fn = lpips.LPIPS(net='alex', verbose=False)
    lam_grid_final = [10, 30, 60, 100, 200, 400, 700, 1000]
    dct_q_grid = [25, 35, 50]
    
    print("\nR-D sweep (natural):")
    rd_natural = run_rd_sweep(kodak_images, config_bsds, lam_grid_final, dct_q_grid, lpips_fn=lp_fn)
    print("\nR-D sweep (structured):")
    rd_structured = run_rd_sweep(test_svgs, config_svg, lam_grid_final, dct_q_grid, lpips_fn=lp_fn)
    
    jpeg_q = [10, 20, 30, 50, 70, 85, 95]
    webp_q = [10, 20, 30, 50, 70, 85, 95]
    jpeg_natural    = encode_raster_baseline(kodak_images, 'jpeg', jpeg_q)
    webp_natural    = encode_raster_baseline(kodak_images, 'webp', webp_q)
    jpeg_structured = encode_raster_baseline(test_svgs,   'jpeg', jpeg_q)
    webp_structured = encode_raster_baseline(test_svgs,   'webp', webp_q)
    
    print("\nR-D curves (natural):")
    plot_rd_curves(rd_natural, {'jpeg': jpeg_natural, 'webp': webp_natural}, metric='psnr', savepath='paper_figures/fig3_rd_natural_psnr.svg')
    plot_rd_curves(rd_natural, {'jpeg': jpeg_natural, 'webp': webp_natural}, metric='ssim', savepath='paper_figures/fig3_rd_natural_ssim.svg')
    plot_rd_curves(rd_natural, {'jpeg': jpeg_natural, 'webp': webp_natural}, metric='lpips', savepath='paper_figures/fig3_rd_natural_lpips.svg')
    
    print("\nR-D curves (structured):")
    plot_rd_curves(rd_structured, {'jpeg': jpeg_structured, 'webp': webp_structured}, metric='psnr', savepath='paper_figures/fig3_rd_structured_psnr.svg')
    plot_rd_curves(rd_structured, {'jpeg': jpeg_structured, 'webp': webp_structured}, metric='ssim', savepath='paper_figures/fig3_rd_structured_ssim.svg')
    plot_rd_curves(rd_structured, {'jpeg': jpeg_structured, 'webp': webp_structured}, metric='lpips', savepath='paper_figures/fig3_rd_structured_lpips.svg')
    
    # Threshold sweep for Fig 2 — run on stability set (disjoint from test)
    sc_sweep_natural  = [250, 300, 350, 400, 450]
    tau_sweep_natural = [10, 30, 60, 90, 120, 150, 180, 200, 250]
    sc_sweep_svg  = [55, 65, 75, 85, 95, 105]
    tau_sweep_svg = [5, 10, 14, 17, 20, 23, 25, 28, 30]

    print("\nThreshold sweep (natural):")
    thresh_grid_natural = run_threshold_sweep(stability_images[:8], sc_sweep_natural, tau_sweep_natural, config_bsds)
    plot_threshold_sweep(thresh_grid_natural, sc_sweep_natural, tau_sweep_natural, savepath='paper_figures/fig2_threshold_natural.svg')
    print("\nThreshold sweep (structured):")
    thresh_grid_svg = run_threshold_sweep(val_svgs[:8], sc_sweep_svg, tau_sweep_svg, config_svg)
    plot_threshold_sweep(thresh_grid_svg, sc_sweep_svg, tau_sweep_svg, savepath='paper_figures/fig2_threshold_structured.svg')
    
    # Table IX
    mode_natural = report_mode_allocation(kodak_images, config_bsds, rd_lambda=100, label='Natural')
    mode_structured = report_mode_allocation(test_svgs, config_svg, rd_lambda=100, label='Structured')
    
    # Table X
    diag_natural = report_diagnostics(kodak_images, config_bsds, rd_lambda=100, label='Natural')
    diag_structured = report_diagnostics(test_svgs, config_svg, rd_lambda=100, label='Structured')
    
    # Timing experiment for Fig 4
    timing_images = kodak_images[:6]
    scales = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    print("\nTiming experiment (natural):")
    timing_natural = run_timing_experiment(timing_images, scales, config_bsds)
    plot_timing(timing_natural, savepath='paper_figures/fig4_timing_natural.svg')
    print("\nTiming experiment (structured):")
    timing_structured = run_timing_experiment(test_svgs[:6], scales, config_svg)
    plot_timing(timing_structured, savepath='paper_figures/fig4_timing_structured.svg')

    # Write results to file
    write_final_report(decomp_natural, decomp_structured, bnd_natural, bnd_structured,
                        mode_natural, mode_structured, diag_natural, diag_structured)