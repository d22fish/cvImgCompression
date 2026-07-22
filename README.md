# CV Image Compression Research Notebooks

This repository tracks iterative development of a segmentation-based image compression pipeline.  
The project evolved from a polynomial-boundary prototype into a more robust contour-driven system using deterministic filling, hole handling, and spline-based boundary models with fallback logic.

## Repository Layout

- `v1.0funcCompress.ipynb` to `v6.0FuncCompress.ipynb`: main versioned notebooks.
- `pics/`: input images used for testing.
- `Results/`: reconstructed output examples.
- `imComp.txt`: serialized geometry/color output used for debugging and reconstruction.
- `test.jpg`, `test.png`, `test.tiff`: small test assets.

## Version Evolution

### v1.0 Baseline Prototype from MPhil
- **Current file:** `v1.0funcCompress.ipynb`
- Implemented the first full pipeline: clustering, contour extraction, polynomial fitting, and reconstruction.
- Established the core serialization/reconstruction workflow.
- Main limitations at this stage were scalability and instability on larger/more complex images.

### v2.0 Nearest-Neighbor and Contour Cleanup
- **Current file:** `v2.0FuncCompress.ipynb`
- Reworked cluster assignment/contour behavior to improve boundary consistency.
- Corrected issues around fitting inputs by focusing on boundary structure rather than noisy full-region point sets.
- Improved practical stability of region tracing versus the original baseline.

### v2.1 Painter Ordering and Chebyshev Alternatives
- **Current file:** `v2.1FuncCompress.ipynb`
- Added painter-style rendering order (largest/background regions first, then smaller details).
- Evaluated alternatives to raw polynomial fitting (including Chebyshev-style paths) to improve numerical behavior.
- Improved reconstruction robustness where strict boundary overlap/order mattered.

### v2.2 Explicit Hole-Aware Branch
- **Current file:** `v2.2FuncCompress.ipynb`
- Added explicit handling for regions with inner rings/holes.
- Introduced loop writing/parsing helpers to encode topology more directly.
- Explored trade-off between topological correctness and encoding/implementation complexity.

### v3.0 Bezier Boundary Modeling and Painter's
- **Current file:** `v3.0FuncCompress.ipynb`
- Added piecewise cubic Bezier contour modeling.
- Added contour validity checks, closure controls, and improved boundary sampling logic.
- Served as a key comparison branch against polynomial and B-spline approaches.

### v3.1 Closed B-Spline Transition with Painter's
- **Current file:** `v3.1FuncCompress.ipynb`
- Shifted boundary modeling toward closed B-splines.
- Improved contour smoothness and closure consistency, especially on larger images (including 256x256 cases).
- Included stronger reconstruction safeguards (coverage checks and compatibility paths during decode).

### v3.2 Hole-Awareness and Chebyshev Hybrid
- **Current file:** `v3.2FuncCompress.ipynb`
- Combined explicit loop/topology handling with compact Chebyshev-style fitting.
- Focused on balancing boundary compactness with reconstruction reliability.
- Retained painter-style reconstruction logic while revisiting hole-aware behavior.

### v4.0 Integrated B-Spline, Painter's, and Fallback Path
- **Current file:** `v4.0FuncCompress.ipynb`
- Shifted from HSV to CIELAB
- Uses B-spline-first boundary encoding with fallback behavior when fit quality is insufficient.
- Strengthened decode coverage handling to reduce unfilled artifacts.
- Represents the current best candidate for stable end-to-end runs.
- Implemented bilateral filter to smooth out minor color variations while strictly preserving sharp edges
- Added planar parametric color region modeling
- Implemented hybrid codec that compresses difference of original and custom algorithm with dct and reconstructs

### v5.0 Ablation testing with splines vs explicit contour mapping and planar vs mean vs original color mapping
- **Current file:** `v5.0FuncCompress.ipynb`
- Ablation study implementation
    ☐ Exact traced contours + current region fill model: no spline fitting, fill using the exact raster contour from the segmentation stage. This removes spline error.
    ☐ Spline contours + current region fill model: the difference between A and B isolates boundary-model error.
    ☐ Exact traced contours + original per-pixel colours inside each predicted region: any remaining error in C mainly reflects mask or topology issues.
    ☐ Exact traced contours + a better region model: for example, per-region planar colour fit (see below) instead of mean colour. This tells you how much is due to the constant-colour interior assumption.

### v5.1 Ablation testing for polyFit vs Chebyshev vs B-spline vs Bezier curves, hold planar color mapping constant
- **Current file:** `v5.1FuncCompress.ipynb`
- Ablation study implementation
    ☐ Polynomial.Fit
    ☐ Chebyshev
    ☐ B-Splines
    ☐ Bezier

### v6.0 Integrated B-Spline, Painter's, and Fallback Path
- **Current file:** `v6.0FuncCompress.ipynb`
- Saved known good file, reviewed last by Dr. Town

### v7.0 Completed several bug fixes, updated painter's sort algorithm, replaced fixed stride downsampling, updated segmentation similarity, region connectivity --> 8, removed redundant mean color, adaptive thresh and spline_smooth, fixed repair_coverage
- **Current file:** `v7.0FuncCompress.ipynb`
- Fixed repair_coverage function to fix bug of returning too early
- Updated spline decoding to use adaptive perimeter-based sampling (n_samples scaled by estimated contour arc length and control-point count) so large S contours are reconstructed smoothly instead of being under-sampled.
- Updated Painter’s ordering to sort decoded regions by true polygon area (cv2.contourArea) instead of contour point count, improving overlap/occlusion robustness during reconstruction.
- Replaced fixed-stride contour downsampling in spline fitting with full-contour fitting and splprep smoothing (s>0), enabling adaptive control-point allocation and better size–quality tradeoffs.
- Updated segmentation similarity to use Euclidean distance in standard-like CIELAB space (OpenCV LAB remapped to L*∈[0,100], a*,b* centered at 0), replacing /255 channel normalization with an approximately ΔE-interpretable threshold. (HUGE IMPROVEMENT)
- Unified region connectivity to 8-neighborhood throughout segmentation and contour extraction (CCL, border detection, and tracing) to remove diagonal-connectivity mismatches and reduce subtle boundary artifacts.
- Removed redundant mean-color storage by eliminating C records and using M (planar coefficients) as the sole region header and color model source during decoding.
- Implemented thresh (segmentation) adaptive hyperparameter: compute an image-level complexity score from local LAB variability (e.g., median local std over 5x5/7x7). Higher complexity -> lower thresh; lower complexity -> higher thresh.
- Implemented contour-adaptive spline smoothing by deriving splprep’s s per boundary from geometric complexity (perimeter, turn-angle statistics, and corner density), using lower s for sharp/corner-rich contours and higher s for smoother/longer contours.
- Replaced the previous nested-loop coverage repair with a vectorized post-fill method using scipy.ndimage.binary_dilation and convolve to identify adjacent unpainted pixels and fill them by averaging painted neighbors. Added a nearest-painted-pixel fallback (distance_transform_edt) that activates only when needed, ensuring full reconstruction coverage and removing remaining white gaps.

### v7.1 Local hybrid DCT encoding and Floating to Fixed point quantization → Delta coding → Binary serialization → Entropy coding
- **Current file:** `v7.1FuncCompress.ipynb`
- Per-region DCT residual with m_k flag serialized to file
- Local rate-distortion metric for planar-only vs. planar+DCT decision (distinct from segmentation threshold)
- Replace floating-point coefficients with fixed-point/scaled integers
- Apply delta coding to coefficients before entropy coding
- Binary serialization of full record structure with actual byte counts
- Entropy coding applied to binary format (Huffman sufficient; arithmetic coding deferred to future work)
- S/L encoding ratio check per image (instrumented in encoding loop)

### v7.2 Copied v5.0 and updated to match v7.1 advances and following changes
- **Current file:** `v7.2FuncCompress.ipynb`
- Add LPIPS and MSE to ablation notebook
- Align with v7.0 pipeline (8-connected segmentation, same pixel set across all six conditions)
- Bugs Bunny ablation: C vs. F results and polyline fallback percentage
- Adaptive smoothing parameter ablation
- sigmaColor x threshold interaction grid
- Acquire and prepare standard datasets (Kodak, BSDS500, structured graphics set)
- Define validation/test split; tune all hyperparameters on validation set only

### v7.3 Copied v5.1 and updated to match v7.1 advances and following changes
- **Current file:** `v7.3FuncCompress.ipynb`
- Verify matched parameter budgets for B-spline vs. Bezier comparison

### v7.4 Held out Validation Set Hyperparameter Tuning
- **Current file:** `v7.4FuncCompress.ipynb`
- Put everything inside functions and iterating through hyperparameter options
- BSDS500: sigmaColor=190, sigmaSpace=145, high_thresh=28.0, low_thresh=10.5, smin=1.0, smax=4.5, bp=300.0, cw=(0.4, 0.6) (PSNR=28.38)
- SVG: sigmaColor=85, sigmaSpace=120, high_thresh=18.0, low_thresh=0.0, smin=0.0, smax=9.0, bp=225.0, cw=(0.4, 0.6) (PSNR=42.55)

## Technical Progression Summary

1. **Scalability:** moved away from brittle/global behavior toward component- and contour-oriented processing.
2. **Boundary modeling:** progressed from raw polynomial fits to Bezier and then closed B-spline workflows.
3. **Topology robustness:** expanded from painter-only ordering to explicit hole-aware representations in dedicated branches.
4. **Reconstruction reliability:** added deterministic fill behavior, closure enforcement, and fallback/coverage safeguards.
5. **Known remaining challenge:** region interior appearance model is still relatively simple for texture-heavy natural images.

## Serialization Notes

imComp.txt record format (per region)

C;L,a,b;

Mean region color in LAB (integer values).
M;a0,a1,a2;b0,b1,b2;c0,c1,c2;

Planar LAB model coefficients, one triple per channel.
Model: value(x,y) = a + b*x + c*y.
S;k;knots;ctrl_row;ctrl_col;

Spline boundary record.
k: spline degree (integer).
knots: comma-separated knot vector.
ctrl_row: comma-separated row control points.
ctrl_col: comma-separated column control points.
L;r,c;r,c;...;

Fallback loop/polyline boundary record.
Sequence of sampled contour points as row/column pairs.
Parsing and grouping rules

Fields are separated by semicolons (;).
Numeric tuples/lists inside a field are comma-separated (,).
A region is stored as:
C + M + (S or L)
followed by a blank line.
S is preferred; L is used when spline fitting fails.
