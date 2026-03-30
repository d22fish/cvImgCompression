# CV Image Compression Research Notebooks

This repository tracks iterative development of a segmentation-based image compression pipeline.  
The project evolved from a polynomial-boundary prototype into a more robust contour-driven system using deterministic filling, hole handling, and spline-based boundary models with fallback logic.

## Repository Layout

- `v1.0funcCompress.ipynb` to `v4.0FuncCompress.ipynb`: main versioned notebooks.
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
- 

## Technical Progression Summary

1. **Scalability:** moved away from brittle/global behavior toward component- and contour-oriented processing.
2. **Boundary modeling:** progressed from raw polynomial fits to Bezier and then closed B-spline workflows.
3. **Topology robustness:** expanded from painter-only ordering to explicit hole-aware representations in dedicated branches.
4. **Reconstruction reliability:** added deterministic fill behavior, closure enforcement, and fallback/coverage safeguards.
5. **Known remaining challenge:** region interior appearance model is still relatively simple for texture-heavy natural images.

## Serialization Notes

`imComp.txt` uses command-style records such as:
- `C;...;` for color entries
- `L;...;` for loop/polyline geometry
- `B;...;` for Bezier-related geometry

This format has been kept readable during development to support debugging and iterative refinement.
