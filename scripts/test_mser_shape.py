"""
MSER matching via shape descriptors of normalized convex hulls.
For each MSER region:
  1. Compute convex hull
  2. Resample uniformly to N points
  3. Center + normalize by max radius
  4. Fourier descriptor (magnitudes of first K harmonics)
     -> invariant to translation, scale, rotation, start point
  5. Optional: append mean region intensity

Outputs:
  mser_regions.jpg     - MSER regions in both images
  mser_shape_matches.jpg - matches based on shape descriptor
"""

import cv2
import numpy as np
from pathlib import Path


def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / 'CMakeLists.txt').exists():
            return p
    raise RuntimeError('Repo root (CMakeLists.txt) not found')


BASE = _repo_root()
N_RESAMPLE = 128   # points resampled on convex hull
N_HARMONICS = 32   # Fourier harmonics kept


def load(p1, p2):
    img1, img2 = cv2.imread(p1), cv2.imread(p2)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2, g1, g2


# ---------------------------------------------------------------------------
# Shape descriptor
# ---------------------------------------------------------------------------

def resample_contour(pts, n):
    """Resample an open/closed polyline to n equally-spaced points."""
    pts = pts.astype(np.float64)
    closed = np.vstack([pts, pts[0]])
    diffs = np.diff(closed, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    total = seg_len.sum()
    if total < 1e-6:
        return None
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    targets = np.linspace(0.0, total, n, endpoint=False)
    result = []
    for t in targets:
        i = np.searchsorted(cum, t, side='right') - 1
        i = min(i, len(pts) - 1)
        ratio = (t - cum[i]) / (seg_len[i] + 1e-12)
        result.append(pts[i] + ratio * (pts[(i + 1) % len(pts)] - pts[i]))
    return np.array(result)


def fourier_shape_descriptor(region_pts, n_harmonics=N_HARMONICS, n_resample=N_RESAMPLE):
    """
    Fourier descriptor of convex hull contour.
    Returns a vector of n_harmonics float32 values, or None if region is degenerate.
    """
    hull = cv2.convexHull(region_pts.reshape(-1, 1, 2))
    pts = hull.reshape(-1, 2)

    if len(pts) < 4:
        return None

    sampled = resample_contour(pts, n_resample)
    if sampled is None:
        return None

    # Complex representation, center at centroid
    z = sampled[:, 0] + 1j * sampled[:, 1]
    z -= z.mean()

    # DFT
    Z = np.fft.fft(z)

    # Normalize by magnitude of first harmonic (scale invariance)
    ref = np.abs(Z[1])
    if ref < 1e-9:
        return None

    # Take magnitudes (rotation + start-point invariance), skip DC
    desc = np.abs(Z[1:n_harmonics + 1]) / ref  # Z[1] itself becomes 1.0
    return desc.astype(np.float32)


def region_intensity(region_pts, gray):
    """Mean intensity of the region interior."""
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillPoly(mask, [region_pts.reshape(-1, 1, 2)], 255)
    return cv2.mean(gray, mask=mask)[0] / 255.0


# ---------------------------------------------------------------------------
# MSER detection + descriptor computation
# ---------------------------------------------------------------------------

def compute_all(gray, img, mser, use_intensity=True):
    regions, bboxes = mser.detectRegions(gray)

    keypoints, descriptors = [], []
    valid_regions = []

    for region, (x, y, w, h) in zip(regions, bboxes):
        desc = fourier_shape_descriptor(region)
        if desc is None:
            continue

        if use_intensity:
            intensity = region_intensity(region, gray)
            desc = np.append(desc, intensity)

        cx, cy = x + w / 2.0, y + h / 2.0
        keypoints.append(cv2.KeyPoint(cx, cy, float(max(w, h))))
        descriptors.append(desc)
        valid_regions.append(region)

    descriptors = np.array(descriptors, dtype=np.float32) if descriptors else None
    return keypoints, descriptors, valid_regions


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_regions(img, regions, color=(0, 220, 0)):
    out = img.copy()
    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.drawContours(out, [hull], -1, color, 2)
    return out


def save_matches(path, img1, kps1, img2, kps2, matches, max_draw=100):
    best = sorted(matches, key=lambda m: m.distance)[:max_draw]
    out = cv2.drawMatches(img1, kps1, img2, kps2, best, None,
                          matchColor=(0, 255, 0),
                          singlePointColor=(0, 160, 0),
                          flags=cv2.DrawMatchesFlags_DEFAULT)
    cv2.imwrite(path, out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    img1, img2, gray1, gray2 = load(
        f'{BASE}/data/in/lane1.jpg',
        f'{BASE}/data/in/lane2.jpg'
    )

    # positional: delta, min_area, max_area, max_variation, min_diversity
    mser = cv2.MSER_create(5, 300, 60000, 0.25, 0.2)

    kps1, desc1, regions1 = compute_all(gray1, img1, mser)
    kps2, desc2, regions2 = compute_all(gray2, img2, mser)
    print(f"Valid MSER regions: {len(kps1)} in lane1, {len(kps2)} in lane2")

    # Save region visualization
    vis = np.hstack([draw_regions(img1, regions1), draw_regions(img2, regions2)])
    cv2.imwrite(f'{BASE}/data/out/mser_regions.jpg', vis)
    print("  -> mser_regions.jpg")

    if desc1 is None or desc2 is None:
        print("Not enough valid regions.")
        return

    # Cross-check matching on shape descriptors
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda m: m.distance)
    print(f"  {len(matches)} matches (cross-check)")

    save_matches(f'{BASE}/data/out/mser_shape_matches.jpg',
                 img1, kps1, img2, kps2, matches)
    print("  -> mser_shape_matches.jpg")

    # Print top matches for inspection
    print("\nTop 10 matches (distance = shape dissimilarity):")
    for i, m in enumerate(matches[:10]):
        kp1, kp2 = kps1[m.queryIdx], kps2[m.trainIdx]
        print(f"  [{i+1}] dist={m.distance:.3f}  "
              f"lane1=({kp1.pt[0]:.0f},{kp1.pt[1]:.0f}) s={kp1.size:.0f}  "
              f"lane2=({kp2.pt[0]:.0f},{kp2.pt[1]:.0f}) s={kp2.size:.0f}")


if __name__ == '__main__':
    run()
