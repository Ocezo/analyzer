"""
Standalone test: Dense SIFT, HOG, and LBP patch matching on lane1/lane2.
Produces matching visualizations similar to homography_matches.jpg.
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.feature import local_binary_pattern, hog


def _repo_root():
    for p in Path(__file__).resolve().parents:
        if (p / 'CMakeLists.txt').exists():
            return p
    raise RuntimeError('Repo root (CMakeLists.txt) not found')


BASE = _repo_root()


def load_images(path1, path2):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2, gray1, gray2


# ---------------------------------------------------------------------------
# Dense SIFT
# ---------------------------------------------------------------------------

def dense_sift_matches(img1, img2, gray1, gray2, step=16, scale=16, ratio=0.75):
    def grid_keypoints(h, w):
        return [cv2.KeyPoint(float(x), float(y), float(scale))
                for y in range(step // 2, h, step)
                for x in range(step // 2, w, step)]

    sift = cv2.SIFT_create()
    kps1, desc1 = sift.compute(gray1, grid_keypoints(*gray1.shape))
    kps2, desc2 = sift.compute(gray2, grid_keypoints(*gray2.shape))

    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [m for m, n in bf.knnMatch(desc1, desc2, k=2) if m.distance < ratio * n.distance]
    return list(kps1), list(kps2), good


# ---------------------------------------------------------------------------
# Patch-based descriptors (HOG and LBP)
# ---------------------------------------------------------------------------

def _hog_desc(patch):
    return hog(patch, orientations=8, pixels_per_cell=(8, 8),
               cells_per_block=(1, 1), visualize=False).astype(np.float32)


def _lbp_desc(patch):
    n_points = 16
    lbp = local_binary_pattern(patch, n_points, 2, method='uniform')
    hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-7
    return hist


def patch_match(gray1, gray2, descriptor_fn, step=16, patch_size=32, ratio=0.75):
    def extract(gray):
        h, w = gray.shape
        half = patch_size // 2
        kps, descs = [], []
        for y in range(half, h - half, step):
            for x in range(half, w - half, step):
                patch = gray[y - half:y + half, x - half:x + half]
                descs.append(descriptor_fn(patch))
                kps.append(cv2.KeyPoint(float(x), float(y), float(patch_size)))
        return kps, np.array(descs, dtype=np.float32)

    kps1, desc1 = extract(gray1)
    kps2, desc2 = extract(gray2)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [m for m, n in bf.knnMatch(desc1, desc2, k=2) if m.distance < ratio * n.distance]
    return kps1, kps2, good


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_match_image(path, img1, kps1, img2, kps2, matches, max_matches=150):
    if len(matches) > max_matches:
        matches = sorted(matches, key=lambda m: m.distance)[:max_matches]
    out = cv2.drawMatches(img1, kps1, img2, kps2, matches, None,
                          matchColor=(0, 255, 0),
                          singlePointColor=(0, 160, 0),
                          flags=cv2.DrawMatchesFlags_DEFAULT)
    cv2.imwrite(path, out)
    print(f"  {len(matches)} matches shown -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    img1, img2, gray1, gray2 = load_images(f'{BASE}/data/in/lane1.jpg',
                                            f'{BASE}/data/in/lane2.jpg')

    print("Dense SIFT...")
    kps1, kps2, matches = dense_sift_matches(img1, img2, gray1, gray2)
    save_match_image(f'{BASE}/data/out/dense_sift_matches.jpg',
                     img1, kps1, img2, kps2, matches)

    print("HOG patches...")
    kps1, kps2, matches = patch_match(gray1, gray2, _hog_desc)
    save_match_image(f'{BASE}/data/out/hog_matches.jpg',
                     img1, kps1, img2, kps2, matches)

    print("LBP patches...")
    kps1, kps2, matches = patch_match(gray1, gray2, _lbp_desc)
    save_match_image(f'{BASE}/data/out/lbp_matches.jpg',
                     img1, kps1, img2, kps2, matches)

    print("Done.")
