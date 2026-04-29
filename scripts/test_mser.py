"""
MSER matching test on lane1/lane2.
Produces:
  - mser_regions.jpg  : MSER regions detected in each image (side by side)
  - mser_matches.jpg  : matched regions (side by side, like homography_matches.jpg)
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


def load(p1, p2):
    img1, img2 = cv2.imread(p1), cv2.imread(p2)
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    return img1, img2, g1, g2


def detect_mser(gray, mser):
    regions, bboxes = mser.detectRegions(gray)
    kps = []
    for (x, y, w, h) in bboxes:
        cx, cy = x + w / 2.0, y + h / 2.0
        kps.append(cv2.KeyPoint(cx, cy, float(max(w, h))))
    return kps, regions


def draw_regions(img, regions, color=(0, 220, 0)):
    out = img.copy()
    for region in regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.drawContours(out, [hull], -1, color, 1)
    return out


def run():
    img1, img2, gray1, gray2 = load(
        f'{BASE}/data/in/lane1.jpg',
        f'{BASE}/data/in/lane2.jpg'
    )

    # positional: delta, min_area, max_area, max_variation, min_diversity
    mser = cv2.MSER_create(5, 300, 60000, 0.25, 0.2)

    kps1, regions1 = detect_mser(gray1, mser)
    kps2, regions2 = detect_mser(gray2, mser)
    print(f"MSER: {len(kps1)} regions in lane1, {len(kps2)} regions in lane2")

    # Visualise regions
    vis = np.hstack([draw_regions(img1, regions1), draw_regions(img2, regions2)])
    cv2.imwrite(f'{BASE}/data/out/mser_regions.jpg', vis)
    print(f"  -> mser_regions.jpg")

    # SIFT descriptors at MSER keypoints
    sift = cv2.SIFT_create()
    kps1, desc1 = sift.compute(gray1, kps1)
    kps2, desc2 = sift.compute(gray2, kps2)

    if desc1 is None or desc2 is None or len(kps1) < 2 or len(kps2) < 2:
        print("Not enough keypoints to match.")
        return

    # Cross-check matching (plus robuste que ratio test sur des régions hétérogènes)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(bf.match(desc1, desc2), key=lambda m: m.distance)
    print(f"  {len(matches)} matches (cross-check)")

    out = cv2.drawMatches(img1, kps1, img2, kps2, matches[:120], None,
                          matchColor=(0, 255, 0),
                          singlePointColor=(0, 160, 0),
                          flags=cv2.DrawMatchesFlags_DEFAULT)
    cv2.imwrite(f'{BASE}/data/out/mser_matches.jpg', out)
    print(f"  -> mser_matches.jpg")


if __name__ == '__main__':
    run()
