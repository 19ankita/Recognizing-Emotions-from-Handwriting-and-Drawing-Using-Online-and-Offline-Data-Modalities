import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert

def extract_pseudo_dynamic_features(img):
    """
    img: numpy array (H,W,3) RGB HANDWRITING image
    returns: 5-D vector of pseudo dynamic features
    """

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # binary threshold (invert so strokes=1)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    # skeleton
    skel = skeletonize(invert(th/255)).astype(np.uint8)
    stroke_length = skel.sum()

    # stroke thickness via distance transform
    dist = cv2.distanceTransform(th, cv2.DIST_L2, 5)
    mean_thickness = np.mean(dist[dist > 0]) if np.any(dist > 0) else 0

    # stroke segments = number of contours
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    stroke_segments = len(contours)

    # aspect ratio of handwriting region
    ys, xs = np.where(th > 0)
    if len(xs) > 0:
        h = ys.max() - ys.min()
        w = xs.max() - xs.min()
        aspect_ratio = w / (h + 1e-6)
    else:
        aspect_ratio = 1.0

    # curvature proxy
    curvature = sum(len(cv2.convexHull(cnt)) for cnt in contours if len(cnt) > 5)

    return np.array([
        stroke_length / 1000,
        mean_thickness / 10,
        stroke_segments / 10,
        aspect_ratio,
        curvature / 1000
    ], dtype=np.float32)
