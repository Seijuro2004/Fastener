"""Fastener detector: HSV colour segmentation + FAST keypoints + ORB descriptors (OpenCV only)."""
import glob, os, pickle
import cv2
import numpy as np

LABELS = {
    0: "Flange Nut", 1: "Hex Nut", 2: "Flat Washer", 3: "Spring Washer",
    4: "Hex Bolt (Short)", 5: "Hex Bolt (Long)", 6: "Countersunk Screw",
    7: "Phillips Washer-Head Screw", 8: "Machine Screw", 9: "Coupling Nut",
}
WORK = 640  # working width in px

_fast = cv2.FastFeatureDetector_create(threshold=15, nonmaxSuppression=True)
_orb = cv2.ORB_create(nfeatures=500)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def _resize(img):
    h, w = img.shape[:2]
    s = WORK / w
    return cv2.resize(img, (WORK, int(h * s)), interpolation=cv2.INTER_AREA)


def segment(img):
    """Colour detection: the backdrop is teal, so mask it out and keep the largest blob."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    # background hue = median hue of image border strip
    border = np.concatenate([hsv[:15].reshape(-1, 3), hsv[-15:].reshape(-1, 3),
                             hsv[:, :15].reshape(-1, 3), hsv[:, -15:].reshape(-1, 3)])
    h0 = int(np.median(border[:, 0]))
    dh = np.abs(hsv[..., 0].astype(int) - h0)
    bg = (dh < 12) & (hsv[..., 1] > 80)
    mask = (~bg).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.0004 * img.shape[0] * img.shape[1]:
        return None, None
    return c, mask


def describe(img, cnt):
    """Crop to object, FAST keypoints -> ORB descriptors, plus colour/shape features."""
    x, y, w, h = cv2.boundingRect(cnt)
    pad = int(0.08 * max(w, h))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    crop = img[y0:y + h + pad, x0:x + w + pad]
    scale = 256 / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale)
    gray = cv2.createCLAHE(2.0, (4, 4)).apply(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    kps = _fast.detect(gray, None)
    kps, des = _orb.compute(gray, kps)
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.drawContours(m, [cnt], -1, 255, -1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = [float(v) for v in cv2.mean(hsv, m)[:3]]
    rect = cv2.minAreaRect(cnt)
    rw, rh = sorted(rect[1])
    area = cv2.contourArea(cnt)
    hull = cv2.contourArea(cv2.convexHull(cnt))
    shape = dict(aspect=rw / max(rh, 1), solidity=area / max(hull, 1),
                 hue=hue, sat=sat, val=val, rel_size=max(w, h) / img.shape[1])
    return des, kps, crop, shape, (x, y, w, h)


def _good_matches(q, r):
    if q is None or r is None or len(q) < 2 or len(r) < 2:
        return 0
    n = 0
    for p in _bf.knnMatch(q, r, k=2):
        if len(p) == 2 and p[0].distance < 0.75 * p[1].distance:
            n += 1
    return n


class FastenerDetector:
    def __init__(self, data_dir, cache="refs_cache.pkl"):
        self.refs = []  # (class, descriptors, shape, tag)
        if os.path.exists(cache):  # prebuilt reference features, no dataset needed
            with open(cache, "rb") as fh:
                self.refs = pickle.load(fh)
            return
        for c in LABELS:
            for f in sorted(glob.glob(os.path.join(data_dir, "fasteners", str(c), "*.jpg"))):
                self.add_reference(c, cv2.imread(f), os.path.basename(f))
        with open(cache, "wb") as fh:
            pickle.dump(self.refs, fh)

    def add_reference(self, c, img, tag=None):
        img = _resize(img)
        cnt, _ = segment(img)
        if cnt is None:
            return
        des, _, _, shape, _ = describe(img, cnt)
        self.refs.append((c, des, shape, tag))

    @staticmethod
    def _shape_dist(a, b):
        return (abs(a["aspect"] - b["aspect"]) / 0.3 + abs(a["rel_size"] - b["rel_size"]) / 0.15
                + abs(a["hue"] - b["hue"]) / 25 + abs(a["sat"] - b["sat"]) / 80
                + abs(a["solidity"] - b["solidity"]) / 0.15)

    def classify(self, img, exclude=None):
        img = _resize(img)
        cnt, _ = segment(img)
        if cnt is None:
            return None
        des, kps, crop, shape, box = describe(img, cnt)
        scores = {c: [] for c in LABELS}
        for c, rd, rs, tag in self.refs:
            if exclude and tag == exclude:
                continue
            orb = _good_matches(des, rd) / max(len(des), 1)
            geo = max(0.0, 1 - self._shape_dist(shape, rs) / 5)
            scores[c].append(0.4 * orb * 4 + 0.6 * geo)
        # mean of the top-3 reference scores per class
        final = {c: float(np.mean(sorted(v, reverse=True)[:3])) for c, v in scores.items() if v}
        best = max(final, key=final.get)
        tot = sum(max(v, 1e-6) ** 3 for v in final.values())
        conf = max(final[best], 1e-6) ** 3 / tot
        return dict(cls=best, label=LABELS[best], confidence=conf, box=box,
                    kps=kps, crop=crop, contour=cnt, scale=_resize(img).shape[1] / img.shape[1], img=img)
