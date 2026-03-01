import cv2
import numpy as np

def crop_head_from_bbox(frame, bbox, min_size=40):
    """
    frame: np.ndarray, shape (H, W, 3)
    bbox:  (x1, y1, x2, y2) in pixel coords
    """
    H, W, _ = frame.shape
    x1, y1, x2, y2 = map(int, bbox)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    # head region (industrial heuristic)
    hx1 = int(x1 + 0.15 * bw)
    hx2 = int(x1 + 0.85 * bw)
    hy1 = int(y1)
    hy2 = int(y1 + 0.30 * bh)

    # clamp to image boundary
    hx1 = max(0, min(hx1, W - 1))
    hx2 = max(0, min(hx2, W))
    hy1 = max(0, min(hy1, H - 1))
    hy2 = max(0, min(hy2, H))

    if hx2 - hx1 < min_size or hy2 - hy1 < min_size:
        return None

    head = frame[hy1:hy2, hx1:hx2]
    return head

def crop_torso_from_bbox(frame, bbox, min_size=60):
    """
    Crop upper-body / torso region for reflective vest detection.

    frame: np.ndarray, shape (H, W, 3)
    bbox:  (x1, y1, x2, y2) in pixel coords
    return: cropped torso image or None
    """
    H, W, _ = frame.shape
    x1, y1, x2, y2 = map(int, bbox)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    # ===== Industrial heuristic: upper torso =====
    tx1 = int(x1 + 0.10 * bw)
    tx2 = int(x1 + 0.90 * bw)

    ty1 = int(y1 + 0.20 * bh)   # below head
    ty2 = int(y1 + 0.60 * bh)   # above waist

    # ===== Clamp to image boundary =====
    tx1 = max(0, min(tx1, W - 1))
    tx2 = max(0, min(tx2, W))
    ty1 = max(0, min(ty1, H - 1))
    ty2 = max(0, min(ty2, H))

    if tx2 - tx1 < min_size or ty2 - ty1 < min_size:
        return None

    torso = frame[ty1:ty2, tx1:tx2]
    return torso


def bbox_touch_border(bbox, img_w, img_h, margin=5):
    x1, y1, x2, y2 = bbox
    return (
        x1 < margin or
        y1 < margin or
        x2 > img_w - margin or
        y2 > img_h - margin
    )

def is_blank(img, thresh=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.var(gray) < thresh

def extract_valid_head(frame, bbox):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # 1. bbox 撞边界
    if bbox_touch_border((x1,y1,x2,y2), W, H):
        return None

    h = y2 - y1
    w = x2 - x1

    # 2. head crop
    head_h = int(0.25 * h)
    if head_h < 20 or w < 20:
        return None

    head = frame[y1:y1+head_h, x1:x2]

    # 3. 比例约束
    if head.shape[0] / h < 0.18 or head.shape[0] / h > 0.35:
        return None

    # 4. 空白检测
    if is_blank(head):
        return None

    return head