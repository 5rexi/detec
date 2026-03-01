import cv2
import numpy as np
from collections import deque


class FireDetector:
    """
    Robust fire detector for:
    - warm-color background
    - handheld camera shake
    - outdoor scenes

    Core idea:
    Color (HSV) + Global motion compensation + Local dynamic motion
    """

    def __init__(
            self,
            min_fire_area=70,
            diff_threshold=15,
            motion_energy_threshold=0.015,
            window_size=10,  # n
            min_hits=6,  # m
    ):
        self.min_fire_area = min_fire_area
        self.diff_threshold = diff_threshold
        self.motion_energy_threshold = motion_energy_threshold

        self.window_size = window_size
        self.min_hits = min_hits

        self.hsv_lower = np.array([0, 120, 160])
        self.hsv_upper = np.array([30, 255, 255])

        self.prev_frame = None
        self.fire_history = deque(maxlen=window_size)

    # =========================================================
    # Global motion estimation (camera shake compensation)
    # =========================================================
    def _estimate_global_motion(self, prev_gray, curr_gray):
        pts_prev = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
        )

        if pts_prev is None:
            return None

        pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts_prev, None
        )

        pts_prev = pts_prev[status == 1]
        pts_curr = pts_curr[status == 1]

        if len(pts_prev) < 10:
            return None

        M, _ = cv2.estimateAffinePartial2D(
            pts_prev, pts_curr, method=cv2.RANSAC
        )

        return M

    # =========================================================
    # Color-based fire candidate mask
    # =========================================================
    def _color_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        return mask

    # =========================================================
    # Motion compensated difference
    # =========================================================
    def _motion_compensated_diff(self, prev_frame, curr_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        M = self._estimate_global_motion(prev_gray, curr_gray)
        if M is None:
            return None

        h, w = curr_gray.shape
        prev_warped = cv2.warpAffine(prev_gray, M, (w, h))

        diff = cv2.absdiff(curr_gray, prev_warped)
        _, diff_bin = cv2.threshold(
            diff, self.diff_threshold, 255, cv2.THRESH_BINARY
        )

        return diff_bin

    # =========================================================
    # Main update (call once per frame)
    # =========================================================
    def update(self, frame):
        """
        Args:
            frame: BGR image

        Returns:
            confirmed_fire (bool)
            fire_boxes (list of bbox)
            debug_mask (for visualization)
        """
        fire_boxes = []
        debug_mask = None

        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            self.fire_history.append(False)
            return False, fire_boxes, debug_mask

        # Step 1: color candidate
        color_mask = self._color_mask(frame)

        # Step 2: motion compensation
        diff_mask = self._motion_compensated_diff(
            self.prev_frame, frame
        )

        if diff_mask is None:
            self.prev_frame = frame.copy()
            self.fire_history.append(False)
            return False, fire_boxes, debug_mask

        # Step 3: dynamic fire mask
        fire_mask = cv2.bitwise_and(color_mask, diff_mask)
        debug_mask = fire_mask.copy()

        # Step 4: region analysis
        contours, _ = cv2.findContours(
            fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        fire_detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_fire_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi = fire_mask[y:y+h, x:x+w]
            motion_energy = np.sum(roi > 0) / roi.size

            if motion_energy > self.motion_energy_threshold:
                fire_boxes.append((x, y, x + w, y + h))
                fire_detected = True

        # Step 5: temporal confirmation
        # Step 5: temporal voting (m / n logic)
        self.fire_history.append(fire_detected)

        hit_count = sum(self.fire_history)
        confirmed_fire = hit_count >= self.min_hits

        self.prev_frame = frame.copy()
        return confirmed_fire, fire_boxes, debug_mask


cap = cv2.VideoCapture("./origin_data/fire.mp4")
detector = FireDetector()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fire, boxes, mask = detector.update(frame)

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if fire:
        cv2.putText(
            frame,
            "FIRE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

    cv2.imshow("frame", frame)
    if mask is not None:
        cv2.imshow("fire_mask", mask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
