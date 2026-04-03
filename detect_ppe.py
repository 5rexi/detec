import cv2
import numpy as np
import torch
from collections import defaultdict
from torchvision import transforms
from ultralytics import YOLO

from model.resnet import HeadHelmetResNet
from model.utils import crop_head_from_bbox, crop_torso_from_bbox, extract_valid_head

STATE_OK = 0
STATE_NG = 1
STATE_UNKNOWN = 2


def decide_frame_state(probs: torch.Tensor, ok_threshold: float, violation_threshold: float, invalid_threshold: float) -> int:
    p_ok = probs[0].item()
    p_ng = probs[1].item()
    p_invalid = probs[2].item()

    if p_invalid >= invalid_threshold:
        return STATE_UNKNOWN
    if p_ok >= ok_threshold:
        return STATE_OK
    if p_ng >= violation_threshold:
        return STATE_NG
    return STATE_UNKNOWN


def update_track_state(
    track_scores,
    track_ok_streak,
    track_ng_streak,
    track_id,
    frame_state,
    score_decay: float,
    score_step: float,
    trigger_score: float,
    clear_ok_streak: int,
    min_violation_streak: int,
):
    if frame_state == STATE_NG:
        track_scores[track_id] = track_scores[track_id] * score_decay + score_step
        track_ok_streak[track_id] = 0
        track_ng_streak[track_id] += 1
    elif frame_state == STATE_OK:
        track_scores[track_id] = max(0.0, track_scores[track_id] * 0.5)
        track_ok_streak[track_id] += 1
        track_ng_streak[track_id] = 0
    else:
        track_scores[track_id] = track_scores[track_id] * 0.9
        track_ok_streak[track_id] = 0
        track_ng_streak[track_id] = 0

    if track_ok_streak[track_id] >= clear_ok_streak:
        track_scores[track_id] = 0.0

    if track_scores[track_id] >= trigger_score and track_ng_streak[track_id] >= min_violation_streak:
        return STATE_NG
    if frame_state == STATE_OK:
        return STATE_OK
    return STATE_UNKNOWN


def draw_column(canvas, x0, width, title, color, items, frame_h):
    cv2.rectangle(canvas, (x0, 0), (x0 + width, 34), (40, 40, 40), -1)
    cv2.putText(canvas, title, (x0 + 6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    y = 40
    for tid, img in items:
        if img is None or img.size == 0:
            continue
        h = int(width * img.shape[0] / max(1, img.shape[1]))
        if y + h > frame_h:
            break

        resized = cv2.resize(img, (width, h))
        canvas[y : y + h, x0 : x0 + width] = resized
        cv2.putText(canvas, f"ID {tid}", (x0 + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += h + 12


def main():
    video_path = "./origin_data/test.mp4"
    yolo_weights = "weights/yolo11n.pt"
    helmet_weights = "weights/resnet_helmet.pth"
    vest_weights = "weights/resnet_vest.pth"
    output_path = "output_ppe.mp4"

    # 与 detect_helmet / detect_vest 保持一致
    conf = 0.5
    ok_threshold = 0.6
    violation_threshold = 0.95
    invalid_threshold = 0.5

    score_decay = 0.8
    score_step = 1.0
    trigger_score = 1.0
    clear_ok_streak = 3
    min_violation_streak_helmet = 7
    min_violation_streak_vest = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = YOLO(yolo_weights)

    helmet_model = HeadHelmetResNet(num_classes=3)
    helmet_model.load_state_dict(torch.load(helmet_weights, map_location=device))
    helmet_model.to(device).eval()

    vest_model = HeadHelmetResNet(num_classes=3)
    vest_model.load_state_dict(torch.load(vest_weights, map_location=device))
    vest_model.to(device).eval()

    transform = transforms.Compose([transforms.ToTensor()])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read video: {video_path}")

    frame_h, frame_w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    col_w = 220
    panel_w = col_w * 3

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w + panel_w, frame_h),
    )

    helmet_scores = defaultdict(float)
    helmet_ok_streak = defaultdict(int)
    helmet_ng_streak = defaultdict(int)

    vest_scores = defaultdict(float)
    vest_ok_streak = defaultdict(int)
    vest_ng_streak = defaultdict(int)

    helmet_final = defaultdict(lambda: STATE_UNKNOWN)
    vest_final = defaultdict(lambda: STATE_UNKNOWN)

    # 固定快照画廊：一旦命中违规即固定，不随之后帧变化
    helmet_gallery = {}
    vest_gallery = {}
    both_gallery = {}

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        results = detector.track(frame, persist=True, classes=[0], conf=conf)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for bbox, track_id in zip(boxes, ids):
                track_id = int(track_id)
                if extract_valid_head(frame, bbox) is None:
                    continue

                head_crop = crop_head_from_bbox(frame, bbox)
                torso_crop = crop_torso_from_bbox(frame, bbox)

                if head_crop is None:
                    helmet_frame_state = STATE_UNKNOWN
                else:
                    tensor = transform(head_crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs = torch.softmax(helmet_model(tensor), dim=1).squeeze(0)
                    helmet_frame_state = decide_frame_state(probs, ok_threshold, violation_threshold, invalid_threshold)

                if torso_crop is None:
                    vest_frame_state = STATE_UNKNOWN
                else:
                    tensor = transform(torso_crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs = torch.softmax(vest_model(tensor), dim=1).squeeze(0)
                    vest_frame_state = decide_frame_state(probs, ok_threshold, violation_threshold, invalid_threshold)

                helmet_final[track_id] = update_track_state(
                    track_scores=helmet_scores,
                    track_ok_streak=helmet_ok_streak,
                    track_ng_streak=helmet_ng_streak,
                    track_id=track_id,
                    frame_state=helmet_frame_state,
                    score_decay=score_decay,
                    score_step=score_step,
                    trigger_score=trigger_score,
                    clear_ok_streak=clear_ok_streak,
                    min_violation_streak=min_violation_streak_helmet,
                )

                vest_final[track_id] = update_track_state(
                    track_scores=vest_scores,
                    track_ok_streak=vest_ok_streak,
                    track_ng_streak=vest_ng_streak,
                    track_id=track_id,
                    frame_state=vest_frame_state,
                    score_decay=score_decay,
                    score_step=score_step,
                    trigger_score=trigger_score,
                    clear_ok_streak=clear_ok_streak,
                    min_violation_streak=min_violation_streak_vest,
                )

                h_state = helmet_final[track_id]
                v_state = vest_final[track_id]

                if h_state == STATE_NG and track_id not in helmet_gallery and track_id not in both_gallery and head_crop is not None:
                    helmet_gallery[track_id] = head_crop

                if v_state == STATE_NG and track_id not in vest_gallery and track_id not in both_gallery and torso_crop is not None:
                    vest_gallery[track_id] = torso_crop

                # 发现双违规后：把头盔图迁移到双违规列，并删除反光衣列图片
                if h_state == STATE_NG and v_state == STATE_NG:
                    if track_id not in both_gallery:
                        both_img = helmet_gallery.get(track_id)
                        if both_img is None and head_crop is not None:
                            both_img = head_crop
                        if both_img is None:
                            both_img = vest_gallery.get(track_id)
                        if both_img is None:
                            both_img = torso_crop
                        if both_img is not None:
                            both_gallery[track_id] = both_img

                    helmet_gallery.pop(track_id, None)
                    vest_gallery.pop(track_id, None)

                x1, y1, x2, y2 = map(int, bbox)
                if h_state == STATE_NG and v_state == STATE_NG:
                    color = (0, 0, 255)
                    text = "No Helmet + No Vest"
                elif h_state == STATE_NG:
                    color = (0, 102, 255)
                    text = "No Helmet"
                elif v_state == STATE_NG:
                    color = (255, 102, 0)
                    text = "No Vest"
                else:
                    color = (0, 255, 0) if (h_state == STATE_OK and v_state == STATE_OK) else (128, 128, 128)
                    text = "PPE OK" if color == (0, 255, 0) else "Detecting"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{text} ID:{track_id} Hs={helmet_scores[track_id]:.2f} Vs={vest_scores[track_id]:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

        panel = np.zeros((frame_h, panel_w, 3), dtype=np.uint8)
        draw_column(panel, 0, col_w, "No Helmet", (0, 102, 255), list(helmet_gallery.items()), frame_h)
        draw_column(panel, col_w, col_w, "No Vest", (255, 102, 0), list(vest_gallery.items()), frame_h)
        draw_column(panel, col_w * 2, col_w, "No Helmet + No Vest", (0, 0, 255), list(both_gallery.items()), frame_h)

        vis = np.hstack([frame, panel])
        writer.write(vis)

        cv2.imshow("Helmet + Vest Detection", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
