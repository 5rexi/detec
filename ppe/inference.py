import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
from torchvision import transforms
from ultralytics import YOLO

from model.resnet import HeadHelmetResNet
from model.utils import extract_valid_head
from ppe.tasks import TASKS


STATE_OK = 0
STATE_NG = 1
STATE_UNKNOWN = 2


def decide_frame_state(
    probs: torch.Tensor,
    ok_threshold: float,
    violation_threshold: float,
    invalid_threshold: float,
) -> int:
    """
    保守判违规策略：
    - 违规必须达到更高阈值 violation_threshold 才算违规候选；
    - 合规达到 ok_threshold 时可直接判合规；
    - invalid 概率高时判未知；
    - 其他情况一律 UNKNOWN，避免误报违规。
    """
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
    track_id,
    frame_state,
    score_decay: float,
    score_step: float,
    trigger_score: float,
    clear_ok_streak: int,
):
    """
    轨迹级证据累积：
    - NG 帧增加分数；
    - OK 帧衰减分数并累积 ok streak；
    - UNKNOWN 轻微衰减；
    - 达到 trigger_score 才最终告警 NG。
    """
    if frame_state == STATE_NG:
        track_scores[track_id] = track_scores[track_id] * score_decay + score_step
        track_ok_streak[track_id] = 0
    elif frame_state == STATE_OK:
        track_scores[track_id] = max(0.0, track_scores[track_id] * 0.5)
        track_ok_streak[track_id] += 1
    else:
        track_scores[track_id] = track_scores[track_id] * 0.9
        track_ok_streak[track_id] = 0

    if track_ok_streak[track_id] >= clear_ok_streak:
        track_scores[track_id] = 0.0

    if track_scores[track_id] >= trigger_score:
        return STATE_NG
    if frame_state == STATE_OK:
        return STATE_OK
    return STATE_UNKNOWN


def run_video(
    task_name: str,
    video_path: str,
    yolo_weights: str,
    cls_weights: str,
    output_path: str,
    conf: float = 0.5,
    ok_threshold: float = 0.6,
    violation_threshold: float = 0.95,
    invalid_threshold: float = 0.5,
    score_decay: float = 0.8,
    score_step: float = 1.0,
    trigger_score: float = 1.0,
    clear_ok_streak: int = 3,
) -> None:
    task = TASKS[task_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = YOLO(yolo_weights)
    classifier = HeadHelmetResNet(num_classes=3)
    classifier.load_state_dict(torch.load(cls_weights, map_location=device))
    classifier.to(device).eval()

    transform = transforms.Compose([transforms.ToTensor()])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError(f"Cannot read video: {video_path}")

    frame_h, frame_w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w + 220, frame_h),
    )

    track_scores = defaultdict(float)
    track_ok_streak = defaultdict(int)
    violation_gallery = {}

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

                crop = task.crop_fn(frame, bbox)
                if crop is None:
                    frame_state = STATE_UNKNOWN
                else:
                    tensor = transform(crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs = torch.softmax(classifier(tensor), dim=1).squeeze(0)
                    frame_state = decide_frame_state(
                        probs=probs,
                        ok_threshold=ok_threshold,
                        violation_threshold=violation_threshold,
                        invalid_threshold=invalid_threshold,
                    )

                final_state = update_track_state(
                    track_scores=track_scores,
                    track_ok_streak=track_ok_streak,
                    track_id=track_id,
                    frame_state=frame_state,
                    score_decay=score_decay,
                    score_step=score_step,
                    trigger_score=trigger_score,
                    clear_ok_streak=clear_ok_streak,
                )

                if final_state == STATE_NG and track_id not in violation_gallery and crop is not None:
                    violation_gallery[track_id] = crop

                x1, y1, x2, y2 = map(int, bbox)
                if final_state == STATE_OK:
                    color = (0, 255, 0)
                    text = task.title_ok
                elif final_state == STATE_NG:
                    color = (0, 0, 255)
                    text = task.title_ng
                else:
                    color = (128, 128, 128)
                    text = "Detecting"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{text} ID:{track_id} S={track_scores[track_id]:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        wall = np.zeros((frame_h, 220, 3), dtype=np.uint8)
        y = 10
        for tid, img in violation_gallery.items():
            if img is None or img.size == 0:
                continue
            h = int(220 * img.shape[0] / max(1, img.shape[1]))
            if y + h > frame_h:
                break
            resized = cv2.resize(img, (220, h))
            wall[y : y + h, :] = resized
            cv2.putText(wall, f"ID {tid}", (5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y += h + 15

        vis = np.hstack([frame, wall])
        writer.write(vis)
        cv2.imshow(f"{task.name.title()} Detection", vis)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run helmet or vest detection on video.")
    parser.add_argument("--task", choices=TASKS.keys(), required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--yolo-weights", default="weights/yolo11n.pt")
    parser.add_argument("--cls-weights", default=None)
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--ok-threshold", type=float, default=0.6)
    parser.add_argument("--violation-threshold", type=float, default=0.95)
    parser.add_argument("--invalid-threshold", type=float, default=0.5)
    parser.add_argument("--score-decay", type=float, default=0.8)
    parser.add_argument("--score-step", type=float, default=1.0)
    parser.add_argument("--trigger-score", type=float, default=1.0)
    parser.add_argument("--clear-ok-streak", type=int, default=3)
    args = parser.parse_args()

    task = TASKS[args.task]
    run_video(
        task_name=args.task,
        video_path=args.video,
        yolo_weights=args.yolo_weights,
        cls_weights=args.cls_weights or task.weights_path,
        output_path=args.output,
        conf=args.conf,
        ok_threshold=args.ok_threshold,
        violation_threshold=args.violation_threshold,
        invalid_threshold=args.invalid_threshold,
        score_decay=args.score_decay,
        score_step=args.score_step,
        trigger_score=args.trigger_score,
        clear_ok_streak=args.clear_ok_streak,
    )
