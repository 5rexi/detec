import argparse
from collections import defaultdict, deque

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


def update_track_state(track_states, track_id, cls, time_window):
    track_states[track_id].append(cls)
    if len(track_states[track_id]) < time_window:
        return STATE_UNKNOWN

    if all(c == STATE_NG for c in track_states[track_id]):
        return STATE_NG
    if all(c == STATE_OK for c in track_states[track_id]):
        return STATE_OK
    return STATE_UNKNOWN


def run_video(
    task_name: str,
    video_path: str,
    yolo_weights: str,
    cls_weights: str,
    output_path: str,
    time_window: int = 10,
    conf: float = 0.5,
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

    track_states = defaultdict(lambda: deque(maxlen=time_window))
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
                    cls = STATE_UNKNOWN
                else:
                    tensor = transform(crop).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred = classifier(tensor).argmax(1).item()

                    if pred == 0:
                        cls = STATE_OK
                    elif pred == 1:
                        cls = STATE_NG
                    else:
                        cls = STATE_UNKNOWN

                final_state = update_track_state(track_states, track_id, cls, time_window)

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
                    f"{text} ID:{track_id}",
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
    parser.add_argument("--time-window", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    task = TASKS[args.task]
    run_video(
        task_name=args.task,
        video_path=args.video,
        yolo_weights=args.yolo_weights,
        cls_weights=args.cls_weights or task.weights_path,
        output_path=args.output,
        time_window=args.time_window,
        conf=args.conf,
    )
