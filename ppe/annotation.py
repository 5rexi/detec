import argparse
import os
import time

import cv2
from ultralytics import YOLO

from model.utils import extract_valid_head
from ppe.tasks import TASKS


def annotate_video(task_name: str, video_path: str, yolo_weights: str, save_every_n_frames: int = 3) -> None:
    task = TASKS[task_name]
    detector = YOLO(yolo_weights)
    cap = cv2.VideoCapture(video_path)

    key_map = {
        ord("1"): task.class_names[0],
        ord("2"): task.class_names[1],
        ord("3"): task.class_names[2],
    }

    frame_index = 0
    window_name = f"Annotate {task.name} (1={task.class_names[0]}, 2={task.class_names[1]}, 3={task.class_names[2]}, q=quit)"

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        if frame_index % save_every_n_frames != 0:
            continue

        results = detector.track(frame, persist=True, classes=[0], conf=0.6)
        if not results or results[0].boxes is None:
            continue

        for bbox in results[0].boxes.xyxy:
            if extract_valid_head(frame, bbox) is None:
                continue

            crop = task.crop_fn(frame, bbox)
            if crop is None:
                continue

            cv2.imshow(window_name, crop)
            key = cv2.waitKey(0)

            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

            if key not in key_map:
                continue

            label_name = key_map[key]
            target_dir = os.path.join(task.dataset_root, label_name)
            os.makedirs(target_dir, exist_ok=True)
            file_name = f"{task.name}_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(os.path.join(target_dir, file_name), crop)
            print(f"[saved] {task.name}: {label_name}/{file_name}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Annotate helmet or vest crops from video.")
    parser.add_argument("--task", choices=TASKS.keys(), required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--yolo-weights", default="weights/yolo11s.pt")
    parser.add_argument("--sample-rate", type=int, default=3, help="Label one frame every N frames")
    args = parser.parse_args()

    annotate_video(
        task_name=args.task,
        video_path=args.video,
        yolo_weights=args.yolo_weights,
        save_every_n_frames=args.sample_rate,
    )
