"""Backward-compatible entrypoint. Prefer detect_helmet.py."""

from ppe.inference import run_video

if __name__ == "__main__":
    run_video(
        task_name="helmet",
        video_path="./origin_data/HZIdNgLggCAUJZqRhrW2000000_1766367544.mp4",
        yolo_weights="weights/yolo11n.pt",
        cls_weights="weights/resnet_helmet.pth",
        output_path="output_helmet.mp4",
        violation_threshold=0.95,
        trigger_score=1.0,
        min_violation_streak=3,
        conf=0.5,
    )
