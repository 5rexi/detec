"""Backward-compatible entrypoint. Prefer detect_vest.py."""

from ppe.inference import run_video

if __name__ == "__main__":
    run_video(
        task_name="vest",
        video_path="./origin_data/wF0Z6eJHVtOkFzh7kGoo000000_1766383200.mp4",
        yolo_weights="weights/yolo11n.pt",
        cls_weights="weights/resnet_vest.pth",
        output_path="output_vest.mp4",
        violation_threshold=0.95,
        trigger_score=1.0,
        conf=0.5,
    )
