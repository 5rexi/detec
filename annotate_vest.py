from ppe.annotation import annotate_video

if __name__ == "__main__":
    annotate_video(
        task_name="vest",
        video_path="./origin_data/video(11).mp4",
        yolo_weights="weights/yolo11s.pt",
        save_every_n_frames=20,
    )
