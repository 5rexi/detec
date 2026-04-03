from ppe.annotation import annotate_video

if __name__ == "__main__":
    annotate_video(
        task_name="helmet",
        video_path="./origin_data/test.mp4",
        yolo_weights="weights/yolo11s.pt",
        save_every_n_frames=10,
    )
