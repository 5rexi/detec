"""Legacy helper. Use annotate_helmet.py / annotate_vest.py for explicit task separation."""

import argparse

from ppe.annotation import annotate_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["helmet", "vest"], required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--yolo-weights", default="weights/yolo11s.pt")
    parser.add_argument("--sample-rate", type=int, default=3)
    args = parser.parse_args()

    annotate_video(args.task, args.video, args.yolo_weights, args.sample_rate)
