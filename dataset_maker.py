import cv2
from model.utils import *
from ultralytics import YOLO
import os
import time
import numpy as np

MODE = 'head'  # ['head', 'turso']
model = YOLO("weights/yolo11s.pt")
path = './origin_data/wF0Z6eJHVtOkFzh7kGoo000000_1766383200.mp4'
cap = cv2.VideoCapture(path)

if MODE == 'head':
    KEY_MAP = {
        ord('1'): 'with_helmet',
        ord('2'): 'without_helmet',
        ord('3'): 'invalid'
    }
else:
    KEY_MAP = {
        ord('1'): 'with_clothing',
        ord('2'): 'without_clothing',
        ord('3'): 'invalid'
    }


def manual_label_and_save(head_img, save_root, prefix="img"):
    """
    head_img: np.ndarray
    save_root: dataset root dir
    """
    cv2.imshow("Head (1=helmet, 2=no helmet, 3=invalid, q=quit)", head_img)

    while True:
        key = cv2.waitKey(0)

        if key == ord('q'):
            print("Quit labeling.")
            return False  # signal to stop

        if key in KEY_MAP:
            label = KEY_MAP[key]
            save_dir = os.path.join(save_root, label)
            os.makedirs(save_dir, exist_ok=True)

            filename = f"{prefix}_{int(time.time() * 1000)}.jpg"
            save_path = os.path.join(save_dir, filename)

            cv2.imwrite(save_path, head_img)
            print(f"Saved to {label}/{filename}")
            return True


time_num = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        time_num += 1
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0], conf=0.6)
        if time_num % 3 != 0:
            continue
        for box in results[0].boxes.xyxy:
            head = crop_head_from_bbox(frame, box) if MODE == 'head' else crop_torso_from_bbox(frame, box)
            if head is None:
                continue
            if extract_valid_head(frame, box) is None:
                continue
            if MODE == 'head':
                keep_going = manual_label_and_save(
                    head_img=head,
                    save_root='dataset',
                    prefix="head"
                )
            else:
                keep_going = manual_label_and_save(
                    head_img=head,
                    save_root='dataset_cloth',
                    prefix="head"
                )

            if not keep_going:
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()