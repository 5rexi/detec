"""Combined demo for helmet + vest. For production, use detect_helmet.py / detect_vest.py separately."""

from collections import defaultdict, deque

import cv2
import torch
from torchvision import transforms
from ultralytics import YOLO

from model.resnet import HeadHelmetResNet
from model.utils import crop_head_from_bbox, crop_torso_from_bbox, extract_valid_head

STATE_OK = 0
STATE_NG = 1
STATE_UNKNOWN = 2


def update_state(buf, track_id, cls, window=10):
    buf[track_id].append(cls)
    if len(buf[track_id]) < window:
        return STATE_UNKNOWN
    if all(x == STATE_NG for x in buf[track_id]):
        return STATE_NG
    if all(x == STATE_OK for x in buf[track_id]):
        return STATE_OK
    return STATE_UNKNOWN


def infer_cls(model, transform, crop, device):
    if crop is None:
        return STATE_UNKNOWN
    tensor = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(1).item()
    if pred == 0:
        return STATE_OK
    if pred == 1:
        return STATE_NG
    return STATE_UNKNOWN


def main():
    video_path = "./origin_data/wF0Z6eJHVtOkFzh7kGoo000000_1766383200.mp4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo = YOLO("weights/yolo11n.pt")
    helmet_model = HeadHelmetResNet(num_classes=3)
    helmet_model.load_state_dict(torch.load("weights/resnet_helmet.pth", map_location=device))
    helmet_model.to(device).eval()

    vest_model = HeadHelmetResNet(num_classes=3)
    vest_model.load_state_dict(torch.load("weights/resnet_vest.pth", map_location=device))
    vest_model.to(device).eval()

    transform = transforms.Compose([transforms.ToTensor()])
    helmet_states = defaultdict(lambda: deque(maxlen=10))
    vest_states = defaultdict(lambda: deque(maxlen=10))

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        results = yolo.track(frame, persist=True, classes=[0], conf=0.5)
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for bbox, track_id in zip(boxes, ids):
                track_id = int(track_id)
                if extract_valid_head(frame, bbox) is None:
                    continue

                helmet_cls = infer_cls(helmet_model, transform, crop_head_from_bbox(frame, bbox), device)
                vest_cls = infer_cls(vest_model, transform, crop_torso_from_bbox(frame, bbox), device)
                helmet_state = update_state(helmet_states, track_id, helmet_cls)
                vest_state = update_state(vest_states, track_id, vest_cls)

                x1, y1, x2, y2 = map(int, bbox)
                bad = helmet_state == STATE_NG or vest_state == STATE_NG
                color = (0, 0, 255) if bad else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"ID:{track_id} helmet={helmet_state} vest={vest_state}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        cv2.imshow("PPE Demo", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
