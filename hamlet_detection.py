import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from model.resnet import HeadHelmetResNet
from model.utils import *
from collections import defaultdict, deque

DETEC_TIME = 10
track_states = defaultdict(lambda: deque(maxlen=DETEC_TIME))
track_last_head = {}
violation_gallery = {}

HELMET = 0
NO_HELMET = 1
INVALID = 2

path = './origin_data/HZIdNgLggCAUJZqRhrW2000000_1766367544.mp4'
yolo = YOLO("./weights/yolo11n.pt")
helmet_model = HeadHelmetResNet(num_classes=3)
helmet_model.load_state_dict(torch.load("weights/resnet_model.pth"))
helmet_model.cuda().eval()

final_w = 2780
final_h = 1440
fps = 25

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    'output.mp4',
    fourcc,
    fps,
    (final_w, final_h)
)


cls_transform = transforms.Compose([
    transforms.ToTensor()
])
def update_track_state(track_id, cls):
    track_states[track_id].append(cls)

    if len(track_states[track_id]) < DETEC_TIME:
        return INVALID

    # 连续三帧不戴头盔
    if all(c == NO_HELMET for c in track_states[track_id]):
        return NO_HELMET

    # 连续三帧戴头盔
    if all(c == HELMET for c in track_states[track_id]):
        return HELMET

    return INVALID
def render_violation_wall(wall, height, width=220):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    y = 10
    for tid, img in wall.items():
        h = int(width * img.shape[0] / img.shape[1])
        resized = cv2.resize(img, (width, h))

        if y + h > height:
            break

        canvas[y:y+h, :] = resized
        cv2.putText(
            canvas, f"ID {tid}",
            (5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0,0,255), 2
        )
        y += h + 15

    return canvas


cap = cv2.VideoCapture(path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo.track(frame, persist=True, classes=[0], conf=0.5)

    no_helmet_gallery = []  # 右侧窗口

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for bbox, track_id in zip(boxes, ids):
            head = crop_head_from_bbox(frame, bbox)
            if extract_valid_head(frame, bbox) is None:
                continue

            if head is None:
                cls = INVALID
            else:
                track_last_head[track_id] = head
                img = cls_transform(head).unsqueeze(0).cuda()
                with torch.no_grad():
                    pred = helmet_model(img).argmax(1).item()
                if pred == 0:
                    cls = HELMET
                elif pred == 1:
                    cls = NO_HELMET
                else:
                    cls = INVALID
                # cls = HELMET if pred == 0 else NO_HELMET

            final_state = update_track_state(int(track_id), cls)

            # 记录最近一次 head
            if final_state == NO_HELMET and head is not None:
                if track_id not in violation_gallery:
                    if track_id in track_last_head:
                        violation_gallery[track_id] = track_last_head[track_id]

            # ====== 可视化 ======
            x1, y1, x2, y2 = map(int, bbox)

            if final_state == HELMET:
                color = (0, 255, 0)
                text = "Helmet"
            elif final_state == NO_HELMET:
                color = (0, 0, 255)
                text = "No Helmet"
            else:
                color = (128, 128, 128)
                text = "Detecting"

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(
                frame, f"{text} ID:{track_id}",
                (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

    # # ====== 拼接右侧画廊 ======
    # gallery = np.zeros((frame.shape[0], 200, 3), dtype=np.uint8)
    #
    # for i, img in enumerate(no_helmet_gallery[:5]):
    #     h = int(200 * img.shape[0] / img.shape[1])
    #     resized = cv2.resize(img, (200, h))
    #     y = i * (h + 10)
    #     gallery[y:y+h, :] = resized

    # vis = np.hstack([frame, gallery])

    wall_img = render_violation_wall(
        violation_gallery,
        height=frame.shape[0]
    )

    vis = np.hstack([frame, wall_img])
    video_writer.write(vis)

    cv2.imshow("Helmet Detection Demo", vis)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


