import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import transforms
from model.resnet import HeadHelmetResNet
from model.utils import *
from collections import defaultdict, deque

# =============================
# Config
# =============================
DETEC_TIME = 10  # 你原本叫 10，但注释写3帧，这里按你原逻辑：队列满 DETEC_TIME 才判断 all(...)
CONF = 0.5

# ===== helmet classes =====
HELMET = 0
NO_HELMET = 1
H_INVALID = 2

# ===== vest classes =====
VEST = 0
NO_VEST = 1
V_INVALID = 2

# =============================
# Paths
# =============================
path = './origin_data/wF0Z6eJHVtOkFzh7kGoo000000_1766383200.mp4'  # TODO: 换成你要检测的同一个视频

# =============================
# Models
# =============================
yolo = YOLO("./weights/yolo11n.pt")

helmet_model = HeadHelmetResNet(num_classes=3)
helmet_model.load_state_dict(torch.load("weights/resnet_model.pth", map_location="cpu"))
helmet_model.cuda().eval()

vest_model = HeadHelmetResNet(num_classes=3)
vest_model.load_state_dict(torch.load("weights/resnet_model_cloth.pth", map_location="cpu"))
vest_model.cuda().eval()

cls_transform = transforms.Compose([transforms.ToTensor()])

# =============================
# Track states (separated)
# =============================
helmet_states = defaultdict(lambda: deque(maxlen=DETEC_TIME))
vest_states = defaultdict(lambda: deque(maxlen=DETEC_TIME))

track_last_head = {}  # 存最近一次 head 小图，用于画廊（你原来是这样做的）

no_helmet_gallery = {}  # {track_id: head_img}
no_vest_gallery = {}    # {track_id: head_img}  # 反光衣违规也用 head 图展示（沿用你的逻辑）

# =============================
# Video IO
# =============================
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1e-6:
    fps = 25

# 输出宽高：原画面 + 右侧两列违规墙（每列 220）
ret, tmp = cap.read()
if not ret:
    raise RuntimeError("Cannot read video.")
h0, w0 = tmp.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

WALL_COL_W = 220
RIGHT_W = WALL_COL_W * 2  # 两列
final_w = w0 + RIGHT_W
final_h = h0

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter("output_both.mp4", fourcc, fps, (final_w, final_h))

# =============================
# Helper functions
# =============================
def update_track_state(states_dict, track_id: int, cls: int, ok_cls: int, bad_cls: int, invalid_cls: int):
    """
    states_dict: helmet_states or vest_states
    ok_cls/bad_cls/invalid_cls: for that task
    """
    states_dict[track_id].append(cls)

    if len(states_dict[track_id]) < DETEC_TIME:
        return invalid_cls

    # 你原逻辑：队列内全是某类才判定
    if all(c == bad_cls for c in states_dict[track_id]):
        return bad_cls
    if all(c == ok_cls for c in states_dict[track_id]):
        return ok_cls

    return invalid_cls


def render_violation_wall(wall_dict, height, width=WALL_COL_W, title=None):
    """
    wall_dict: {tid: img}
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    y = 10

    if title is not None:
        cv2.putText(
            canvas, title,
            (5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (255, 255, 255), 2
        )
        y += 35

    for tid, img in wall_dict.items():
        if img is None or img.size == 0:
            continue

        h = int(width * img.shape[0] / max(1, img.shape[1]))
        if h <= 0:
            continue
        resized = cv2.resize(img, (width, h))

        if y + h > height:
            break

        canvas[y:y + h, :] = resized
        cv2.putText(
            canvas, f"ID {tid}",
            (5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )
        y += h + 15

    return canvas


def infer_resnet(model, bgr_img):
    """
    bgr_img: numpy HxWx3 (BGR)
    return pred int in [0,1,2] or invalid if None
    """
    if bgr_img is None:
        return None
    img_t = cls_transform(bgr_img).unsqueeze(0).cuda()
    with torch.no_grad():
        pred = model(img_t).argmax(1).item()
    return pred


# =============================
# Main loop
# =============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo.track(frame, persist=True, classes=[0], conf=CONF)

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for bbox, track_id in zip(boxes, ids):
            track_id_int = int(track_id)

            # ===== validity check (你原来用 extract_valid_head(frame, bbox) is None 就跳过)
            if extract_valid_head(frame, bbox) is None:
                continue

            # -------------------------
            # 1) Helmet classification
            # -------------------------
            head_img = crop_head_from_bbox(frame, bbox)
            if head_img is None:
                helmet_cls = H_INVALID
            else:
                track_last_head[track_id_int] = head_img
                pred = infer_resnet(helmet_model, head_img)
                if pred == 0:
                    helmet_cls = HELMET
                elif pred == 1:
                    helmet_cls = NO_HELMET
                else:
                    helmet_cls = H_INVALID

            helmet_state = update_track_state(
                helmet_states,
                track_id_int,
                helmet_cls,
                ok_cls=HELMET,
                bad_cls=NO_HELMET,
                invalid_cls=H_INVALID
            )

            # 记录无头盔画廊（用最近 head）
            if helmet_state == NO_HELMET and track_id_int not in no_helmet_gallery:
                if track_id_int in track_last_head:
                    no_helmet_gallery[track_id_int] = track_last_head[track_id_int]

            # -------------------------
            # 2) Vest classification
            # -------------------------
            torso_img = crop_torso_from_bbox(frame, bbox)
            if torso_img is None:
                vest_cls = V_INVALID
            else:
                # 你原反光衣逻辑：画廊里存 head（hhead）
                if track_id_int not in track_last_head and head_img is not None:
                    track_last_head[track_id_int] = head_img

                pred = infer_resnet(vest_model, torso_img)
                if pred == 0:
                    vest_cls = VEST
                elif pred == 1:
                    vest_cls = NO_VEST
                else:
                    vest_cls = V_INVALID

            vest_state = update_track_state(
                vest_states,
                track_id_int,
                vest_cls,
                ok_cls=VEST,
                bad_cls=NO_VEST,
                invalid_cls=V_INVALID
            )

            # 记录无反光衣画廊（也存 head，和你原来一致）
            if vest_state == NO_VEST and track_id_int not in no_vest_gallery:
                if track_id_int in track_last_head:
                    no_vest_gallery[track_id_int] = track_last_head[track_id_int]

            # =========================
            # Visualization
            # =========================
            x1, y1, x2, y2 = map(int, bbox)

            # bbox 颜色：只要任一违规 → 红；两者都OK → 绿；否则灰
            any_bad = (helmet_state == NO_HELMET) or (vest_state == NO_VEST)
            all_ok = (helmet_state == HELMET) and (vest_state == VEST)

            if all_ok:
                color = (0, 255, 0)
            elif any_bad:
                color = (0, 0, 255)
            else:
                color = (128, 128, 128)

            # text lines
            def helmet_text(s):
                if s == HELMET:
                    return "Helmet"
                if s == NO_HELMET:
                    return "No Helmet"
                return "Helmet:Detecting"

            def vest_text(s):
                if s == VEST:
                    return "Vest"
                if s == NO_VEST:
                    return "No Vest"
                return "Vest:Detecting"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 两行文字更清晰
            cv2.putText(
                frame, f"ID:{track_id_int}  {helmet_text(helmet_state)}",
                (x1, max(0, y1 - 28)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
            cv2.putText(
                frame, f"{vest_text(vest_state)}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

    # =========================
    # Right-side dual wall
    # =========================
    wall_helmet = render_violation_wall(no_helmet_gallery, height=frame.shape[0], title="No Helmet")
    wall_vest = render_violation_wall(no_vest_gallery, height=frame.shape[0], title="No Vest")

    wall = np.hstack([wall_helmet, wall_vest])
    vis = np.hstack([frame, wall])

    video_writer.write(vis)
    cv2.imshow("Helmet + Reflective Vest Detection Demo", vis)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
