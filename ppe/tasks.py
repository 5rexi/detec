from dataclasses import dataclass
from typing import Callable, Dict, List

from model.utils import crop_head_from_bbox, crop_torso_from_bbox


@dataclass(frozen=True)
class PPETask:
    name: str
    class_names: List[str]
    crop_fn: Callable
    dataset_root: str
    weights_path: str
    title_ok: str
    title_ng: str


TASKS: Dict[str, PPETask] = {
    "helmet": PPETask(
        name="helmet",
        class_names=["with_helmet", "without_helmet", "invalid"],
        crop_fn=crop_head_from_bbox,
        dataset_root="dataset/helmet",
        weights_path="weights/resnet_helmet.pth",
        title_ok="Helmet",
        title_ng="No Helmet",
    ),
    "vest": PPETask(
        name="vest",
        class_names=["with_vest", "without_vest", "invalid"],
        crop_fn=crop_torso_from_bbox,
        dataset_root="dataset/vest",
        weights_path="weights/resnet_vest.pth",
        title_ok="Reflective Vest",
        title_ng="No Reflective Vest",
    ),
}
