import argparse
import os
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model.resnet import HeadHelmetResNet
from ppe.tasks import TASKS


COMPLIANT_CLASS = 0
VIOLATION_CLASS = 1
INVALID_CLASS = 2


class PPEClassificationDataset(Dataset):
    def __init__(self, root: str, class_names: List[str], transform=None):
        self.samples = []
        self.transform = transform

        for class_id, class_name in enumerate(class_names):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)
                if os.path.isfile(file_path):
                    self.samples.append((file_path, class_id))

        if not self.samples:
            raise RuntimeError(f"No samples found in {root}. Please check dataset folders.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, label = self.samples[index]
        image = Image.open(file_path).convert("RGB")
        image = self.transform(image) if self.transform else image
        return image, label


class AsymmetricPPELoss(nn.Module):
    """
    目标：
    1) 严格惩罚“合规 -> 违规”误判（业务强约束）
    2) 保持对“违规样本”召回的学习能力

    实现方式：
    - 基础项：CrossEntropyLoss（支持 class_weight）
    - 额外项：当 GT=合规 时，额外惩罚模型对违规类的概率
    """

    def __init__(self, class_weights: List[float], false_violation_penalty: float):
        super().__init__()
        self.false_violation_penalty = false_violation_penalty
        if class_weights:
            weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.base_ce = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.base_ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = self.base_ce(logits, labels)

        probs = torch.softmax(logits, dim=1)
        compliant_mask = labels == COMPLIANT_CLASS

        if compliant_mask.any():
            # 仅在 GT=合规时，对“预测违规概率”增加罚项，降低误报违规
            violation_probs = probs[compliant_mask, VIOLATION_CLASS]
            penalty_loss = violation_probs.mean()
        else:
            penalty_loss = torch.tensor(0.0, device=logits.device)

        return ce_loss + self.false_violation_penalty * penalty_loss


def train_task(
    task_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    class_weights: List[float],
    false_violation_penalty: float,
    dataset_root: str,
    save_path: str,
) -> None:
    task = TASKS[task_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = PPEClassificationDataset(dataset_root, task.class_names, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = HeadHelmetResNet(num_classes=len(task.class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = AsymmetricPPELoss(
        class_weights=class_weights,
        false_violation_penalty=false_violation_penalty,
    ).to(device)

    best_loss = float("inf")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[{task.name}] epoch={epoch + 1}/{epochs}, avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"[saved] {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train helmet or vest classifier.")
    parser.add_argument("--task", choices=TASKS.keys(), required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--class-weights", nargs="*", type=float, default=[])
    parser.add_argument(
        "--false-violation-penalty",
        type=float,
        default=3.0,
        help="Extra penalty weight for compliant->violation false alarms.",
    )
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--save-path", default=None)
    args = parser.parse_args()

    task = TASKS[args.task]
    train_task(
        task_name=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=args.class_weights,
        false_violation_penalty=args.false_violation_penalty,
        dataset_root=args.dataset_root or task.dataset_root,
        save_path=args.save_path or task.weights_path,
    )
