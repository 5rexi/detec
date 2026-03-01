from torch.utils.data import Dataset, DataLoader
import cv2
import os
import random
import torchvision.transforms as T
import torch
from torchvision.transforms import functional as F
from PIL import Image

transform = T.Compose([
    T.ToTensor()
])

class ColorSensitiveAugmentation:
    """适用于安全帽识别的轻量数据增强，颜色敏感"""
    def __init__(self, resize=224):
        self.resize = resize

    def __call__(self, img):
        """
        img: PIL Image
        return: Tensor [C,H,W] in [0,1]
        """
        # 1. Resize
        img = F.resize(img, (self.resize, self.resize))

        # 2. Random horizontal flip
        if random.random() < 0.5:
            img = F.hflip(img)

        # 3. Random rotation ±10°
        angle = random.uniform(-10, 10)
        img = F.rotate(img, angle)

        # 4. Random affine (轻微缩放和平移)
        translate = (random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05))
        scale = random.uniform(0.95, 1.05)
        img = F.affine(img, angle=0, translate=(int(translate[0]*self.resize), int(translate[1]*self.resize)),
                       scale=scale, shear=0)

        # 5. Random brightness / contrast
        brightness_factor = random.uniform(0.9, 1.1)
        contrast_factor = random.uniform(0.9, 1.1)
        img = F.adjust_brightness(img, brightness_factor)
        img = F.adjust_contrast(img, contrast_factor)

        # 6. To Tensor
        img = F.to_tensor(img)

        # 7. Add Gaussian noise
        noise = torch.randn_like(img) * 0.01  # sigma=0.01，轻微
        img = img + noise
        img = torch.clamp(img, 0.0, 1.0)

        return img

class HeadDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        root/
          helmet/
          no_helmet/
        """
        self.samples = []
        self.transform = transform
        self.color_transformation = ColorSensitiveAugmentation()

        # for label, cls in enumerate(["with_helmet", "without_helmet", "invalid"]):
        for label, cls in enumerate(["with_clothing", "without_clothing", "invalid"]):
            cls_dir = os.path.join(root, cls)
            for name in os.listdir(cls_dir):
                self.samples.append(
                    (os.path.join(cls_dir, name), label)
                )

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # img = cv2.imread(path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.color_transformation(img)
            # img = self.transform(img)

        return img, label

def collate_fn(batch):
    imgs, labels = zip(*batch)

    max_h = max(img.shape[1] for img in imgs)
    max_w = max(img.shape[2] for img in imgs)

    padded_imgs = []
    for img in imgs:
        c, h, w = img.shape
        pad = torch.zeros((c, max_h, max_w))
        pad[:, :h, :w] = img
        padded_imgs.append(pad)

    return torch.stack(padded_imgs), torch.tensor(labels)

def generate_dataset(root, batch_size, transform=transform):
    dataset = HeadDataset(root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    return loader

