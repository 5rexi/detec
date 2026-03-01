import torch
import torch.nn as nn
from resnet import HeadHelmetResNet
from resnet_data import generate_dataset

model = HeadHelmetResNet(num_classes=3).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
weight = torch.tensor([4.0, 1.0, 1.0]).cuda()
criterion = nn.CrossEntropyLoss(weight=weight)
loader = generate_dataset(root='./dataset_cloth',
                          batch_size=32)

min_loss = 1000000
for epoch in range(40):
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs = imgs.cuda()
        labels = labels.cuda()

        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if total_loss < min_loss:
        min_loss = total_loss
        torch.save(model.state_dict(), './weights/resnet_model_cloth.pth')

    print(f"Epoch {epoch}, loss = {total_loss:.4f}")