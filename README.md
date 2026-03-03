# PPE Detection (Helmet & Reflective Vest)

本项目当前仅保留并维护以下核心流程：

- `annotate_helmet.py` / `annotate_vest.py`：标注
- `train_helmet.py` / `train_vest.py`：训练
- `detect_helmet.py` / `detect_vest.py` / `detect_ppe.py`：推理

---

## 1. 安装

```bash
pip install -r requirements.txt
```

---

## 2. Pipeline 总览

### 2.1 标注阶段（按任务拆分）

- 使用 YOLO 人体检测（`classes=[0]`）先定位人。
- 基于人体框做规则裁剪：
  - 头盔任务：裁剪头部区域
  - 反光衣任务：裁剪上半身区域
- 仅保存有效 crop，按键标注：
  - `1`：合规（with）
  - `2`：违规（without）
  - `3`：invalid
  - `q`：退出

数据目录：

```text
dataset/helmet/
  with_helmet/
  without_helmet/
  invalid/

dataset/vest/
  with_vest/
  without_vest/
  invalid/
```

运行：

```bash
python annotate_helmet.py
python annotate_vest.py
```

---

### 2.2 训练阶段（分类模型）

模型为 `HeadHelmetResNet(num_classes=3)`，类别顺序固定为：

- helmet: `[with_helmet, without_helmet, invalid]`
- vest: `[with_vest, without_vest, invalid]`

训练核心 trick：

1. **类别权重（class_weights）**：提高违规类别学习强度。
2. **非对称惩罚（false_violation_penalty）**：
   - 在 GT 为合规时，额外惩罚预测为违规的概率；
   - 目标是尽量避免“把合规误报为违规”。

默认训练参数：

- `epochs=40`
- `batch_size=32`
- `lr=1e-4`
- `weight_decay=5e-4`
- helmet: `class_weights=[1.0, 2.0, 1.0]`
- vest: `class_weights=[1.0, 4.0, 1.0]`
- `false_violation_penalty=3.0`

运行：

```bash
python train_helmet.py
python train_vest.py
```

输出权重：

- `weights/resnet_helmet.pth`
- `weights/resnet_vest.pth`

---

### 2.3 推理阶段（单任务）

单任务推理脚本：

```bash
python detect_helmet.py
python detect_vest.py
```

关键机制（减少误报）：

1. **阈值分层决策**
   - `ok_threshold`：合规阈值
   - `violation_threshold`：违规阈值（通常更高）
   - `invalid_threshold`：invalid 阈值
2. **轨迹级证据累积**
   - `score_decay`：历史衰减
   - `score_step`：违规帧加分
   - `trigger_score`：触发告警分数
   - `min_violation_streak`：最小连续违规帧
   - `clear_ok_streak`：连续合规后清空证据

这套策略能缓解短时抖动、遮挡和 `invalid` 带来的误触发。

---

### 2.4 推理阶段（头盔+反光衣联合）

新增脚本：

```bash
python detect_ppe.py
```

功能：

- 同时运行头盔分类与反光衣分类。
- 右侧三列实时展示：
  1. `No Helmet`
  2. `No Vest`
  3. `No Helmet + No Vest`
- 自动处理状态迁移：
  - 若某 ID 先进入 `No Helmet`，后续又确认 `No Vest`，会**自动从第一列移除并加入第三列**。

说明：

- 由于有较长 `invalid/unknown` 阶段，联合脚本采用“每帧重算归属列”的方式，确保最终列归属与当前证据一致。

---

## 3. 可调参数建议

- 现场误报偏多：
  - 提高 `violation_threshold`
  - 提高 `min_violation_streak`
  - 提高 `trigger_score`
- 漏报偏多：
  - 适当降低 `violation_threshold`
  - 降低 `min_violation_streak`
- 训练阶段若“合规被误判违规”：
  - 增大 `false_violation_penalty`
  - 适当上调违规类/合规类权重比例做平衡

---

## 4. 当前代码结构（精简后）

```text
annotate_helmet.py
annotate_vest.py
train_helmet.py
train_vest.py
detect_helmet.py
detect_vest.py
detect_ppe.py
ppe/
  annotation.py
  training.py
  inference.py
  tasks.py
model/
  resnet.py
  resnet_data.py
  utils.py
README.md
requirements.txt
```
