# PPE Detection (Helmet & Reflective Vest)

本项目已按任务拆分为**两条独立流程**：

- 头盔流程（标注 / 训练 / 推理）
- 反光衣流程（标注 / 训练 / 推理）

这样可以避免“二合一标注”和脚本耦合导致的维护困难。

## 1. 安装

```bash
pip install -r requirements.txt
```

## 2. 数据目录规范（已拆分）

### 头盔数据集

```text
dataset/helmet/
  with_helmet/
  without_helmet/
  invalid/
```

### 反光衣数据集

```text
dataset/vest/
  with_vest/
  without_vest/
  invalid/
```

## 3. 标注（分开执行）

### 标注头盔

```bash
python annotate_helmet.py
```

### 标注反光衣

```bash
python annotate_vest.py
```

> 交互按键：`1=合规`，`2=违规`，`3=无效`，`q=退出`

## 4. 训练（分开执行）

### 训练头盔分类器

```bash
python train_helmet.py
```

### 训练反光衣分类器

```bash
python train_vest.py
```

默认输出权重：

- `weights/resnet_helmet.pth`
- `weights/resnet_vest.pth`

## 5. 应用（分开执行）

### 头盔检测

```bash
python detect_helmet.py
```

### 反光衣检测

```bash
python detect_vest.py
```

## 6. 新增模块说明

- `ppe/tasks.py`：任务配置中心（类别名、裁剪函数、默认路径）
- `ppe/annotation.py`：统一标注逻辑（按任务参数化）
- `ppe/training.py`：统一训练逻辑（按任务参数化）
- `ppe/inference.py`：统一视频推理逻辑（按任务参数化）

## 7. 兼容旧脚本

以下旧脚本仍可运行，但已改为调用新的拆分逻辑：

- `hamlet_detection.py` -> 头盔检测
- `clothing_detection.py` -> 反光衣检测
- `dataset_maker.py` -> 通过 `--task` 指定标注任务
- `model/resnet_train.py` -> 通过 `--task` 指定训练任务

推荐优先使用新的显式脚本（`annotate_* / train_* / detect_*`）。

## 8. 业务约束对应的训练策略（已在代码实现）

为满足“不能把合规识别成违规，同时违规只要一帧命中即可”的目标，训练代码已引入**非对称损失**：

- 基础损失：`CrossEntropyLoss`（可配置 `--class-weights`）
- 额外罚项：当 GT 为合规类别时，显式惩罚模型对违规类别的预测概率

可通过参数调整强度：

```bash
python ppe/training.py --task helmet --false-violation-penalty 3.0
```

`false-violation-penalty` 越大，越保守，越不容易把合规报成违规。


## 9. 线上误报控制（已在推理代码实现）

为了避免把“已佩戴”误报成“未佩戴”，推理阶段采用保守判违规策略：

- 只有 `P(violation) >= violation_threshold` 才记为违规候选（默认 0.95）
- `P(ok) >= ok_threshold` 直接判合规
- 其余全部为 `Detecting/Unknown`，避免误报
- 轨迹级使用证据分数 `score`，达到 `trigger_score` 且连续违规帧数达到 `min_violation_streak` 才最终告警

示例：

```bash
python ppe/inference.py --task helmet --video your.mp4 --violation-threshold 0.97 --trigger-score 1.2 --min-violation-streak 3
```
