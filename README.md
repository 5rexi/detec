# 头盔与反光衣检测器

## 安装教程

### （可选） 0 创建环境
```bash
conda create -n detec python=3.12
conda activate detec
```
### 1 安装所需环境
```bash
pip install -r requirements.txt
```
### 2 运行代码
执行头盔检测：
```bash
python hamlet_detection.py
```
若要修改检测视频则需要修改`hamlet_detection.py`中第19行的`path`路径内容。

执行反光衣检测：
```bash
python clothing_detection.py
```
若要修改检测视频则需要修改`clothing_detection.py`中第19行的`path`路径内容。