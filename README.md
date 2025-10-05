# ChinaMoblieCup_Transformer
# ViT目标检测模型实现说明文档


## 项目概述
本项目基于Vision Transformer（ViT）架构实现了多个版本的目标检测模型，包含`ViT.py`、`ViT2_ez.py`、`ViT3_base.py`三个核心文件。模型通过将图像分割为补丁（Patch），利用Transformer提取全局特征，结合自定义检测头实现目标边界框和类别的预测，适用于通用目标检测任务。


## 文件结构与核心功能
### 1. `ViT3_base.py`
#### 核心组件
- **`CustomDetectionDataset` 类**：自定义数据集加载器，负责：
  - 加载图像（支持`jpg`/`png`格式，兼容大小写）和对应标签（`txt`格式）
  - 图像预处理（尺寸调整、转Tensor、标准化）
  - 标签格式转换（将相对坐标转为绝对坐标，处理无目标样本）

- **`DetectionHead` 类**：目标检测头，由三个子网络组成：
  - 边界框预测头：通过全连接层预测边界框参数（输出范围0-1）
  - 类别预测头：预测目标类别（含背景类）
  - 置信度预测头：预测目标存在的置信度（输出范围0-1）

- **`train_model` 函数**：主训练流程，包含：
  - 数据加载器初始化
  - 模型、优化器、调度器、损失函数配置
  - 训练/验证循环（记录损失、保存最佳模型和检查点）
  - 训练日志保存（损失曲线、最佳验证损失）


### 2. `ViT.py`
#### 核心组件
- **`nms_boxes` 函数**：非极大值抑制（NMS）后处理，用于去除重复预测框：
  - 按置信度降序排序预测框
  - 迭代保留高置信度框，过滤与当前框IoU超过阈值的框

- **ViT主干网络 `forward` 方法**：
  - 图像补丁嵌入（将图像转为补丁序列并映射到特征维度）
  - 拼接类别令牌（cls_token）和位置编码
  - Transformer编码器提取全局特征

- **检测头前向传播**：
  - 将Transformer输出的补丁特征输入检测头，得到原始预测（边界框参数、置信度、类别logits）
  - 边界框参数转换（相对坐标→绝对坐标）
  - 置信度与类别分数融合（最终置信度=置信度×类别最高分）

- **损失函数 `forward` 方法**：
  - 处理无真实目标样本（仅优化置信度损失）
  - 计算预测框与真实框的IoU矩阵，实现正负样本匹配（每个真实框匹配IoU最大的预测框）
  - 多损失融合：边界框损失（SmoothL1）、类别损失（CrossEntropy）、置信度损失（BCEWithLogits）

- **`calculate_simple_map` 函数**：简化版mAP计算（平均精度均值），用于模型评估：
  - 按类别统计预测框与真实框
  - 计算真正例（TP）、假正例（FP），生成Precision-Recall曲线
  - 梯形积分计算AP，平均所有类别AP得到mAP


### 3. `ViT2_ez.py`
#### 核心改进与功能
- **窗口注意力机制**：在Transformer编码器中引入窗口注意力（`WindowAttention`类）：
  - 对补丁序列分窗口计算注意力，减少计算量
  - 支持补丁序列长度补零以适配窗口大小
  - 窗口内注意力计算与特征重组

- **简化版检测流程**：
  - 优化补丁特征处理（分离cls_token与补丁令牌，仅对补丁令牌应用Transformer层）
  - 检测头输出格式简化（固定维度为9，适配特定类别数）
  - 损失函数实现更简洁（明确正负样本掩码处理，优化设备兼容性）

- **训练流程优化**：
  - 增强GPU检测与提示（训练前检查CUDA可用性）
  - 简化参数配置，降低使用门槛


## 模型原理
### 1. 整体流程
1. **图像预处理**：图像 resize 至固定尺寸，转换为Tensor并标准化
2. **补丁嵌入**：将图像分割为`16×16`或`32×32`的补丁，通过线性层映射为特征向量
3. **Transformer编码**：
   - 拼接cls_token（用于全局特征聚合）
   - 添加位置编码（注入空间位置信息）
   - 通过Transformer编码器（自注意力/窗口注意力）提取全局特征
4. **检测头预测**：对每个补丁特征预测：
   - 边界框参数（中心坐标、宽高）
   - 目标置信度（是否包含目标）
   - 类别概率（目标所属类别）
5. **后处理**：
   - 边界框坐标转换（相对→绝对）
   - NMS去除重复框
   - 置信度筛选（保留高置信度预测）


### 2. 损失计算
总损失 = 边界框损失×λ_box + 类别损失×λ_cls + 置信度损失  
- **边界框损失**：SmoothL1损失（对异常值更鲁棒），仅计算正样本（匹配到真实框的预测）
- **类别损失**：交叉熵损失，仅计算正样本的类别预测
- **置信度损失**：二分类交叉熵（BCE），正样本目标为1，负样本目标为0


## 使用方法
### 1. 环境依赖
- Python 3.8+
- PyTorch 1.10+
- OpenCV-python
- NumPy
- 可选：CUDA 11.0+（加速训练）


### 2. 数据集准备
- 目录结构：
  ```
  dataset_root/
  ├── images/
  │   ├── train/  # 训练图像（jpg/png）
  │   └── val/    # 验证图像（jpg/png）
  └── labels/
      ├── train/  # 训练标签（txt，与图像同名）
      └── val/    # 验证标签（txt，与图像同名）
  ```
- 标签格式（YOLO格式）：每个txt文件每行表示一个目标，格式为 `cls_id cx cy bw bh`（相对坐标，范围0-1）
  - `cls_id`：类别索引（整数）
  - `cx, cy`：目标中心相对坐标
  - `bw, bh`：目标宽高相对坐标


### 3. 模型训练
#### 运行方式
- 直接运行对应版本的主函数：
  ```bash
  # 运行ViT3_base版本
  python ViT3_base.py

  # 运行ViT版本
  python ViT.py

  # 运行ViT2_ez版本
  python ViT2_ez.py
  ```

#### 训练配置（可在代码中修改）
- `IMG_SIZE`：输入图像尺寸（如`(832, 1472)`）
- `PATCH_SIZE`：补丁大小（如32）
- `EPOCHS`：训练轮数
- `BATCH_SIZE`：批次大小
- `LOSS_LAMBDA_BOX`/`LOSS_LAMBDA_CLS`：损失权重
- `BEST_MODEL_PATH`：最佳模型保存路径


### 4. 模型评估
- 训练过程中自动计算验证损失，保存最佳模型
- 可调用`calculate_simple_map`函数计算mAP（需传入预测结果和真实标签）：
  ```python
  # 示例
  map_score = calculate_simple_map(all_preds, all_targets, iou_threshold=0.5)
  print(f"mAP@0.5: {map_score:.4f}")
  ```


## 注意事项
1. 确保数据集路径正确，图像与标签文件一一对应
2. 无目标的图像需对应空标签文件（或代码自动处理为`[0,0,0,0,-1]`）
3. 训练大型模型建议使用GPU（显存≥8GB）
4. 可通过调整`PATCH_SIZE`、`EPOCHS`、学习率等参数优化性能
5. 模型输出预测框格式为`[x1, y1, x2, y2, confidence, cls_id]`（绝对像素坐标）


## 版本差异说明
| 版本         | 核心特点                     | 适用场景                 |
|--------------|------------------------------|--------------------------|
| `ViT.py`     | 基础ViT架构，完整损失与评估  | 通用目标检测，需要标准评估 |
| `ViT2_ez.py` | 窗口注意力，简化流程        | 轻量化部署，快速训练     |
| `ViT3_base.py` | 模块化设计，完善数据集处理  | 工程化应用，定制化需求   |


以下是对三个核心文件（`ViT5_4k.py`、`vote_config.py`、`pred.py`）的功能说明及使用指南：


### 1. `ViT5_4k.py`：核心模型与训练框架
#### 功能说明
该文件是基于Vision Transformer（ViT）的目标检测系统核心实现，包含完整的模型定义、数据处理、损失函数和训练流程，主要功能如下：
- **数据集处理**：通过`CustomDetectionDataset`类实现数据加载、增强（如随机裁剪、水平翻转、小目标优先增强）和标签归一化，支持多种图像格式（jpg、png等，兼容大小写）。
- **模型结构**：`ViTDetector`类基于预训练的`vit_b_16`构建，包含：
  - 特征提取 backbone（支持冻结指定层数的Transformer层）
  - 检测头（`DetectionHead`）：输出边界框坐标、类别概率和置信度
- **损失函数**：`DetectionLoss`类实现多组件加权损失（边界框损失用SmoothL1Loss，对小目标增加权重；类别损失用CrossEntropyLoss；置信度损失用BinaryCrossEntropyLoss）。
- **训练管理**：`Trainer`类封装训练流程，支持学习率预热（warmup）+余弦退火调度、梯度累积、模型保存（最佳模型和检查点）等。

#### 使用指南
1. **配置参数**：修改文件头部的核心配置（需重点关注）：
   ```python
   DATASET_ROOT = "你的数据集路径"  # 需包含images/train、labels/train等子目录
   LOCAL_VIT_WEIGHTS_PATH = "预训练ViT权重路径"  # 本地权重路径（可选）
   EPOCHS = 200  # 训练轮数
   BATCH_SIZE = 8  # 批次大小
   FREEZE_LAYERS = 4  # 冻结的Transformer层数
   ```
2. **启动训练**：直接运行脚本即可启动训练：
   ```bash
   python ViT5_4k.py
   ```
3. **输出文件**：训练结果保存至`TRAIN_OUTPUT_DIR`（默认`./runs/vit_detector`），包含：
   - 最佳模型：`weights/best.pt`
   - 训练日志和检查点


### 2. `vote_config.py`：模型推理与预测结果生成
#### 功能说明
该文件负责加载训练好的模型并生成预测结果，核心功能包括：
- **模型加载**：`load_vit_model`函数加载保存的模型权重，自动适配CPU/GPU设备。
- **单图预测**：`vit_predict_image`函数对单张图片进行推理，输出边界框（含坐标、置信度、类别），并应用非极大值抑制（NMS）过滤冗余框。
- **批量预测**：`create_vit_predictions`函数批量处理验证集图片，生成预测标签文件（COCO TXT格式，含归一化坐标），并保存至`./ViT_Pred`目录。
- **标签处理**：`load_labels`函数加载真实标签或预测标签，转换为绝对坐标用于后续评估。

#### 使用指南
1. **生成预测结果**：调用`create_vit_predictions`函数，示例：
   ```python
   # 在文件末尾main函数中配置路径
   model_path = "./runs/vit_detector/vit_detector/weights/best.pt"  # 训练好的模型路径
   val_images_path = "数据集/images/val"  # 验证集图片路径
   val_labels_path = "数据集/labels/val"  # 验证集标签路径
   pred_dir, _, _ = create_vit_predictions(model_path, val_images_path, val_labels_path, conf=0.5)
   ```
2. **参数说明**：
   - `conf`：置信度阈值（默认0.5，过滤低置信度预测框）
   - `NMS_IOU_THRESHOLD`：NMS的IoU阈值（默认0.5，控制重复框过滤严格程度）
3. **输出文件**：预测结果保存至`./ViT_Pred`，包含：
   - `images/`：复制的验证集图片
   - `labels/`：预测标签文件（格式：`类别ID 中心x 中心y 宽度 高度 置信度`）
   - `data.yaml`：数据集配置文件（含类别信息）


### 3. `pred.py`：模型评估（mAP计算）
#### 功能说明
该文件用于计算目标检测模型的评估指标，核心功能如下：
- **指标计算**：`evaluate_map`函数计算不同IoU阈值（0.5~0.95，步长0.05）下的平均精度（AP），包括：
  - `mAP50`：IoU=0.5时的平均精度
  - `mAP50-95`：IoU从0.5到0.95的平均精度
  - 各类别的AP值
- **辅助函数**：`calculate_iou`计算边界框交并比，`compute_ap`计算单类别AP，`load_labels`加载标签用于评估。

#### 使用指南
1. **配置评估路径**：修改文件中的路径参数：
   ```python
   GT_LABELS_DIR = "数据集/labels/val"  # 真实标签目录
   PRED_LABELS_DIR = "./ViT_Pred/labels"  # 预测标签目录（vote_config.py生成）
   IMAGES_DIR = "数据集/images/val"  # 验证集图片目录（用于获取图片尺寸）
   ```
2. **运行评估**：直接执行脚本生成评估结果：
   ```bash
   python pred.py
   ```
3. **输出结果**：控制台将打印：
   - mAP50和mAP50-95数值
   - 各IoU阈值下的mAP
   - 各类别的AP@0.50


### 三者关系与工作流
1. **训练**：通过`ViT5_4k.py`训练模型，得到权重文件（`best.pt`）。
2. **推理**：使用`vote_config.py`加载权重，对验证集图片生成预测标签。
3. **评估**：通过`pred.py`对比真实标签和预测标签，计算mAP指标评估模型性能。

通过该工作流可完成模型的训练、推理与评估全流程，核心参数可根据实际需求在各文件中调整。
