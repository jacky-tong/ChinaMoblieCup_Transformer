import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import psutil
import cv2
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import shutil


# ============================== 全局配置（所有参数集中修改此处）==============================
# 1. 路径配置（根据自己的目录调整）
DATASET_ROOT = r"E:\2025CompetitionLog\9.9\tRVALCUP_2\ChinaMoblieCup\dataset"               # 数据集根目录（需包含images/train、labels/val等子目录）
TRAIN_OUTPUT_DIR = "./runs/vit_detect"   # 训练结果保存目录（权重、日志等）
MODEL_NAME = "vit_detector_single_gpu"   # 实验名称（区分不同训练任务）
BEST_MODEL_PATH = os.path.join(TRAIN_OUTPUT_DIR, MODEL_NAME, "weights", "best.pt")  # 最佳权重路径
VAL_IMAGES_PATH = os.path.join(DATASET_ROOT, "images", "val")  # 验证集图像路径
VAL_LABELS_PATH = os.path.join(DATASET_ROOT, "labels", "val")  # 验证集标签路径

# 2. 模型超参数（单GPU显存友好型配置）
IMG_SIZE = (832, 1472)  # 输入图像尺寸 (高, 宽)，需能被PATCH_SIZE整除
PATCH_SIZE = 32         # ViT图像分块大小（32×32像素/块）
DIM = 512               # Transformer特征维度（原1024，单卡降为512减少显存占用）
DEPTH = 4               # Transformer编码器层数（原6，单卡降为4降低计算量）
HEADS = 8               # 多头注意力头数（需整除DIM，512÷8=64，符合注意力机制要求）
MLP_DIM = 1024          # MLP隐藏层维度（原2048，单卡减半）
NUM_CLASSES = 4         # 目标类别数（ship/people/car/motor）

# 3. 训练超参数（单卡适配）
EPOCHS = 200            # 训练轮次（原300，单卡适当减少以缩短时间）
BATCH_SIZE = 4          # 单卡批次大小（根据显存调整：12GB显存建议2-4，24GB建议4-8）
BASE_LR = 5e-5          # 基础学习率（单卡无需缩放，原1e-4易震荡）
WEIGHT_DECAY = 1e-4     # 权重衰减（防止过拟合）
WARMUP_EPOCHS = 3       # 学习率预热轮次（单卡无需长预热）
LOSS_LAMBDA_BOX = 5.0   # 边界框损失权重（突出位置预测精度）
LOSS_LAMBDA_CLS = 1.0   # 类别损失权重

# 4. 推理超参数
CONF_THRESHOLD = 0.5    # 置信度阈值（过滤低置信度预测框）
IOU_THRESHOLD = 0.5     # NMS交并比阈值（去除重复框）

# 5. 设备配置（自动适配单GPU/CPU）
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
# =====================================================================================


# ============================== 核心工具函数（避免外部依赖）==============================
def nms_boxes(boxes, iou_threshold):
    """内置NMS函数（无需依赖vote_config.py）：去除重复预测框"""
    if len(boxes) == 0:
        return []
    
    # 按置信度降序排序
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep_boxes = []
    
    while boxes:
        current_box = boxes.pop(0)
        keep_boxes.append(current_box)
        
        # 计算当前框与剩余框的IoU，过滤超过阈值的框
        boxes = [
            box for box in boxes
            if calculate_iou(np.array(current_box[:4]), np.array(box[:4])) < iou_threshold
        ]
    
    return keep_boxes

def calculate_iou(box1, box2):
    """计算两个边界框的IoU（交并比）"""
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    if x2_min <= x1_max or y2_min <= y1_max:
        return 0.0
    
    intersection = (x2_min - x1_max) * (y2_min - y1_max)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


# ============================== 模型定义（ViT特征提取 + 检测头）==============================
class ViTBackbone(nn.Module):
    """Vision Transformer特征提取器：将图像→分块→嵌入→Transformer编码"""
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, mlp_dim=MLP_DIM):
        super().__init__()
        image_height, image_width = img_size
        patch_height, patch_width = patch_size, patch_size
        
        # 计算补丁数量和单个补丁维度（3通道×补丁高×补丁宽）
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = 3 * patch_height * patch_width
        
        # 1. 图像分块 + 线性嵌入（将每个3×32×32的补丁转为512维特征）
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, dim),
        )
        
        # 2. 位置编码（可学习，给每个补丁添加"位置信息"，避免Transformer忽略顺序）
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))  # +1是cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 分类token（整合全局特征）
        
        # 3. Transformer编码器（堆叠4层，每层含多头注意力+MLP，提取全局关联特征）
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=0.1,
                batch_first=True  # 输入格式：(batch, 补丁数, 特征维度)
            ),
            num_layers=depth
        )

    def forward(self, img):
        # img输入格式：(batch, 3, 832, 1472)
        batch_size = img.shape[0]
        
        # 步骤1：生成补丁嵌入（(batch, 3, 832, 1472) → (batch, 1176, 512)，1176=26×45，即832/32=26，1472/32=45）
        x = self.to_patch_embedding(img)
        
        # 步骤2：添加cls_token（在补丁特征前拼接，格式变为(batch, 1177, 512)）
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 步骤3：添加位置编码（给每个补丁+cls_token注入位置信息）
        x += self.pos_embedding
        
        # 步骤4：Transformer编码（提取全局特征，输出格式不变）
        x = self.transformer(x)
        
        return x  # 返回含cls_token的特征：(batch, 1177, 512)


class ViTDetector(nn.Module):
    """基于ViT的目标检测器：Backbone提特征 + 检测头预测框/类别"""
    def __init__(self, num_classes=NUM_CLASSES, img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=DIM, depth=DEPTH, heads=HEADS):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        
        # 1. 特征提取（调用ViT Backbone）
        self.backbone = ViTBackbone(img_size=img_size, patch_size=patch_size, dim=dim, depth=depth, heads=heads)
        
        # 2. 检测头（将每个补丁的512维特征→预测6个值：4坐标+1置信度+4类别）
        self.detection_head = nn.Sequential(
            nn.Linear(dim, dim // 2),  # 降维：512→256，减少计算量
            nn.GELU(),                 # 激活函数（比ReLU更适合Transformer，缓解梯度消失）
            nn.Dropout(0.2),           #  dropout：防止过拟合
            nn.Linear(dim // 2, 5 + num_classes)  # 输出层：5(框+置信度)+4(类别)=9维
        )
        
        # 权重初始化（保证训练稳定性）
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)  # 截断正态分布：避免权重过大
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)  # LayerNorm权重初始化为1

    def forward(self, x):
        # x输入格式：(batch, 3, 832, 1472)
        batch_size = x.shape[0]
        h, w = self.img_size
        
        # 步骤1：获取ViT特征，去掉cls_token（只保留补丁特征：(batch, 1176, 512)）
        features = self.backbone(x)
        patch_features = features[:, 1:]
        
        # 步骤2：检测头预测（每个补丁输出(5 + num_classes)维原始值：(batch, 1176, 5+num_classes)）
        detections = self.detection_head(patch_features)  # 原始输出（未经sigmoid/softmax/坐标转换）
        
        # 以下为推理/展示用的后处理（与之前代码一致）
        bboxes = detections[..., :4]  # 原始bbox参数
        confidences = torch.sigmoid(detections[..., 4:5])  # 置信度（0-1）
        class_logits = detections[..., 5:]  # 类别logits
        
        # 相对参数 -> 绝对坐标（与之前一致）
        bboxes[..., 0] = (bboxes[..., 0] * 2 - 1) * w / 2  # x_center
        bboxes[..., 1] = (bboxes[..., 1] * 2 - 1) * h / 2  # y_center
        bboxes[..., 2] = torch.exp(bboxes[..., 2]) * (w / (w // self.patch_size))  # width
        bboxes[..., 3] = torch.exp(bboxes[..., 3]) * (h / (h // self.patch_size))  # height
        
        x1 = bboxes[..., 0] - bboxes[..., 2] / 2
        y1 = bboxes[..., 1] - bboxes[..., 3] / 2
        x2 = bboxes[..., 0] + bboxes[..., 2] / 2
        y2 = bboxes[..., 1] + bboxes[..., 3] / 2
        
        class_ids = torch.argmax(F.softmax(class_logits, dim=-1), dim=-1, keepdim=True)
        class_scores = torch.max(F.softmax(class_logits, dim=-1), dim=-1, keepdim=True).values
        final_confidence = confidences * class_scores
        
        processed_outputs = torch.cat([x1.unsqueeze(-1), y1.unsqueeze(-1), x2.unsqueeze(-1), y2.unsqueeze(-1), final_confidence, class_ids.float()], dim=-1)
        
        # 返回： (processed_outputs_for_inference, raw_detection_head_outputs_for_loss)
        return processed_outputs, detections


# ============================== 损失函数（边界框+类别+置信度联合优化）==============================
class DetectionLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, lambda_box=LOSS_LAMBDA_BOX, lambda_cls=LOSS_LAMBDA_CLS):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box  # 边界框损失权重
        self.lambda_cls = lambda_cls  # 类别损失权重
        
        # 基础损失函数
        self.bbox_loss = nn.SmoothL1Loss(reduction='sum')  # 边界框损失（对异常值鲁棒，比MSE好）
        self.class_loss = nn.CrossEntropyLoss(reduction='sum')  # 类别损失
        self.confidence_loss = nn.BCEWithLogitsLoss(reduction='sum')  # 置信度损失（二分类：是/否目标）

    def forward(self, predictions, targets):
        """
        predictions：原始检测头输出 (batch, 1176, 5+num_classes)
        targets：列表 → 每个元素是(num_objects, 5) → 真实目标（x1,y1,x2,y2,类别ID），坐标为绝对像素（与 dataset 中一致）
        """
        total_loss = 0.0
        batch_size = predictions.shape[0]
        
        for i in range(batch_size):
            pred = predictions[i]  # (1176, 5+num_classes) 原始输出
            target = targets[i]    # (num_objects, 5)
            
            # 情况1：无真实目标 → 只优化置信度（希望为0）
            if target.numel() == 0:
                # pred[:,4] 为置信度 logit（BCEWithLogitsLoss 直接使用）
                conf_loss = self.confidence_loss(pred[:, 4], torch.zeros_like(pred[:, 4]))
                total_loss += conf_loss
                continue
            
            # 先将原始预测的 bbox 参数 转换为 与 processed_outputs 相同的 绝对坐标格式 x1,y1,x2,y2
            # pred_bbox_params: (1176,4) -> 与 ViTDetector.forward 的同样变换
            pred_bbox = pred[:, :4].clone()
            # 注意：pred_bbox 使用相同的数学变换（相对->绝对）
            w_img, h_img = self._get_img_wh()  # helper below uses global IMG_SIZE
            # x_center, y_center, w_pred, h_pred
            pred_bbox[:, 0] = (pred_bbox[:, 0] * 2 - 1) * w_img / 2
            pred_bbox[:, 1] = (pred_bbox[:, 1] * 2 - 1) * h_img / 2
            pred_bbox[:, 2] = torch.exp(pred_bbox[:, 2]) * (w_img / (w_img // PATCH_SIZE))
            pred_bbox[:, 3] = torch.exp(pred_bbox[:, 3]) * (h_img / (h_img // PATCH_SIZE))
            
            # 转换为 x1,y1,x2,y2 格式（与 target 一致）
            pred_x1 = pred_bbox[:, 0] - pred_bbox[:, 2] / 2
            pred_y1 = pred_bbox[:, 1] - pred_bbox[:, 3] / 2
            pred_x2 = pred_bbox[:, 0] + pred_bbox[:, 2] / 2
            pred_y2 = pred_bbox[:, 1] + pred_bbox[:, 3] / 2
            
            pred_boxes_abs = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)  # (1176,4)
            
            # 计算 IoU 矩阵（预测框 vs 真实目标），供匹配使用
            num_preds = pred_boxes_abs.shape[0]
            num_tgts = target.shape[0]
            ious = torch.zeros((num_preds, num_tgts), device=pred.device)
            for p_idx in range(num_preds):
                for t_idx in range(num_tgts):
                    # 原赋值会产生 numpy.float32 -> CUDA tensor 类型不兼容错误
                    # 修复：先转换为 Python float 再赋值
                    p_box_np = pred_boxes_abs[p_idx].detach().cpu().numpy()
                    t_box_np = target[t_idx, :4].detach().cpu().numpy()
                    val = float(calculate_iou(p_box_np, t_box_np))
                    ious[p_idx, t_idx] = val
            
            # 每个真实目标匹配IoU最大的预测框（作为正样本）
            matched_preds = torch.argmax(ious, dim=0)  # (num_tgts,)
            matched_targets = torch.arange(num_tgts, device=pred.device)
            
            # 生成正负样本掩码（用于置信度损失）
            positive_mask = torch.zeros(num_preds, dtype=torch.bool, device=pred.device)
            positive_mask[matched_preds] = True
            
            # 使用 matched_preds 按顺序取出对应的预测（保证与 target 一一对应）
            pred_matched_boxes = pred_boxes_abs[matched_preds]        # (num_tgts,4)
            target_matched_boxes = target[matched_targets, :4]        # (num_tgts,4)
            
            # 边界框损失（仅正样本）
            box_loss = self.bbox_loss(pred_matched_boxes, target_matched_boxes)
            
            # 类别损失：使用原始 class logits（pred[..., 5:5+num_classes]），不经过 argmax
            class_logits = pred[matched_preds, 5:5+self.num_classes]  # (num_tgts, num_classes)
            class_targets = target[matched_targets, 4].long()        # (num_tgts,)
            cls_loss = self.class_loss(class_logits, class_targets)
            
            # 置信度损失：使用置信度 logit（pred[:,4]），正负样本均参与
            conf_targets = torch.zeros(num_preds, device=pred.device)
            conf_targets[positive_mask] = 1.0
            conf_loss = self.confidence_loss(pred[:, 4], conf_targets)
            
            total_loss += self.lambda_box * box_loss + self.lambda_cls * cls_loss + conf_loss
        
        return total_loss / batch_size  # 批次平均损失

    # helper：返回图像宽高（与模型和 dataset 保持一致）
    def _get_img_wh(self):
        # IMG_SIZE 为 (H, W)
        return IMG_SIZE[1], IMG_SIZE[0]


# ============================== 数据集（加载YOLO格式数据）==============================
class CustomDetectionDataset(Dataset):
    """自定义数据集：加载图像+YOLO格式txt标签，自动适配输入尺寸"""
    def __init__(self, img_dir, label_dir, img_size=IMG_SIZE, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        
        # 默认数据预处理（可扩展）
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),  # HWC→CHW，像素值归一化到[0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #  imagenet标准化
        ])
        
        # 加载所有图像路径（支持多种格式）
        self.images = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.images.extend(list(self.img_dir.glob(f'*{ext}')))
            self.images.extend(list(self.img_dir.glob(f'*{ext.upper()}')))

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / (img_path.stem + '.txt')  # 标签文件与图像同名
        
        # 步骤1：加载并预处理图像
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR→RGB（cv2默认BGR，PyTorch默认RGB）
        h_origin, w_origin = image.shape[:2]
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))  # 缩放到832×1472
        
        # 步骤2：加载并转换标签（YOLO相对坐标→绝对坐标）
        targets = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # 跳过格式错误的行
                    
                    cls_id, cx_rel, cy_rel, bw_rel, bh_rel = map(float, parts)
                    
                    # 相对坐标→原始图像绝对坐标
                    x1_origin = (cx_rel - bw_rel/2) * w_origin
                    y1_origin = (cy_rel - bh_rel/2) * h_origin
                    x2_origin = (cx_rel + bw_rel/2) * w_origin
                    y2_origin = (cy_rel + bh_rel/2) * h_origin
                    
                    # 适配到缩放后的图像尺寸
                    x1 = x1_origin * self.img_size[1] / w_origin
                    y1 = y1_origin * self.img_size[0] / h_origin
                    x2 = x2_origin * self.img_size[1] / w_origin
                    y2 = y2_origin * self.img_size[0] / h_origin
                    
                    targets.append([x1, y1, x2, y2, cls_id])
        
        # 步骤3：应用数据增强（如定义了transform）
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(targets, dtype=torch.float32)


# ============================== 训练器（单卡训练逻辑）==============================
class ViTDetectorTrainer:
    def __init__(self, dataset_root=DATASET_ROOT):
        self.dataset_root = Path(dataset_root)
        self.data_yaml = self.dataset_root / 'data.yaml'
        self.class_names = ['ship', 'people', 'car', 'motor']  # 类别名称
        self.num_classes = NUM_CLASSES
        
        # 验证设备和数据集（单卡友好）
        self.validate_device()
        self.validate_dataset_structure()

    def validate_device(self):
        """验证单卡/CPU环境"""
        print("=" * 60)
        print("🖥️ 设备环境验证")
        print("=" * 60)
        if torch.cuda.is_available():
            print(f"✅ 检测到单GPU：{torch.cuda.get_device_name(0)}")
            print(f"   显存容量：{GPU_MEMORY_GB:.1f} GB")
        else:
            print("⚠️ 未检测到GPU，将使用CPU训练（速度较慢，建议优先配置GPU）")
        print(f"   最终使用设备：{DEVICE}")

    def validate_dataset_structure(self):
        """验证数据集目录结构（确保符合要求）"""
        required_dirs = [
            self.dataset_root / 'images' / 'train',
            self.dataset_root / 'images' / 'val',
            self.dataset_root / 'labels' / 'train',
            self.dataset_root / 'labels' / 'val'
        ]
        
        print("\n" + "=" * 60)
        print("📁 数据集结构验证")
        print("=" * 60)
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"目录不存在：{dir_path}（请检查数据集路径）")
            print(f"✓ {dir_path}")
        
        # 自动创建data.yaml（若不存在）
        if not self.data_yaml.exists():
            print(f"⚠ data.yaml不存在，将自动创建：{self.data_yaml}")
            self.create_data_yaml()
        else:
            print(f"✓ {self.data_yaml}")

    def create_data_yaml(self):
        """创建数据集配置文件"""
        dataset_config = {
            'train': './images/train',
            'val': './images/val',
            'nc': self.num_classes,
            'names': self.class_names
        }
        with open(self.data_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)

    def train_model(self):
        """单卡训练主逻辑"""
        print("\n" + "=" * 80)
        print(f"🚀 开始ViT目标检测训练（单卡/CPU）")
        print(f"📐 输入尺寸：{IMG_SIZE[0]}×{IMG_SIZE[1]} | 📦 批次大小：{BATCH_SIZE} | 🔄 训练轮次：{EPOCHS}")
        print("=" * 80)

        # 1. 创建模型并移至设备
        model = ViTDetector().to(DEVICE)
        print(f"✅ 模型初始化完成（参数总量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M）")

        # 2. 加载数据集
        train_dataset = CustomDetectionDataset(
            img_dir=self.dataset_root / 'images' / 'train',
            label_dir=self.dataset_root / 'labels' / 'train'
        )
        val_dataset = CustomDetectionDataset(
            img_dir=self.dataset_root / 'images' / 'val',
            label_dir=self.dataset_root / 'labels' / 'val'
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=min(4, psutil.cpu_count() // 2),  # 单卡无需多worker（避免CPU瓶颈）
            collate_fn=self.collate_fn,
            pin_memory=True if torch.cuda.is_available() else False  #  pinned memory加速GPU数据传输
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=min(4, psutil.cpu_count() // 2),
            collate_fn=self.collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )
        print(f"✅ 数据集加载完成（训练集：{len(train_dataset)}张 | 验证集：{len(val_dataset)}张）")

        # 3. 初始化优化器、损失函数、学习率调度
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=BASE_LR,
            weight_decay=WEIGHT_DECAY
        )
        criterion = DetectionLoss()
        # 余弦退火调度（比固定学习率收敛更稳定）
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=BASE_LR * 0.01  # 最终学习率为基础的1%
        )

        # 4. 创建保存目录
        save_dir = Path(TRAIN_OUTPUT_DIR) / MODEL_NAME
        weights_dir = save_dir / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 保存目录创建完成：{save_dir}")

        # 5. 训练循环
        best_map = 0.0
        for epoch in range(EPOCHS):
            # 训练阶段
            model.train()
            train_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [训练]")
            
            for images, targets in progress_bar:
                images = images.to(DEVICE)
                targets = [t.to(DEVICE) for t in targets]
                
                optimizer.zero_grad()
                # 注意：model 返回 (processed_outputs, raw_detections)
                processed_outputs, raw_outputs = model(images)
                loss = criterion(raw_outputs, targets)  # 使用原始输出计算损失
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})
            
            avg_train_loss = train_loss / len(train_loader)
            scheduler.step()  # 每轮更新学习率

            # 验证阶段（计算损失+mAP）
            model.eval()
            val_loss = 0.0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [验证]")
                for images, targets in val_bar:
                    images = images.to(DEVICE)
                    targets = [t.to(DEVICE) for t in targets]
                    
                    processed_outputs, raw_outputs = model(images)
                    loss = criterion(raw_outputs, targets)
                    val_loss += loss.item()
                    
                    # 收集预测和真实目标（用于计算mAP）
                    for i in range(processed_outputs.shape[0]):
                        pred_boxes = processed_outputs[i].cpu().numpy()
                        pred_boxes = pred_boxes[pred_boxes[:, 4] >= CONF_THRESHOLD]
                        pred_boxes = nms_boxes(pred_boxes, IOU_THRESHOLD)
                        all_preds.append(pred_boxes)
                        
                        target_boxes = targets[i].cpu().numpy()
                        all_targets.append(target_boxes)
                    
                    val_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
            
            avg_val_loss = val_loss / len(val_loader)
            # 计算mAP（简化版，完整可参考COCO评估标准）
            current_map = self.calculate_simple_map(all_preds, all_targets)

            # 打印轮次信息
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print(f"  训练损失：{avg_train_loss:.4f} | 验证损失：{avg_val_loss:.4f}")
            print(f"  当前学习率：{optimizer.param_groups[0]['lr']:.6f} | 验证mAP：{current_map:.4f}")

            # 保存模型
            # 1. 每25轮保存一次中间权重
            if (epoch + 1) % 25 == 0:
                torch.save(model.state_dict(), weights_dir / f"epoch_{epoch+1}.pt")
                print(f"  中间权重保存：{weights_dir / f'epoch_{epoch+1}.pt'}")
            
            # 2. 保存mAP最高的最佳模型
            if current_map > best_map:
                best_map = current_map
                torch.save(model.state_dict(), weights_dir / "best.pt")
                print(f"  最佳模型更新：{weights_dir / 'best.pt'}（当前mAP：{best_map:.4f}）")
            
            # 3. 保存最新模型
            torch.save(model.state_dict(), weights_dir / "last.pt")

        print("\n" + "=" * 80)
        print("🎉 训练完成！")
        print(f"📁 最佳权重：{weights_dir / 'best.pt'}（最佳mAP：{best_map:.4f}）")
        print(f"📁 最新权重：{weights_dir / 'last.pt'}")
        print("=" * 80)
        return model

    def collate_fn(self, batch):
        """处理不同样本目标数量不一致的问题（DataLoader必需）"""
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)  # 图像堆叠（batch, 3, 832, 1472）
        return images, targets  # 目标保持列表格式（每个元素是不同长度的tensor）

    def calculate_simple_map(self, all_preds, all_targets, iou_threshold=0.5):
        """简化版mAP计算（适用于快速评估，非COCO标准）"""
        if len(all_preds) == 0 or len(all_targets) == 0:
            return 0.0
        
        avg_ap = 0.0
        for cls in range(NUM_CLASSES):
            # 收集当前类别的所有预测和真实目标
            pred_boxes = []
            target_boxes = []
            for i in range(len(all_preds)):
                # 预测框：筛选当前类别+按置信度降序
                preds = [p for p in all_preds[i] if int(p[5]) == cls]
                preds = sorted(preds, key=lambda x: x[4], reverse=True)
                pred_boxes.extend(preds)
                
                # 真实目标：筛选当前类别
                targets = [t for t in all_targets[i] if int(t[4]) == cls]
                target_boxes.extend(targets)
            
            if len(pred_boxes) == 0 or len(target_boxes) == 0:
                ap = 0.0
                avg_ap += ap
                continue
            
            # 计算TP（真正例）、FP（假正例）
            tp = [0] * len(pred_boxes)
            target_used = [False] * len(target_boxes)
            
            for pred_idx, pred in enumerate(pred_boxes):
                pred_bbox = pred[:4]
                max_iou = 0.0
                best_target_idx = -1
                
                for target_idx, target in enumerate(target_boxes):
                    if target_used[target_idx]:
                        continue
                    target_bbox = target[:4]
                    iou = calculate_iou(pred_bbox, target_bbox)
                    if iou > max_iou:
                        max_iou = iou
                        best_target_idx = target_idx
                
                if max_iou >= iou_threshold and best_target_idx != -1:
                    tp[pred_idx] = 1
                    target_used[best_target_idx] = True
            
            # 计算Precision和Recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum([1 - x for x in tp])
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)  # 加1e-6避免除0
            recall = tp_cumsum / len(target_boxes)
            
            # 计算AP（平均精度）：梯形积分
            ap = 0.0
            prev_recall = 0.0
            for p, r in zip(precision, recall):
                if r > prev_recall:
                    ap += p * (r - prev_recall)
                    prev_recall = r
            
            avg_ap += ap
        
        # 平均AP（mAP）= 所有类别的AP平均值
        return avg_ap / NUM_CLASSES

    def predict_image(self, model, image_path):
        """单图推理（输出YOLO格式结果）"""
        # 图像预处理
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 无法加载图像：{image_path}")
            return [], image
        
        h_origin, w_origin = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
        
        # 转换为模型输入格式
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image_resized).unsqueeze(0).to(DEVICE)
        
        # 推理
        model.eval()
        with torch.no_grad():
            processed_outputs, _ = model(input_tensor)  # 使用后处理输出用于推理
            outputs = processed_outputs[0].cpu().numpy()
        
        # 后处理：过滤低置信度+NMS+坐标还原
        valid_boxes = outputs[outputs[:, 4] >= CONF_THRESHOLD]
        valid_boxes = nms_boxes(valid_boxes, IOU_THRESHOLD)
        
        # 还原到原始图像尺寸
        scale_x = w_origin / IMG_SIZE[1]
        scale_y = h_origin / IMG_SIZE[0]
        final_boxes = []
        for box in valid_boxes:
            x1 = box[0] * scale_x
            y1 = box[1] * scale_y
            x2 = box[2] * scale_x
            y2 = box[3] * scale_y
            conf = box[4]
            cls_id = int(box[5])
            
            # 修正坐标（避免超出图像范围）
            x1 = max(0, min(x1, w_origin))
            y1 = max(0, min(y1, h_origin))
            x2 = max(0, min(x2, w_origin))
            y2 = max(0, min(y2, h_origin))
            
            final_boxes.append([x1, y1, x2, y2, conf, cls_id])
        
        return final_boxes, image

    def export_predictions(self, model_path):
        """导出验证集预测结果（供vote_config.py使用）"""
        # 加载模型
        model = ViTDetector().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"\n✅ 加载模型：{model_path}")
        
        # 准备输出目录
        pred_dir = Path("./Pred")
        pred_labels_dir = pred_dir / "labels"
        pred_images_dir = pred_dir / "images"
        pred_labels_dir.mkdir(parents=True, exist_ok=True)
        pred_images_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载验证集图像
        image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_paths.extend(list(Path(VAL_IMAGES_PATH).glob(f'*{ext}')))
            image_paths.extend(list(Path(VAL_IMAGES_PATH).glob(f'*{ext.upper()}')))
        print(f"📊 找到验证集图像：{len(image_paths)}张")
        
        # 批量推理并保存结果
        total_boxes = 0
        progress_bar = tqdm(image_paths, desc="导出预测结果")
        for img_path in progress_bar:
            img_name = img_path.name
            # 推理
            boxes, image = self.predict_image(model, img_path)
            if len(boxes) == 0:
                continue
            
            # 保存图像
            shutil.copy2(str(img_path), str(pred_images_dir / img_name))
            
            # 保存标签（YOLO格式：cls_id cx cy bw bh conf）
            label_name = img_path.stem + '.txt'
            label_path = pred_labels_dir / label_name
            h_origin, w_origin = image.shape[:2]
            
            with open(label_path, 'w') as f:
                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box
                    # 转换为YOLO相对坐标
                    cx = ((x1 + x2) / 2) / w_origin
                    cy = ((y1 + y2) / 2) / h_origin
                    bw = (x2 - x1) / w_origin
                    bh = (y2 - y1) / h_origin
                    # 写入格式：cls_id cx cy bw bh conf
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}\n")
                    total_boxes += 1
        
        # 创建data.yaml
        data_yaml_path = pred_dir / "data.yaml"
        data_config = {
            'train': '',
            'val': str(VAL_IMAGES_PATH),
            'test': '',
            'nc': NUM_CLASSES,
            'names': self.class_names,
            'val_labels': str(VAL_LABELS_PATH)
        }
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"🎉 预测结果导出完成：{pred_dir}")
        print(f"   共生成预测框：{total_boxes}个 | 数据配置：{data_yaml_path}")
        return pred_dir


# ============================== 主函数（一键运行）==============================
def main():
    # 初始化训练器
    trainer = ViTDetectorTrainer()
    
    # 训练模型
    trainer.train_model()
    
    # 导出预测结果（若最佳模型存在）
    if Path(BEST_MODEL_PATH).exists():
        trainer.export_predictions(BEST_MODEL_PATH)
    else:
        print(f"⚠️ 最佳模型不存在：{BEST_MODEL_PATH}，跳过预测导出")


if __name__ == '__main__':
    main()