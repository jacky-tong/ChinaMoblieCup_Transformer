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
from torch.amp import autocast, GradScaler  # 修复API：改用torch.amp
from torch.utils.checkpoint import checkpoint  # 梯度检查点

# ============================== 全局配置（保持模型维度不变）==============================
DATASET_ROOT = r"E:\2025CompetitionLog\9.9\tRVALCUP_2\ChinaMoblieCup\dataset" 
TRAIN_OUTPUT_DIR = "./runs/vit_detect_optimized"
MODEL_NAME = "vit_detector_4090_optimized"
BEST_MODEL_PATH = os.path.join(TRAIN_OUTPUT_DIR, MODEL_NAME, "weights", "best.pt")
VAL_IMAGES_PATH = os.path.join(DATASET_ROOT, "images", "val")
VAL_LABELS_PATH = os.path.join(DATASET_ROOT, "labels", "val")

# 模型超参数（保持原有维度，修复窗口大小）
IMG_SIZE = (832, 1472)
PATCH_SIZE = 32
DIM = 512
DEPTH = 4
HEADS = 8
MLP_DIM = 1024
NUM_CLASSES = 4
WINDOW_SIZE = 10  # 修复：10×10=100，1196（总补丁数）可适配窗口划分

# 训练超参数（4090优化版）
EPOCHS = 200
BATCH_SIZE = 2               
GRADIENT_ACCUMULATION_STEPS = 8  # 等效 batch_size=16
BASE_LR = 5e-5
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3
LOSS_LAMBDA_BOX = 5.0
LOSS_LAMBDA_CLS = 1.0

# 推理超参数
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0

# ============================== 工具函数 ==============================
def nms_boxes(boxes, iou_threshold):
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep_boxes = []
    while boxes:
        current_box = boxes.pop(0)
        keep_boxes.append(current_box)
        boxes = [
            box for box in boxes
            if calculate_iou(np.array(current_box[:4]), np.array(box[:4])) < iou_threshold
        ]
    return keep_boxes

def calculate_iou(box1, box2):
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

# ============================== 窗口注意力模块（修复尺寸匹配）==============================
class WindowAttention(nn.Module):
    def __init__(self, dim, heads, window_size):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.window_size = window_size
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        ws2 = self.window_size ** 2
        # 计算所需补零（更稳健的取模方式，结果在[0, ws2-1]）
        pad_num = (ws2 - (N % ws2)) % ws2
        if pad_num > 0:
            x = F.pad(x, (0, 0, 0, pad_num))  # (B, N+pad, C)
        N_pad = x.shape[1]
        # 补零后重新计算窗口数，确保 rearrange 时一致
        num_windows = N_pad // ws2

        # 窗口注意力计算
        x = rearrange(x, 'b (nw ws2) c -> b nw ws2 c', nw=num_windows, ws2=ws2)
        qkv = self.qkv(x).reshape(B, num_windows, ws2, 3, self.heads, C // self.heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(3, 4).reshape(B, num_windows, ws2, C)
        x = rearrange(x, 'b nw ws2 c -> b (nw ws2) c')

        # 裁剪回原始长度（去除补零）
        if pad_num > 0:
            x = x[:, :N, :]

        x = self.proj(x)
        return x

# ============================== 模型定义（修复梯度检查点参数）==============================
class ViTBackbone(nn.Module):
    def __init__(self, img_size=IMG_SIZE, patch_size=PATCH_SIZE, dim=DIM, depth=DEPTH, heads=HEADS, mlp_dim=MLP_DIM):
        super().__init__()
        image_height, image_width = img_size
        patch_height, patch_width = patch_size, patch_size
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)  # 26×46=1196
        self.patch_dim = 3 * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))  # +1 for cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # 窗口注意力层（使用修复后的WindowAttention）
        self.transformer_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                WindowAttention(dim, heads, WINDOW_SIZE),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, dim)
                )
            )
            for _ in range(depth)
        ])

    def forward(self, img):
        batch_size = img.shape[0]
        # 1. 补丁嵌入
        x = self.to_patch_embedding(img)  # (B, 1196, 512)
        # 2. 添加cls_token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)  # (B, 1, 512)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1197, 512)
        # 3. 添加位置编码
        x = x + self.pos_embedding  # (B, 1197, 512)

        # 4. Transformer编码：对patch tokens（不含cls）逐层应用窗口注意力（保持cls不变）
        cls_token = x[:, :1, :]           # (B,1,dim)
        patch_tokens = x[:, 1:, :]        # (B, num_patches, dim)

        for blk in self.transformer_layers:
            # 只对 patch_tokens 应用块（blk 包含 LayerNorm, WindowAttention, ...）
            patch_tokens = checkpoint(blk, patch_tokens, use_reentrant=False)

        # 拼回 cls_token 与 patch_tokens
        x = torch.cat([cls_token, patch_tokens], dim=1)  # (B, 1197, dim)
        return x

class ViTDetector(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = ViTBackbone()
        self.detection_head = nn.Sequential(
            nn.Linear(DIM, DIM // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(DIM // 2, 5 + num_classes)  # 4坐标+1置信度+4类别
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 1. 提取ViT特征
        features = self.backbone(x)  # (B, 1197, 512)
        patch_features = features[:, 1:]  # 去除cls_token：(B, 1196, 512)
        # 2. 检测头预测
        detections = self.detection_head(patch_features)  # (B, 1196, 9)

        # 3. 坐标转换（绝对像素坐标）
        h, w = IMG_SIZE
        bboxes = detections[..., :4].clone()
        # x_center (相对→绝对)
        bboxes[..., 0] = (bboxes[..., 0] * 2 - 1) * w / 2
        # y_center (相对→绝对)
        bboxes[..., 1] = (bboxes[..., 1] * 2 - 1) * h / 2
        # width (指数还原→绝对)
        bboxes[..., 2] = torch.exp(bboxes[..., 2]) * (w / (w // PATCH_SIZE))
        # height (指数还原→绝对)
        bboxes[..., 3] = torch.exp(bboxes[..., 3]) * (h / (h // PATCH_SIZE))

        # 转换为x1,y1,x2,y2格式
        x1 = bboxes[..., 0] - bboxes[..., 2] / 2
        y1 = bboxes[..., 1] - bboxes[..., 3] / 2
        x2 = bboxes[..., 0] + bboxes[..., 2] / 2
        y2 = bboxes[..., 1] + bboxes[..., 3] / 2

        # 4. 置信度与类别处理
        class_logits = detections[..., 5:]  # (B, 1196, 4)
        confidences = torch.sigmoid(detections[..., 4:5])  # 置信度（0-1）
        class_scores = torch.max(F.softmax(class_logits, dim=-1), dim=-1, keepdim=True).values  # 类别最高分
        final_confidence = confidences * class_scores  # 最终置信度（置信度×类别分数）
        class_ids = torch.argmax(F.softmax(class_logits, dim=-1), dim=-1, keepdim=True)  # 类别ID

        # 5. 拼接最终输出
        processed_outputs = torch.cat([
            x1.unsqueeze(-1), y1.unsqueeze(-1), x2.unsqueeze(-1), y2.unsqueeze(-1),
            final_confidence, class_ids.float()
        ], dim=-1)  # (B, 1196, 6)：x1,y1,x2,y2,conf,cls_id

        return processed_outputs, detections

# ============================== 损失函数（补全完整逻辑）==============================
class DetectionLoss(nn.Module):
    def __init__(self, lambda_box=LOSS_LAMBDA_BOX, lambda_cls=LOSS_LAMBDA_CLS):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        # 基础损失函数
        self.bbox_loss = nn.SmoothL1Loss(reduction='sum')  # 边界框损失（对异常值鲁棒）
        self.class_loss = nn.CrossEntropyLoss(reduction='sum')  # 类别损失
        self.confidence_loss = nn.BCEWithLogitsLoss(reduction='sum')  # 置信度损失（二分类）

    def _get_img_wh(self):
        # 返回图像宽高（与模型输入一致）
        return IMG_SIZE[1], IMG_SIZE[0]  # (w, h)

    def forward(self, predictions, targets):
        """
        predictions: 检测头原始输出 (B, 1196, 9)
        targets: 真实标签列表 (B个元素，每个元素是(num_objects, 5)：x1,y1,x2,y2,cls_id)
        """
        total_loss = 0.0
        batch_size = predictions.shape[0]
        w_img, h_img = self._get_img_wh()

        for i in range(batch_size):
            pred = predictions[i]  # (1196, 9)
            target = targets[i]    # (num_objects, 5)

            # 情况1：当前样本无真实目标→仅优化置信度（希望置信度为0）
            if target.numel() == 0:
                conf_loss = self.confidence_loss(pred[:, 4], torch.zeros_like(pred[:, 4], device=pred.device))
                total_loss += conf_loss
                continue

            # 情况2：有真实目标→匹配预测框与真实框
            # 2.1 预测框坐标转换（原始参数→绝对坐标x1,y1,x2,y2）
            pred_bbox = pred[:, :4].clone()
            # x_center
            pred_bbox[:, 0] = (pred_bbox[:, 0] * 2 - 1) * w_img / 2
            # y_center
            pred_bbox[:, 1] = (pred_bbox[:, 1] * 2 - 1) * h_img / 2
            # width
            pred_bbox[:, 2] = torch.exp(pred_bbox[:, 2]) * (w_img / (w_img // PATCH_SIZE))
            # height
            pred_bbox[:, 3] = torch.exp(pred_bbox[:, 3]) * (h_img / (h_img // PATCH_SIZE))
            # 转换为x1,y1,x2,y2
            pred_x1 = pred_bbox[:, 0] - pred_bbox[:, 2] / 2
            pred_y1 = pred_bbox[:, 1] - pred_bbox[:, 3] / 2
            pred_x2 = pred_bbox[:, 0] + pred_bbox[:, 2] / 2
            pred_y2 = pred_bbox[:, 1] + pred_bbox[:, 3] / 2
            pred_boxes_abs = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)  # (1196, 4)

            # 2.2 计算IoU矩阵（预测框 vs 真实框）
            num_preds = pred_boxes_abs.shape[0]
            num_tgts = target.shape[0]
            ious = torch.zeros((num_preds, num_tgts), device=pred.device)
            for p_idx in range(num_preds):
                for t_idx in range(num_tgts):
                    # 计算单个IoU（避免类型不兼容）
                    p_box = pred_boxes_abs[p_idx].detach().cpu().numpy()
                    t_box = target[t_idx, :4].detach().cpu().numpy()
                    iou_val = calculate_iou(p_box, t_box)
                    ious[p_idx, t_idx] = torch.tensor(iou_val, device=pred.device)

            # 2.3 匹配策略：每个真实框匹配IoU最大的预测框（正样本）
            matched_pred_ids = torch.argmax(ious, dim=0)  # (num_tgts,)：每个真实框对应的预测框ID
            matched_tgt_ids = torch.arange(num_tgts, device=pred.device)  # (num_tgts,)

            # 2.4 生成正负样本掩码
            positive_mask = torch.zeros(num_preds, dtype=torch.bool, device=pred.device)
            positive_mask[matched_pred_ids] = True  # 正样本：匹配到真实框的预测框
            negative_mask = ~positive_mask  # 负样本：未匹配的预测框

            # 2.5 计算各部分损失
            # 边界框损失（仅正样本）
            pred_matched_boxes = pred_boxes_abs[matched_pred_ids]  # (num_tgts, 4)
            tgt_matched_boxes = target[matched_tgt_ids, :4]  # (num_tgts, 4)
            box_loss = self.bbox_loss(pred_matched_boxes, tgt_matched_boxes)

            # 类别损失（仅正样本）
            pred_matched_cls = pred[matched_pred_ids, 5:5+NUM_CLASSES]  # (num_tgts, 4)
            tgt_matched_cls = target[matched_tgt_ids, 4].long()  # (num_tgts,)
            cls_loss = self.class_loss(pred_matched_cls, tgt_matched_cls)

            # 置信度损失（正样本→1，负样本→0）
            conf_targets = torch.zeros(num_preds, device=pred.device)
            conf_targets[positive_mask] = 1.0
            conf_loss = self.confidence_loss(pred[:, 4], conf_targets)

            # 2.6 总损失（加权求和）
            total_loss += self.lambda_box * box_loss + self.lambda_cls * cls_loss + conf_loss

        # 平均到每个样本
        return total_loss / batch_size

# ============================== 数据集（修复图片格式与标签转换）==============================
class CustomDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=IMG_SIZE):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size  # (h, w)
        self.h_origin = None  # 原始图像高度
        self.w_origin = None  # 原始图像宽度

        # 数据预处理（与模型输入匹配）
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # HWC→CHW，像素归一化到[0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])

        # 修复：收集所有常见图片格式
        self.images = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
            self.images.extend(list(self.img_dir.glob(f'*{ext}')))
        if len(self.images) == 0:
            raise FileNotFoundError(f"在{img_dir}未找到图片文件（支持格式：jpg/jpeg/png/bmp）")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / (img_path.stem + '.txt')  # 标签文件与图片同名

        # 1. 加载并预处理图像
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"无法加载图像：{img_path}")
        self.h_origin, self.w_origin = image.shape[:2]  # 记录原始尺寸
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR→RGB（cv2默认BGR）
        image_resized = cv2.resize(image_rgb, (self.img_size[1], self.img_size[0]))  # 缩放到模型输入尺寸
        image_tensor = self.transform(image_resized)  # (3, 832, 1472)

        # 2. 加载并转换标签（YOLO格式→绝对坐标）
        targets = []
        if label_path.exists():
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        continue  # 跳过格式错误的行

                    # YOLO格式：cls_id, cx_rel, cy_rel, bw_rel, bh_rel（相对坐标）
                    cls_id, cx_rel, cy_rel, bw_rel, bh_rel = map(float, parts)

                    # 转换为原始图像的绝对坐标（x1,y1,x2,y2）
                    cx_abs = cx_rel * self.w_origin
                    cy_abs = cy_rel * self.h_origin
                    bw_abs = bw_rel * self.w_origin
                    bh_abs = bh_rel * self.h_origin
                    x1_abs = cx_abs - bw_abs / 2
                    y1_abs = cy_abs - bh_abs / 2
                    x2_abs = cx_abs + bw_abs / 2
                    y2_abs = cy_abs + bh_abs / 2

                    # 缩放到模型输入尺寸的绝对坐标
                    scale_x = self.img_size[1] / self.w_origin
                    scale_y = self.img_size[0] / self.h_origin
                    x1 = x1_abs * scale_x
                    y1 = y1_abs * scale_y
                    x2 = x2_abs * scale_x
                    y2 = y2_abs * scale_y

                    targets.append([x1, y1, x2, y2, cls_id])

        # 转换为Tensor（无目标时返回空Tensor）
        targets_tensor = torch.tensor(targets, dtype=torch.float32) if targets else torch.empty((0, 5), dtype=torch.float32)
        return image_tensor, targets_tensor

# ============================== 训练器（修复AMP与数据加载）==============================
class ViTDetectorTrainer:
    def __init__(self, dataset_root=DATASET_ROOT):
        self.dataset_root = Path(dataset_root)
        # 验证数据集目录
        self._validate_dataset()
        # 创建保存目录
        self.save_dir = Path(TRAIN_OUTPUT_DIR) / MODEL_NAME
        self.weights_dir = self.save_dir / 'weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def _validate_dataset(self):
        """验证数据集目录结构"""
        required_dirs = [
            self.dataset_root / 'images' / 'train',
            self.dataset_root / 'images' / 'val',
            self.dataset_root / 'labels' / 'train',
            self.dataset_root / 'labels' / 'val'
        ]
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"数据集目录不存在：{dir_path}")
        print(f"✅ 数据集目录验证通过")

    def collate_fn(self, batch):
        """DataLoader collation：处理不同样本目标数量不一致的问题"""
        images, targets = zip(*batch)
        # 图像堆叠（batch_size, 3, 832, 1472）
        images_tensor = torch.stack(images, dim=0)
        # 目标保持列表（每个元素是(num_objects, 5)的Tensor）
        return images_tensor, targets

    def train_model(self):
        # 1. 加载数据集
        train_dataset = CustomDetectionDataset(
            img_dir=self.dataset_root / 'images' / 'train',
            label_dir=self.dataset_root / 'labels' / 'train'
        )
        val_dataset = CustomDetectionDataset(
            img_dir=self.dataset_root / 'images' / 'val',
            label_dir=self.dataset_root / 'labels' / 'val'
        )

        # 修复：限制num_workers（避免CPU线程过多）
        num_workers = min(psutil.cpu_count() // 2, 8)  # 最多8个worker
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,  # 加速GPU数据传输
            drop_last=True  # 丢弃最后一个不完整批次
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        print(f"✅ 数据集加载完成（训练集：{len(train_dataset)}张，验证集：{len(val_dataset)}张）")

        # 2. 初始化模型、优化器、损失函数
        model = ViTDetector().to(DEVICE)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=BASE_LR,
            weight_decay=WEIGHT_DECAY
        )
        criterion = DetectionLoss().to(DEVICE)
        # 修复：AMP初始化（指定设备为'cuda'）
        scaler = GradScaler(device='cuda')
        # 学习率调度（余弦退火）
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=BASE_LR * 0.01  # 最小学习率
        )

        # 3. 训练循环
        best_val_loss = float('inf')
        for epoch in range(EPOCHS):
            # 训练阶段
            model.train()
            train_total_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [训练]")
            optimizer.zero_grad()  # 初始化梯度

            for step, (images, targets) in enumerate(train_pbar):
                images = images.to(DEVICE, non_blocking=True)  # 非阻塞传输
                targets = [t.to(DEVICE, non_blocking=True) for t in targets]

                # 混合精度训练
                with autocast(device_type='cuda'):  # 显式指定设备类型
                    _, raw_outputs = model(images)
                    loss = criterion(raw_outputs, targets) / GRADIENT_ACCUMULATION_STEPS  # 梯度累积：损失均分

                # 反向传播（缩放损失，避免梯度下溢）
                scaler.scale(loss).backward()

                # 梯度累积：达到步数后更新参数
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)  # 更新优化器
                    scaler.update()  # 更新缩放器
                    optimizer.zero_grad()  # 重置梯度

                # 记录损失
                train_total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                train_pbar.set_postfix({"train_loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})

            # 训练阶段统计
            avg_train_loss = train_total_loss / len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{EPOCHS} | 训练损失：{avg_train_loss:.4f} | 学习率：{current_lr:.6f}")

            # 验证阶段
            model.eval()
            val_total_loss = 0.0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [验证]")
            with torch.no_grad():
                for images, targets in val_pbar:
                    images = images.to(DEVICE, non_blocking=True)
                    targets = [t.to(DEVICE, non_blocking=True) for t in targets]

                    with autocast(device_type='cuda'):
                        _, raw_outputs = model(images)
                        loss = criterion(raw_outputs, targets)

                    val_total_loss += loss.item()
                    val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

            # 验证阶段统计
            avg_val_loss = val_total_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{EPOCHS} | 验证损失：{avg_val_loss:.4f}")

            # 保存模型
            # 3.1 保存最新模型
            torch.save(model.state_dict(), self.weights_dir / "last.pt")
            # 3.2 保存最优模型（基于验证损失）
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), self.weights_dir / "best.pt")
                print(f"✅ 最优模型更新（验证损失：{best_val_loss:.4f}）")
            # 3.3 每25轮保存中间模型
            if (epoch + 1) % 25 == 0:
                torch.save(model.state_dict(), self.weights_dir / f"epoch_{epoch+1}.pt")
                print(f"✅ 中间模型保存：epoch_{epoch+1}.pt")

            # 更新学习率
            scheduler.step()

        print(f"\n🎉 训练完成！")
        print(f"📁 最优模型：{self.weights_dir / 'best.pt'}（验证损失：{best_val_loss:.4f}）")
        print(f"📁 最新模型：{self.weights_dir / 'last.pt'}")

# ============================== 主函数 ==============================
def main():
    # 检查GPU
    if not torch.cuda.is_available():
        print("⚠️ 未检测到GPU，训练将非常缓慢！建议使用CUDA设备。")
    else:
        print(f"✅ 检测到GPU：{torch.cuda.get_device_name(0)}（显存：{GPU_MEMORY_GB:.1f}GB）")

    # 初始化训练器并开始训练
    trainer = ViTDetectorTrainer()
    trainer.train_model()

if __name__ == '__main__':
    main()