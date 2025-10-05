import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import psutil
from PIL import Image

# ================= 核心配置（重点！）=================
DATASET_ROOT = r"E:\2025CompetitionLog\9.9\tRVALCUP_2\ChinaMoblieCup\dataset" 
TRAIN_OUTPUT_DIR = "./runs/vit_detector"
MODEL_NAME = "vit_detector"
BEST_MODEL_PATH = os.path.join(TRAIN_OUTPUT_DIR, MODEL_NAME, "weights", "best.pt")

# 离线权重路径（必须正确）
LOCAL_VIT_WEIGHTS_PATH = r"E:\transformer\model--clip--vit\vit_b_16-c867db91.pth"

# ViT关键参数（和预训练对齐）
IMG_SIZE = (224, 224)
VIT_MEAN = [0.485, 0.456, 0.406]  # ViT预训练用的均值
VIT_STD = [0.229, 0.224, 0.225]   # ViT预训练用的标准差
NUM_CLASSES = 4

# 训练超参数（针对ViT优化）
EPOCHS = 200  # 先跑50轮看收敛
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2  # 减少累积，加快反馈
BASE_LR = 1e-5  # 降低学习率，保护预训练权重
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3  # 延长warmup，平稳过渡
FREEZE_LAYERS = 4  # 减少冻结层数，保留更多可训练参数
DROPOUT_RATE = 0.1

# 小目标配置
SMALL_OBJ_AREA_THR = 0.01  # 面积占比<1%为小目标
SMALL_OBJ_WEIGHT = 1.5     # 小目标损失权重

# 设备配置（主进程打印一次）
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if os.getpid() == 0 or psutil.Process().parent().pid == 0:  # 只在主进程打印
    print(f"✅ 主进程使用设备: {DEVICE}")

# ================= 数据集（修复预处理+标签归一化）=================
class CustomDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=IMG_SIZE, is_train=True):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.is_train = is_train
        
        # 修复：ViT专用预处理（对齐预训练分布）
        self.transform = self._get_transforms()
        
        # 加载图像列表（只在主进程打印数量）
        self.images = list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")) + \
                     list(self.img_dir.glob("*.JPG")) + list(self.img_dir.glob("*.PNG"))
        if os.getpid() == 0 or psutil.Process().parent().pid == 0:
            print(f"✅ 找到 {len(self.images)} 张图像")

    def _get_transforms(self):
        """修复：ViT专用数据增强+归一化，保证输出固定尺寸"""
        # 输出尺寸（PIL transforms 接受 (H, W)）
        out_size = (self.img_size[0], self.img_size[1])
        
        if self.is_train:
            # 训练：随机多尺度裁剪，输出固定尺寸
            aug_list = [
                transforms.RandomResizedCrop(size=out_size, scale=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        else:
            # 验证/测试：先缩放短边再中心裁剪，保证输出固定尺寸
            aug_list = [
                transforms.Resize(out_size),
                transforms.CenterCrop(out_size),
            ]

        # 最后转换为 Tensor 并对齐 ViT 预训练分布
        aug_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=VIT_MEAN, std=VIT_STD)
        ]
        return transforms.Compose(aug_list)

    def _small_obj_crop(self, img, targets):
        """小目标优先裁剪（增强小目标特征）"""
        h, w = img.shape[:2]
        total_area = h * w
        small_targets = [t for t in targets if (t[2]-t[0])*(t[3]-t[1])/total_area < SMALL_OBJ_AREA_THR]
        if not small_targets:
            return img, targets
        
        # 随机选一个小目标裁剪
        target = small_targets[np.random.randint(len(small_targets))]
        x1, y1, x2, y2 = target[:4]
        cx, cy = (x1+x2)/2, (y1+y2)/2
        crop_w = max(x2-x1, w*0.3)  # 最小裁剪30%宽度
        crop_h = max(y2-y1, h*0.3)
        crop_x1 = max(0, cx - crop_w/2)
        crop_y1 = max(0, cy - crop_h/2)
        crop_x2 = min(w, cx + crop_w/2)
        crop_y2 = min(h, cy + crop_h/2)
        
        # 裁剪图像+调整目标坐标
        img_cropped = img[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
        new_targets = []
        for t in targets:
            tx1, ty1, tx2, ty2, tcls = t
            ntx1 = max(0, tx1 - crop_x1)
            nty1 = max(0, ty1 - crop_y1)
            ntx2 = min(crop_x2 - crop_x1, tx2 - crop_x1)
            nty2 = min(crop_y2 - crop_y1, ty2 - crop_y1)
            if ntx1 < ntx2 and nty1 < nty2:
                new_targets.append([ntx1, nty1, ntx2, nty2, tcls])
        
        # 恢复原尺寸+调整坐标尺度
        img_resized = cv2.resize(img_cropped, (w, h))
        scale_x = w / (crop_x2 - crop_x1)
        scale_y = h / (crop_y2 - crop_y1)
        for t in new_targets:
            t[0] *= scale_x
            t[1] *= scale_y
            t[2] *= scale_x
            t[3] *= scale_y
        
        return img_resized, new_targets if new_targets else [[0,0,0,0,-1]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        # 加载图像
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转RGB（和ViT预训练对齐）
        h, w = img.shape[:2]

        # 加载标签（修复：归一化到0-1）
        targets = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id, cx, cy, bw, bh = map(float, parts)
                        # 修复：从相对坐标→绝对坐标→归一化到0-1
                        x1 = (cx - bw/2) * w / self.img_size[1]  # 除以img_size归一化
                        y1 = (cy - bh/2) * h / self.img_size[0]
                        x2 = (cx + bw/2) * w / self.img_size[1]
                        y2 = (cy + bh/2) * h / self.img_size[0]
                        #  clamp防止超界
                        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(1, x2), min(1, y2)
                        targets.append([x1, y1, x2, y2, cls_id])
        
        # 小目标增强（训练阶段）
        if self.is_train and np.random.random() < 0.5:
            img, targets = self._small_obj_crop(img, targets)
        
        # 预处理图像：确保传入 torchvision.transforms 的为 PIL.Image 或 Tensor
        if isinstance(img, np.ndarray):
            # 确保 uint8 类型以兼容 PIL
            img = img.astype('uint8')
            img_pil = Image.fromarray(img)
        else:
            img_pil = img
        img_tensor = self.transform(img_pil)
        
        # 处理空标签
        if not targets:
            targets = [[0, 0, 0, 0, -1]]
        
        return img_tensor, torch.tensor(targets, dtype=torch.float32)

def collate_fn(batch):
    """修复：避免目标填充导致的维度混乱"""
    images, targets = zip(*batch)
    images = torch.stack(images)
    max_targets = max(len(t) for t in targets) if targets else 1
    
    padded_targets = []
    for t in targets:
        if len(t) < max_targets:
            pad = torch.full((max_targets - len(t), 5), -1, dtype=torch.float32)
            t_padded = torch.cat([t, pad])
        else:
            t_padded = t[:max_targets]  # 防止个别样本目标过多
        padded_targets.append(t_padded)
    
    return images, torch.stack(padded_targets)

# ================= 模型（修复冻结层数+输入对齐）=================
class DetectionHead(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 4),
            nn.Sigmoid()  # 输出0-1，和标签对齐
        )
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes + 1)  # +1背景
        )
        self.conf_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        bbox_pred = self.bbox_head(x)
        cls_pred = self.cls_head(x)
        conf_pred = self.conf_head(x)
        return torch.cat([bbox_pred, conf_pred, cls_pred], dim=-1)

class ViTDetector(nn.Module):
    def __init__(self, freeze_layers=FREEZE_LAYERS, num_classes=NUM_CLASSES):
        super().__init__()
        
        # 第一步：先定义 img_size（关键！避免后续引用时未定义）
        self.img_size = IMG_SIZE  # 从全局配置获取图像尺寸
        
        # 离线加载ViT（修复：weights_only=True避免警告）
        if LOCAL_VIT_WEIGHTS_PATH and os.path.isfile(LOCAL_VIT_WEIGHTS_PATH):
            print(f"📥 加载本地ViT权重: {LOCAL_VIT_WEIGHTS_PATH}")
            self.backbone = vit_b_16(weights=None)
            # 添加weights_only=True避免FutureWarning
            state_dict = torch.load(LOCAL_VIT_WEIGHTS_PATH, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state_dict)
        else:
            print("⚠️ 本地权重不存在，在线下载ViT")
            self.backbone = vit_b_16(weights="IMAGENET1K_V1")
        
        # 第二步：计算num_patches（此时self.img_size已定义）
        ps = self.backbone.patch_size  # patch_size是整数（如16）
        self.embed_dim = self.backbone.hidden_dim
        self.num_patches = (self.img_size[0] // ps) * (self.img_size[1] // ps)
        
        # 冻结指定层数
        self._freeze_layers(freeze_layers)
        
        # 检测头
        self.detection_head = DetectionHead(self.embed_dim, num_classes)

    def _freeze_layers(self, freeze_layers):
        """冻结patch嵌入+前N层Transformer"""
        # 冻结patch嵌入层
        for param in self.backbone.conv_proj.parameters():
            param.requires_grad = False
        
        # 冻结前N层Transformer
        for i, layer in enumerate(self.backbone.encoder.layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        print(f"✅ 冻结前 {freeze_layers} 层Transformer，剩余 {12-freeze_layers} 层可训练")

    def forward(self, x):
        # ViT前向传播（严格对齐官方实现）
        x = self.backbone._process_input(x)
        n = x.shape[0]
        
        # 添加class token和位置编码
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.backbone.encoder.pos_embedding
        
        # Transformer编码
        x = self.backbone.encoder(x)
        
        # 取patch token（排除class token）
        patch_tokens = x[:, 1:]  # [B, num_patches, embed_dim]
        
        # 检测头预测
        return self.detection_head(patch_tokens)

# ================= 损失函数（修复尺度匹配+小目标加权）=================
class DetectionLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.bbox_loss = nn.SmoothL1Loss(reduction='none')  # 逐元素计算，方便加权
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, preds, targets):
        """
        preds: [B, N_patches, 5 + num_classes] → [bbox(4), conf(1), cls(N+1)]
        targets: [B, max_objects, 5] → [x1,y1,x2,y2,cls_id]（0-1归一化，-1为填充）
        """
        B, N_patches, _ = preds.shape
        total_loss = 0.0

        for b in range(B):
            pred = preds[b]  # [N_patches, 5+C]
            target = targets[b]  # [max_objects, 5]
            target = target[target[:, 4] != -1]  # 过滤填充标签
            
            # 情况1：无有效目标（只算背景置信度损失）
            if target.size(0) == 0:
                conf_loss = F.binary_cross_entropy(pred[:, 4], torch.zeros(N_patches, device=pred.device))
                total_loss += conf_loss
                continue

            # 情况2：有有效目标（计算匹配损失）
            pred_boxes = pred[:, :4]  # [N_patches, 4]
            target_boxes = target[:, :4]  # [N_targets, 4]
            
            # 计算IoU（修复：确保维度正确）
            ious = self._calc_iou(pred_boxes.unsqueeze(1), target_boxes.unsqueeze(0))  # [N_patches, N_targets]
            best_ious, best_target_idx = ious.max(dim=1)  # 每个patch匹配最佳目标

            # 修复：小目标匹配阈值降低（0.4），提高小目标召回
            matched_mask = best_ious > 0.4
            if matched_mask.sum() == 0:  # 无匹配时，取IoU最大的patch
                matched_mask = best_ious == best_ious.max()

            # 1. 边界框损失（修复：小目标加权）
            matched_pred_boxes = pred_boxes[matched_mask]
            matched_target_boxes = target_boxes[best_target_idx[matched_mask]]
            
            # 小目标加权：面积越小，权重越高
            target_areas = (matched_target_boxes[:, 2] - matched_target_boxes[:, 0]) * \
                           (matched_target_boxes[:, 3] - matched_target_boxes[:, 1])
            bbox_weights = torch.where(target_areas < SMALL_OBJ_AREA_THR, 
                                      torch.tensor(SMALL_OBJ_WEIGHT, device=pred.device),
                                      torch.tensor(1.0, device=pred.device))
            bbox_loss = (self.bbox_loss(matched_pred_boxes, matched_target_boxes) * 
                         bbox_weights.unsqueeze(1)).mean()

            # 2. 分类损失
            matched_pred_cls = pred[matched_mask, 6:]  # 跳过bbox(4)+conf(1)+背景(1)
            matched_target_cls = target[best_target_idx[matched_mask], 4].long()
            cls_loss = self.cls_loss(matched_pred_cls, matched_target_cls).mean()

            # 3. 置信度损失（匹配目标→1，未匹配→0）
            matched_conf_loss = F.binary_cross_entropy(pred[matched_mask, 4], 
                                                      torch.ones(matched_mask.sum(), device=pred.device))
            unmatched_mask = ~matched_mask
            if unmatched_mask.sum() > 0:
                unmatched_conf_loss = F.binary_cross_entropy(pred[unmatched_mask, 4], 
                                                            torch.zeros(unmatched_mask.sum(), device=pred.device))
                conf_loss = (matched_conf_loss + unmatched_conf_loss) / 2
            else:
                conf_loss = matched_conf_loss

            # 总损失（权重平衡）
            total_loss += bbox_loss * 5.0 + cls_loss * 1.0 + conf_loss * 2.0  # bbox损失权重更高

        return total_loss / B  # 平均到每个batch

    def _calc_iou(self, boxes1, boxes2):
        """计算IoU（boxes1: [N,1,4], boxes2: [1,M,4]）"""
        x1 = torch.max(boxes1[:, :, 0], boxes2[:, :, 0])
        y1 = torch.max(boxes1[:, :, 1], boxes2[:, :, 1])
        x2 = torch.min(boxes1[:, :, 2], boxes2[:, :, 2])
        y2 = torch.min(boxes1[:, :, 3], boxes2[:, :, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
        area2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])
        
        union = area1 + area2 - intersection
        return intersection / (union + 1e-6)  # 避免除以0

# ================= 训练器（修复学习率+日志清晰）=================
class Trainer:
    def __init__(self):
        self.device = DEVICE
        self._setup_dirs()

    def _setup_dirs(self):
        """创建输出目录"""
        os.makedirs(os.path.join(TRAIN_OUTPUT_DIR, MODEL_NAME, "weights"), exist_ok=True)
        print(f"✅ 输出目录: {TRAIN_OUTPUT_DIR}")

    def setup_data(self):
        """修复：DataLoader参数优化（避免多worker重复打印）"""
        # 训练集
        train_dataset = CustomDetectionDataset(
            img_dir=os.path.join(DATASET_ROOT, "images", "train"),
            label_dir=os.path.join(DATASET_ROOT, "labels", "train"),
            is_train=True
        )
        # 验证集
        val_dataset = CustomDetectionDataset(
            img_dir=os.path.join(DATASET_ROOT, "images", "val"),
            label_dir=os.path.join(DATASET_ROOT, "labels", "val"),
            is_train=False
        )
        
        # 优化：num_workers=4（避免过多导致资源占用），pin_memory=True加速
        num_workers = min(psutil.cpu_count() // 4, 4)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
            persistent_workers=True  # 复用worker，减少重复初始化
        )
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE*2, shuffle=False,  # 验证集batch加倍
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
            persistent_workers=True
        )
        
        # 清晰打印数据信息
        print(f"📊 数据加载完成:")
        print(f"  - 训练集: {len(train_dataset)} 张图, {len(train_loader)} 批次")
        print(f"  - 验证集: {len(val_dataset)} 张图, {len(val_loader)} 批次")
        print(f"  - 数据worker: {num_workers} 个")
        return train_loader, val_loader

    def setup_model(self):
        """修复：学习率调度+模型参数统计"""
        # 初始化模型
        model = ViTDetector().to(self.device)
        
        # 优化器（只训练可学习参数）
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(
            trainable_params, lr=BASE_LR, weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999)  # 稳定优化
        )
        
        # 学习率调度（修复：warmup后余弦退火）
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=BASE_LR * 0.01  # 最小学习率
        )
        
        # 损失函数
        criterion = DetectionLoss(NUM_CLASSES).to(self.device)
        
        # 统计可训练参数
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_num = sum(p.numel() for p in model.parameters())
        print(f"📈 模型参数: 总参数 {total_num//1000}K, 可训练参数 {trainable_num//1000}K")
        
        return model, optimizer, scheduler, criterion

    def _get_current_lr(self, optimizer, epoch):
        """修复：warmup学习率（从1e-7线性升到BASE_LR）"""
        if epoch < WARMUP_EPOCHS:
            return BASE_LR * (epoch + 1) / WARMUP_EPOCHS + 1e-7
        return optimizer.param_groups[0]['lr']

    def train_epoch(self, model, train_loader, optimizer, criterion, epoch):
        """单轮训练（修复：梯度累积+清晰日志）"""
        model.train()
        total_loss = 0.0
        current_lr = self._get_current_lr(optimizer, epoch)
        
        # 设置当前学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # 进度条（显示学习率）
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.2e} | Train")
        
        for step, (images, targets) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # 前向传播
            preds = model(images)
            loss = criterion(preds, targets)
            
            # 梯度累积
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            # 累积到指定步数后更新
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # 统计损失
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # 更新进度条
            pbar.set_postfix({
                'Step Loss': f'{loss.item()*GRADIENT_ACCUMULATION_STEPS:.4f}',
                'Avg Loss': f'{total_loss/(step+1):.4f}'
            })
        
        # 处理剩余梯度
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        return total_loss / len(train_loader)

    def val_epoch(self, model, val_loader, criterion, epoch):
        """单轮验证（无梯度计算）"""
        model.eval()
        total_loss = 0.0
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} | Val")
        
        with torch.no_grad():
            for images, targets in pbar:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                preds = model(images)
                loss = criterion(preds, targets)
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({
                    'Val Step Loss': f'{loss.item():.4f}',
                    'Val Avg Loss': f'{total_loss/(pbar.n//BATCH_SIZE + 1):.4f}'
                })
        
        return total_loss / len(val_loader)

    def train(self):
        """主训练流程"""
        print("\n" + "="*60)
        print("🚀 开始训练ViT目标检测器（小目标优化版）")
        print("="*60 + "\n")
        
        # 加载数据和模型
        train_loader, val_loader = self.setup_data()
        model, optimizer, scheduler, criterion = self.setup_model()
        
        # 训练记录
        best_val_loss = float('inf')
        
        for epoch in range(EPOCHS):
            print(f"\n📅 Epoch {epoch+1}/{EPOCHS}")
            print("-"*50)
            
            # 训练
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, epoch)
            
            # 验证
            val_loss = self.val_epoch(model, val_loader, criterion, epoch)
            
            # 学习率调度（warmup后生效）
            if epoch >= WARMUP_EPOCHS:
                scheduler.step()
            
            # 打印结果
            print(f"📊 Epoch {epoch+1} 结果:")
            print(f"  - 训练损失: {train_loss:.4f}")
            print(f"  - 验证损失: {val_loss:.4f}")
            print(f"  - 当前最佳验证损失: {best_val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, BEST_MODEL_PATH)
                print(f"💾 保存最佳模型到: {BEST_MODEL_PATH}")
            
            # 每10轮保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    TRAIN_OUTPUT_DIR, MODEL_NAME, "weights", f"checkpoint_epoch_{epoch+1}.pt"
                )
                torch.save(model.state_dict(), checkpoint_path)
                print(f"📁 保存检查点到: {checkpoint_path}")
        
        # 训练完成
        print("\n" + "="*60)
        print(f"🎉 训练完成！最佳验证损失: {best_val_loss:.4f}")
        print(f"📌 最佳模型路径: {BEST_MODEL_PATH}")
        print("="*60)

if __name__ == "__main__":
    # 检查数据集路径
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 数据集路径不存在: {DATASET_ROOT}")
        exit()
    # 检查权重路径
    if not os.path.exists(LOCAL_VIT_WEIGHTS_PATH):
        print(f"❌ ViT权重路径不存在: {LOCAL_VIT_WEIGHTS_PATH}")
        exit()
    # 开始训练
    trainer = Trainer()
    trainer.train()