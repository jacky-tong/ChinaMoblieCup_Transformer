import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import psutil
import math
from einops import rearrange
from contextlib import contextmanager

# ================= 配置 =================
DATASET_ROOT = r"E:\2025CompetitionLog\9.9\tRVALCUP_2\ChinaMoblieCup\dataset" 
TRAIN_OUTPUT_DIR = "./runs/vit_detector"
MODEL_NAME = "vit_detector"
BEST_MODEL_PATH = os.path.join(TRAIN_OUTPUT_DIR, MODEL_NAME, "weights", "best.pt")

IMG_SIZE = (224, 224)  # ViT-Base的标准输入尺寸
PATCH_SIZE = 16
NUM_CLASSES = 4

# 训练超参数（优化后的配置）
EPOCHS = 100
BATCH_SIZE = 8  # 增加批处理大小
GRADIENT_ACCUMULATION_STEPS = 4  # 减少梯度累积步数
BASE_LR = 2e-5
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
FREEZE_LAYERS = 8  # ViT-Base有12层，冻结前8层

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {DEVICE}")

# 兼容不同 PyTorch 版本的 GradScaler/ autocast 封装
def create_grad_scaler(device):
    """创建兼容不同版本的 GradScaler；如果非CUDA返回None"""
    if device.type != 'cuda':
        return None
    # 优先尝试无参数构造（常见版本），回退到 device_type 参数（部分版本）
    try:
        return GradScaler()
    except TypeError:
        try:
            return GradScaler(device_type='cuda')
        except TypeError:
            # 最后兜底再次尝试无参数构造
            return GradScaler()

@contextmanager
def autocast_context(enabled):
    """兼容不同版本 autocast 的上下文管理器（enabled: 是否启用）"""
    if not enabled:
        yield
        return
    try:
        # 新版 torch.amp.autocast 可能需要 device_type 参数
        with autocast(device_type='cuda'):
            yield
    except TypeError:
        # 低版本可能不接受 device_type
        with autocast():
            yield

# ================= 数据集 =================
class CustomDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=IMG_SIZE, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        
        # 数据增强
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # 获取图像文件列表
        self.images = list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")) + \
                     list(self.img_dir.glob("*.JPG")) + list(self.img_dir.glob("*.PNG"))
        
        print(f"✅ 找到 {len(self.images)} 张图像")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        # 加载图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ 无法读取图像: {img_path}")
            # 返回黑色图像作为占位符
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸
        h, w = img.shape[:2]
        img_resized = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        
        # 应用变换
        img_tensor = self.transform(img_resized)

        # 处理标签
        targets = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:  # 确保有5个值
                        cls_id, cx, cy, bw, bh = map(float, parts)
                        # 转换为绝对坐标
                        x1 = max(0, (cx - bw/2) * self.img_size[1])
                        y1 = max(0, (cy - bh/2) * self.img_size[0])
                        x2 = min(self.img_size[1], (cx + bw/2) * self.img_size[1])
                        y2 = min(self.img_size[0], (cy + bh/2) * self.img_size[0])
                        
                        targets.append([x1, y1, x2, y2, cls_id])
        
        # 如果没有目标，添加一个空目标
        if len(targets) == 0:
            targets.append([0, 0, 0, 0, -1])  # 使用-1表示无目标

        return img_tensor, torch.tensor(targets, dtype=torch.float32)

def collate_fn(batch):
    """自定义批处理函数"""
    images, targets = zip(*batch)
    images = torch.stack(images)
    
    # 找到批次中最多的目标数
    max_targets = max(len(t) for t in targets)
    
    # 填充目标张量
    padded_targets = []
    for t in targets:
        if len(t) < max_targets:
            # 使用-1填充
            pad = torch.full((max_targets - len(t), 5), -1, dtype=torch.float32)
            t_padded = torch.cat([t, pad])
        else:
            t_padded = t
        padded_targets.append(t_padded)
    
    return images, torch.stack(padded_targets)

# ================= 模型定义 =================
class DetectionHead(nn.Module):
    """目标检测头 - 预测边界框和类别"""
    def __init__(self, embed_dim, num_classes, num_patches):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        
        # 边界框预测 (x1, y1, x2, y2)
        self.bbox_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.Sigmoid()  # 输出在0-1范围内
        )
        
        # 类别预测
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes + 1)  # +1 for background/no object
        )
        
        # 目标置信度
        self.conf_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """x shape: (B, num_patches, embed_dim)"""
        B, N, D = x.shape
        
        # 为每个patch预测边界框、类别和置信度
        bbox_pred = self.bbox_head(x)  # (B, N, 4)
        cls_pred = self.cls_head(x)    # (B, N, num_classes+1)
        conf_pred = self.conf_head(x)  # (B, N, 1)
        
        # 合并所有预测
        outputs = torch.cat([bbox_pred, conf_pred, cls_pred], dim=-1)  # (B, N, 5 + num_classes)
        return outputs

class ViTDetector(nn.Module):
    """基于ViT的目标检测器"""
    def __init__(self, freeze_layers=FREEZE_LAYERS, num_classes=NUM_CLASSES):
        super().__init__()
        
        # 加载预训练的ViT-Base模型
        weights = ViT_B_16_Weights.DEFAULT
        self.backbone = vit_b_16(weights=weights)
        
        # 获取模型参数
        self.patch_size = self.backbone.patch_size  # 直接使用整数，不下标访问
        self.embed_dim = self.backbone.hidden_dim
        
        # 计算patch数量
        self.num_patches = (IMG_SIZE[0] // self.patch_size) * (IMG_SIZE[1] // self.patch_size)
        
        # 冻结指定层数
        self._freeze_layers(freeze_layers)
        
        # 检测头 - 为每个patch预测目标
        self.detection_head = DetectionHead(self.embed_dim, num_classes, self.num_patches)

    def _freeze_layers(self, freeze_layers):
        """冻结指定层数"""
        # 冻结patch嵌入
        for param in self.backbone.conv_proj.parameters():
            param.requires_grad = False
            
        # 冻结Transformer层
        for i, layer in enumerate(self.backbone.encoder.layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        
        print(f"✅ 冻结了前 {freeze_layers} 层Transformer")

    def forward(self, x):
        """前向传播"""
        B, C, H, W = x.shape
        
        # ViT前向传播
        x = self.backbone._process_input(x)  # Patch embedding
        n = x.shape[0]
        
        # 扩展class token和位置编码
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.backbone.encoder.pos_embedding
        
        # Transformer编码器
        x = self.backbone.encoder(x)
        
        # 使用所有patch tokens（排除class token）进行检测
        patch_tokens = x[:, 1:]  # (B, num_patches, embed_dim)
        
        # 检测头
        outputs = self.detection_head(patch_tokens)
        
        return outputs

# ================= 损失函数 =================
class DetectionLoss(nn.Module):
    """改进的目标检测损失函数"""
    def __init__(self, num_classes, num_patches):
        super().__init__()
        self.num_classes = num_classes
        self.num_patches = num_patches
        
        # 损失函数
        self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        self.conf_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        """
        preds: (B, num_patches, 5 + num_classes)
        targets: (B, max_targets, 5) - 最后一维是[x1, y1, x2, y2, class_id]
        """
        B, N, _ = preds.shape
        total_loss = 0
        
        for i in range(B):
            # 当前批次的预测和目标
            pred = preds[i]  # (N, 5 + num_classes)
            target = targets[i]  # (max_targets, 5)
            
            # 过滤有效目标（class_id >= 0）
            valid_targets = target[target[:, 4] >= 0]
            if len(valid_targets) == 0:
                # 如果没有有效目标，只计算背景损失
                conf_loss = F.binary_cross_entropy_with_logits(
                    pred[:, 4], torch.zeros(N, device=pred.device), reduction='mean'
                )
                total_loss += conf_loss
                continue
            
            # 简化的匹配策略：选择置信度最高的patch作为预测
            # 在实际应用中可以使用更复杂的匹配策略如匈牙利算法
            with torch.no_grad():
                # 计算每个patch与每个目标的IoU
                pred_boxes = pred[:, :4]
                target_boxes = valid_targets[:, :4]
                
                # 计算IoU
                ious = self._calculate_iou(pred_boxes.unsqueeze(1), target_boxes.unsqueeze(0))
                best_ious, best_target_idx = ious.max(dim=1)
                
                # 选择与目标有最大IoU的patch
                matched = best_ious > 0.5
                if matched.sum() == 0:
                    # 如果没有匹配，使用最高IoU的patch
                    matched = best_ious == best_ious.max()
            
            # 计算匹配目标的损失
            if matched.sum() > 0:
                matched_pred = pred[matched]
                matched_target_idx = best_target_idx[matched]
                matched_targets = valid_targets[matched_target_idx]
                
                # 边界框损失
                bbox_loss = self.bbox_loss(matched_pred[:, :4], matched_targets[:, :4]).mean()
                
                # 类别损失
                cls_loss = self.cls_loss(
                    matched_pred[:, 6:],  # 跳过bbox(4)和conf(1)、背景类(1)
                    matched_targets[:, 4].long()
                ).mean()
                
                # 置信度损失（匹配的目标置信度应为1）
                conf_loss = F.binary_cross_entropy_with_logits(
                    matched_pred[:, 4], torch.ones(matched.sum(), device=pred.device)
                )
                
                # 未匹配patch的置信度损失（应为0）
                unmatched = ~matched
                if unmatched.sum() > 0:
                    unmatched_conf_loss = F.binary_cross_entropy_with_logits(
                        pred[unmatched, 4], torch.zeros(unmatched.sum(), device=pred.device)
                    )
                    conf_loss = (conf_loss + unmatched_conf_loss) / 2
                
                total_loss += bbox_loss + cls_loss + conf_loss
        
        return total_loss / B

    def _calculate_iou(self, boxes1, boxes2):
        """计算IoU"""
        # boxes1: (N, 1, 4), boxes2: (1, M, 4)
        x1 = torch.max(boxes1[:, :, 0], boxes2[:, :, 0])
        y1 = torch.max(boxes1[:, :, 1], boxes2[:, :, 1])
        x2 = torch.min(boxes1[:, :, 2], boxes2[:, :, 2])
        y2 = torch.min(boxes1[:, :, 3], boxes2[:, :, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
        area2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])
        
        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-6)
        
        return iou

# ================= 训练器类 =================
class ViTTrainer:
    def __init__(self):
        self.device = DEVICE
        self.setup_directories()
        
    def setup_directories(self):
        """创建输出目录"""
        os.makedirs(os.path.join(TRAIN_OUTPUT_DIR, MODEL_NAME, "weights"), exist_ok=True)
        print(f"✅ 输出目录创建完成: {TRAIN_OUTPUT_DIR}")

    def setup_data_loaders(self):
        """设置数据加载器"""
        # 数据增强
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 数据集
        train_dataset = CustomDetectionDataset(
            img_dir=os.path.join(DATASET_ROOT, "images", "train"),
            label_dir=os.path.join(DATASET_ROOT, "labels", "train"),
            transform=train_transform
        )
        
        val_dataset = CustomDetectionDataset(
            img_dir=os.path.join(DATASET_ROOT, "images", "val"),
            label_dir=os.path.join(DATASET_ROOT, "labels", "val"),
            transform=val_transform
        )
        
        # 数据加载器
        num_workers = min(psutil.cpu_count() // 2, 8)
        
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )
        
        print(f"✅ 数据加载器创建完成")
        print(f"  训练集: {len(train_dataset)} 张图像, {len(train_loader)} 个批次")
        print(f"  验证集: {len(val_dataset)} 张图像, {len(val_loader)} 个批次")
        
        return train_loader, val_loader

    def setup_model_and_optimizer(self):
        """设置模型和优化器"""
        model = ViTDetector().to(self.device)
        
        # 优化器
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=BASE_LR, weight_decay=WEIGHT_DECAY
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        # 损失函数
        num_patches = (IMG_SIZE[0] // PATCH_SIZE) * (IMG_SIZE[1] // PATCH_SIZE)
        criterion = DetectionLoss(NUM_CLASSES, num_patches).to(self.device)
        
        # 梯度缩放（混合精度训练） - 使用兼容创建函数
        scaler = create_grad_scaler(self.device)
        
        return model, optimizer, scheduler, criterion, scaler

    def train_epoch(self, model, train_loader, optimizer, criterion, scaler, epoch):
        """训练一个epoch"""
        model.train()
        train_loss = 0
        accumulation_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [训练]")
        
        for i, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # 混合精度训练 - 使用兼容 autocast 上下文
            if scaler is not None:
                with autocast_context(True):
                    preds = model(images)
                    loss = criterion(preds, targets) / GRADIENT_ACCUMULATION_STEPS
                
                # 梯度累积
                scaler.scale(loss).backward()
                accumulation_steps += 1
                
                if accumulation_steps % GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # CPU训练
                preds = model(images)
                loss = criterion(preds, targets) / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                accumulation_steps += 1
                
                if accumulation_steps % GRADIENT_ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}',
                'Avg Loss': f'{train_loss / (i + 1):.4f}'
            })
        
        # 处理剩余的梯度
        if accumulation_steps % GRADIENT_ACCUMULATION_STEPS != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        return train_loss / len(train_loader)

    def validate_epoch(self, model, val_loader, criterion, epoch):
        """验证一个epoch"""
        model.eval()
        val_loss = 0
        
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [验证]")
        
        with torch.no_grad():
            for images, targets in progress_bar:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # 混合精度验证 - 使用兼容 autocast 上下文
                if self.device.type == 'cuda':
                    with autocast_context(True):
                        preds = model(images)
                        loss = criterion(preds, targets)
                else:
                    preds = model(images)
                    loss = criterion(preds, targets)
                
                val_loss += loss.item()
                
                progress_bar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Avg Val Loss': f'{val_loss / (progress_bar.n + 1):.4f}'
                })
        
        return val_loss / len(val_loader)

    def train_model(self):
        """主训练函数"""
        print("🚀 开始训练ViT目标检测器...")
        
        # 设置数据加载器
        train_loader, val_loader = self.setup_data_loaders()
        
        # 设置模型和优化器
        model, optimizer, scheduler, criterion, scaler = self.setup_model_and_optimizer()
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        for epoch in range(EPOCHS):
            print(f"\n{'='*50}")
            print(f"📅 Epoch {epoch+1}/{EPOCHS}")
            print(f"{'='*50}")
            
            # 训练
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
            train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate_epoch(model, val_loader, criterion, epoch)
            val_losses.append(val_loss)
            
            # 学习率调度
            scheduler.step()
            
            print(f"📈 训练损失: {train_loss:.4f} | 验证损失: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss
                }, BEST_MODEL_PATH)
                print(f"💾 保存最佳模型到: {BEST_MODEL_PATH}")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    TRAIN_OUTPUT_DIR, MODEL_NAME, "weights", f"checkpoint_epoch_{epoch+1}.pt"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                print(f"📁 保存检查点到: {checkpoint_path}")
        
        print(f"\n🎉 训练完成! 最佳验证损失: {best_val_loss:.4f}")
        
        # 保存训练记录
        training_log = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'epochs': EPOCHS
        }
        
        log_path = os.path.join(TRAIN_OUTPUT_DIR, MODEL_NAME, "training_log.npy")
        np.save(log_path, training_log)
        print(f"📊 训练记录保存到: {log_path}")

# ================= 主函数 =================
def main():
    # 检查数据集是否存在
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ 数据集目录不存在: {DATASET_ROOT}")
        return
    
    print("🔍 检查数据集结构...")
    train_img_dir = os.path.join(DATASET_ROOT, "images", "train")
    train_label_dir = os.path.join(DATASET_ROOT, "labels", "train")
    
    if not os.path.exists(train_img_dir) or not os.path.exists(train_label_dir):
        print("❌ 数据集结构不正确，请确保存在 images/train 和 labels/train 目录")
        return
    
    # 开始训练
    trainer = ViTTrainer()
    trainer.train_model()

if __name__ == "__main__":
    main()