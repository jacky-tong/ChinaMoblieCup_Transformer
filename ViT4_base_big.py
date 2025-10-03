import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vit_l_16, ViT_L_16_Weights
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import psutil

# ============================ 1. 全局配置 ============================
DATASET_ROOT = r"/root/lowTrain/dataset"
TRAIN_OUTPUT_DIR = "./runs/vit_large_detector_832x1472"
MODEL_NAME = "vit_large_832x1472_detector"
BEST_MODEL_PATH = os.path.join(TRAIN_OUTPUT_DIR, MODEL_NAME, "weights", "best.pt")
LAST_MODEL_PATH = os.path.join(TRAIN_OUTPUT_DIR, MODEL_NAME, "weights", "last.pt")

IMG_SIZE = (832, 1472)
PATCH_SIZE = 16
NUM_CLASSES = 4
EMBED_DIM = 1024
NUM_PATCHES = (IMG_SIZE[0] // PATCH_SIZE) * (IMG_SIZE[1] // PATCH_SIZE)  # 4784

EPOCHS = 100
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
BASE_LR = 1e-5
WEIGHT_DECAY = 1e-4
FREEZE_LAYERS = 16
LOSS_LAMBDA_BOX = 5.0
LOSS_LAMBDA_CLS = 1.0
WARMUP_EPOCHS = 20  # 前 20 轮用单向匹配，之后用双向匹配

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备：{DEVICE}")


# ============================ 2. 数据集 ============================
class CustomDetectionDataset(Dataset):
    def __init__(self, img_dir, label_dir, is_train=True):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.img_paths = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG"]:
            self.img_paths.extend(list(self.img_dir.glob(f"*{ext}")))
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"❌ 在 {img_dir} 未找到图片")
        print(f"📊 加载{len(self.img_paths)}张{'训练' if is_train else '验证'}图")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"❌ 无法加载图像：{img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_SIZE[1], IMG_SIZE[0]))
        img_tensor = self.transform(img_resized)

        label_path = self.label_dir / (img_path.stem + ".txt")
        targets = []
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or len(line.split()) != 5:
                        continue
                    cls_id, cx_rel, cy_rel, bw_rel, bh_rel = map(float, line.split())
                    cx_abs = cx_rel * IMG_SIZE[1]
                    cy_abs = cy_rel * IMG_SIZE[0]
                    bw_abs = bw_rel * IMG_SIZE[1]
                    bh_abs = bh_rel * IMG_SIZE[0]
                    x1 = cx_abs - bw_abs / 2
                    y1 = cy_abs - bh_abs / 2
                    x2 = cx_abs + bw_abs / 2
                    y2 = cy_abs + bh_abs / 2
                    targets.append([x1, y1, x2, y2, cls_id])

        targets_tensor = torch.tensor(targets, dtype=torch.float32) if targets else torch.empty((0, 5), dtype=torch.float32)
        return img_tensor, targets_tensor

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


# ============================ 3. 模型 ============================
class ViTLargeBackbone(nn.Module):
    def __init__(self, freeze_layers=FREEZE_LAYERS):
        super().__init__()
        weights = ViT_L_16_Weights.DEFAULT
        vit = vit_l_16(weights=weights)
        self.conv_proj = vit.conv_proj
        self.transformer_layers = vit.encoder.layers

        for param in self.conv_proj.parameters():
            param.requires_grad = False
        for layer in self.transformer_layers[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        self.pos_embedding = nn.Parameter(torch.randn(1, NUM_PATCHES, EMBED_DIM))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embedding
        for layer in self.transformer_layers:
            x = layer(x)
        return x


class ViTDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ViTLargeBackbone(freeze_layers=FREEZE_LAYERS)
        self.detection_head = nn.Sequential(
            nn.Linear(EMBED_DIM, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 5 + NUM_CLASSES)
        )
        self._init_detection_head()

    def _init_detection_head(self):
        for m in self.detection_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        patch_features = self.backbone(x)
        raw_preds = self.detection_head(patch_features)
        batch_size = x.shape[0]
        processed_preds = []
        for i in range(batch_size):
            pred = raw_preds[i]
            bboxes = pred[:, :4].clone()
            x1 = bboxes[:, 0].clamp(0, IMG_SIZE[1])
            y1 = bboxes[:, 1].clamp(0, IMG_SIZE[0])
            x2 = bboxes[:, 2].clamp(0, IMG_SIZE[1])
            y2 = bboxes[:, 3].clamp(0, IMG_SIZE[0])
            conf = torch.sigmoid(pred[:, 4:5])
            cls_logits = pred[:, 5:]
            cls_probs = F.softmax(cls_logits, dim=-1)
            cls_scores, cls_ids = torch.max(cls_probs, dim=-1, keepdim=True)
            final_conf = conf * cls_scores
            pred_i = torch.cat([x1.unsqueeze(-1), y1.unsqueeze(-1), x2.unsqueeze(-1), y2.unsqueeze(-1),
                                final_conf, cls_ids.float()], dim=-1)
            processed_preds.append(pred_i)
        processed_preds = torch.stack(processed_preds, dim=0)
        return processed_preds, raw_preds


# ============================ 4. 自适应匹配损失函数 ============================
class DetectionLoss(nn.Module):
    def __init__(self, warmup_epochs=WARMUP_EPOCHS):
        super().__init__()
        self.box_loss = nn.SmoothL1Loss(reduction="sum")
        self.cls_loss = nn.CrossEntropyLoss(reduction="sum")
        self.conf_loss = nn.BCEWithLogitsLoss(reduction="sum")
        self.warmup_epochs = warmup_epochs

    def calculate_iou(self, box1, box2):
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

    def forward(self, raw_preds, targets, epoch):
        total_loss = 0.0
        batch_size = raw_preds.shape[0]

        for i in range(batch_size):
            pred = raw_preds[i]  # shape: (num_preds, 5 + num_classes)
            target = targets[i]  # shape: (num_targets, 5) or empty

            if target.numel() == 0:
                # 没有目标时，只计算置信度为 0 的损失
                conf_loss = self.conf_loss(pred[:, 4], torch.zeros(pred.shape[0], device=pred.device))
                total_loss += conf_loss
                continue

            num_preds = pred.shape[0]
            num_targets = target.shape[0]

            # 计算 IoU 矩阵（num_preds, num_targets）
            iou_matrix = torch.zeros((num_preds, num_targets), device=pred.device)
            # 使用 CPU numpy 计算 IoU（保持原风格），但只转换必要的小张量
            pred_boxes_cpu = pred[:, :4].detach().cpu().numpy()
            target_boxes_cpu = target[:, :4].detach().cpu().numpy()
            for p_idx in range(num_preds):
                for t_idx in range(num_targets):
                    iou_matrix[p_idx, t_idx] = self.calculate_iou(pred_boxes_cpu[p_idx], target_boxes_cpu[t_idx])

            # 匹配逻辑：返回成对索引 lists，保证 pred 与 target 一一对应（相同长度）
            matched_pred_indices = []
            matched_target_indices = []

            if epoch < self.warmup_epochs:
                # 单向最大匹配：每个 target 选取 IoU 最大的 pred（可能存在多个 target 指向同一个 pred）
                matched_pred_ids = torch.argmax(iou_matrix, dim=0)  # len = num_targets
                # matched_pred_ids[t] 是对应 target t 的 pred 索引
                matched_pred_indices = matched_pred_ids.tolist()
                matched_target_indices = list(range(num_targets))
            else:
                # 双向最大匹配：要求互为最优
                matched_pred_ids = torch.argmax(iou_matrix, dim=0)  # target -> pred
                matched_target_ids = torch.argmax(iou_matrix, dim=1)  # pred -> target
                for t_idx in range(num_targets):
                    p_idx = int(matched_pred_ids[t_idx].item())
                    if int(matched_target_ids[p_idx].item()) == t_idx:
                        matched_pred_indices.append(p_idx)
                        matched_target_indices.append(t_idx)

                if len(matched_pred_indices) != num_targets:
                    # 仅作提示，实际使用的是互为最优的对子
                    print(f"警告：样本 {i} 互匹配对数 {len(matched_pred_indices)} 与目标数 {num_targets} 不一致")

            # 如果没有任何匹配对，跳过 box/cls 损失（只计算置信度损失）
            if len(matched_pred_indices) == 0:
                conf_target = torch.zeros(num_preds, device=pred.device)
                conf_loss = self.conf_loss(pred[:, 4], conf_target)
                total_loss += conf_loss
                continue

            # 按配对索引取出对应的预测和目标（保证形状一致）
            pred_pos_boxes = pred[matched_pred_indices, :4]
            target_pos_boxes = target[matched_target_indices, :4].to(pred.device)

            # 框损失
            box_loss = self.box_loss(pred_pos_boxes, target_pos_boxes)

            # 类别损失：CrossEntropy requires (N, C) and (N,)
            pred_pos_cls = pred[matched_pred_indices, 5:].to(pred.device)  # (M, num_classes)
            target_pos_cls = target[matched_target_indices, 4].long().to(pred.device)  # (M,)

            cls_loss = self.cls_loss(pred_pos_cls, target_pos_cls)

            # 置信度损失：正样本为 1，其它为 0
            conf_target = torch.zeros(num_preds, device=pred.device)
            for p_idx in matched_pred_indices:
                conf_target[p_idx] = 1.0
            conf_loss = self.conf_loss(pred[:, 4], conf_target)

            total_loss += LOSS_LAMBDA_BOX * box_loss + LOSS_LAMBDA_CLS * cls_loss + conf_loss

        return total_loss / batch_size


# ============================ 5. 训练函数 ============================
def train():
    os.makedirs(os.path.join(TRAIN_OUTPUT_DIR, MODEL_NAME, "weights"), exist_ok=True)

    train_dataset = CustomDetectionDataset(
        img_dir=os.path.join(DATASET_ROOT, "images", "train"),
        label_dir=os.path.join(DATASET_ROOT, "labels", "train"),
        is_train=True
    )
    val_dataset = CustomDetectionDataset(
        img_dir=os.path.join(DATASET_ROOT, "images", "val"),
        label_dir=os.path.join(DATASET_ROOT, "labels", "val"),
        is_train=False
    )

    num_workers = min(psutil.cpu_count() // 2, 8)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = ViTDetector().to(DEVICE)
    print(f"✅ ViT-Large/16模型初始化完成（参数总数：{sum(p.numel() for p in model.parameters())/1e6:.1f}M）")

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.transformer_layers[FREEZE_LAYERS:].parameters(), "lr": BASE_LR * 0.1, "weight_decay": WEIGHT_DECAY},
        {"params": model.backbone.pos_embedding, "lr": BASE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": model.detection_head.parameters(), "lr": BASE_LR, "weight_decay": WEIGHT_DECAY}
    ])

    criterion = DetectionLoss(warmup_epochs=WARMUP_EPOCHS).to(DEVICE)
    scaler = GradScaler(device=DEVICE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=BASE_LR * 0.01)

    best_val_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        train_total_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [训练]")
        optimizer.zero_grad()

        for step, (images, targets) in enumerate(train_pbar):
            images = images.to(DEVICE, non_blocking=True)
            targets = [t.to(DEVICE, non_blocking=True) for t in targets]

            with autocast(device_type=DEVICE.type):
                _, raw_preds = model(images)
                loss = criterion(raw_preds, targets, epoch) / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            train_pbar.set_postfix({"train_loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"})

        avg_train_loss = train_total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{EPOCHS} | 训练损失：{avg_train_loss:.4f} | 学习率：{current_lr:.6f}")

        model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [验证]"):
                images = images.to(DEVICE, non_blocking=True)
                targets = [t.to(DEVICE, non_blocking=True) for t in targets]
                with autocast(device_type=DEVICE.type):
                    _, raw_preds = model(images)
                    loss = criterion(raw_preds, targets, epoch)
                val_total_loss += loss.item()

        avg_val_loss = val_total_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | 验证损失：{avg_val_loss:.4f}")

        torch.save(model.state_dict(), LAST_MODEL_PATH)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✅ 最优模型更新！（验证损失：{best_val_loss:.4f}，保存路径：{BEST_MODEL_PATH}）")

        scheduler.step()

    print(f"\n🎉 训练全部完成！")
    print(f"📌 最优模型路径：{BEST_MODEL_PATH}（验证损失：{best_val_loss:.4f}）")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("⚠️ 警告：未检测到GPU！")
        input("按Enter继续（不推荐），或Ctrl+C终止：")
    train()