import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# ===================== 路径配置（根据实际情况修改以下路径） =====================
# 真实标签目录（COCO格式TXT文件，5值：类别ID + 4个坐标）
GT_LABELS_DIR = r"E:\2025CompetitionLog\9.9\tRVALCUP_2\ChinaMoblieCup\dataset\labels\val"
# 预测标签目录（6值格式TXT文件：类别ID + 4个坐标 + 置信度）
PRED_LABELS_DIR = r'E:\2025CompetitionLog\9.9\Travel3_Transformer\ViT_Pred\labels'
# 图片目录（用于获取图片尺寸转换坐标）
IMAGES_DIR = r"E:\2025CompetitionLog\9.9\Travel3_Transformer\ViT_Pred\images"
# ==============================================================================

def load_labels(label_dir, image_dir, is_prediction=False):
    """
    加载标签文件并转换为绝对坐标
    :param is_prediction: 是否为预测标签（6值格式，含置信度）
    :return: 标签字典 {图片名: [(x1, y1, x2, y2, cls_id, [conf])]}
    """
    labels = {}
    label_files = list(Path(label_dir).glob("*.txt"))
    
    for label_file in label_files:
        img_name = label_file.stem
        img_path = os.path.join(image_dir, f"{img_name}.jpg")
        # 尝试其他图片格式
        if not os.path.exists(img_path):
            for ext in [".jpeg", ".png", ".bmp"]:
                img_path = os.path.join(image_dir, f"{img_name}{ext}")
                if os.path.exists(img_path):
                    break
        
        # 获取图片尺寸
        if not os.path.exists(img_path):
            print(f"警告：未找到图片 {img_name}，跳过该标签文件")
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # 读取标签内容
        with open(label_file, "r") as f:
            lines = f.readlines()
        
        boxes = []
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            
            # 验证标签格式
            if (not is_prediction and len(parts) != 5) or (is_prediction and len(parts) != 6):
                print(f"警告：{label_file} 第{line_idx+1}行格式错误（预期{6 if is_prediction else 5}个值），跳过该行")
                continue
            
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            
            # 转换归一化坐标到绝对坐标 (x1, y1, x2, y2)
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            
            # 确保坐标在有效范围内
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if is_prediction:
                # 预测标签：提取第六个值作为置信度
                confidence = float(parts[5])
                boxes.append((x1, y1, x2, y2, cls_id, confidence))
            else:
                # 真实标签不需要置信度
                boxes.append((x1, y1, x2, y2, cls_id))
        
        labels[img_name] = boxes
    
    return labels

def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1, y1, x2, y2 = box1[:4]
    x1g, y1g, x2g, y2g = box2[:4]
    
    # 计算交集
    x1i = max(x1, x1g)
    y1i = max(y1, y1g)
    x2i = min(x2, x2g)
    y2i = min(y2, y2g)
    
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    
    area_inter = (x2i - x1i) * (y2i - y1i)
    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x2g - x1g) * (y2g - y1g)
    area_union = area_box1 + area_box2 - area_inter
    
    return area_inter / area_union if area_union > 0 else 0.0

def compute_ap(precision, recall):
    """计算平均精度AP"""
    # 确保precision是单调递减的
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    
    # 计算不同recall点的精度
    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    ap = np.sum((recall[indices] - recall[indices-1]) * precision[indices])
    return ap

def evaluate_map(gt_labels, pred_labels, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    """计算mAP指标"""
    # 获取所有类别
    all_classes = set()
    for img_boxes in gt_labels.values():
        for box in img_boxes:
            all_classes.add(box[4])
    for img_boxes in pred_labels.values():
        for box in img_boxes:
            all_classes.add(box[4])
    all_classes = sorted(list(all_classes))
    if not all_classes:
        return {"error": "未找到任何类别"}
    
    # 按类别和IoU阈值计算AP
    ap_results = {cls: {iou: 0.0 for iou in iou_thresholds} for cls in all_classes}
    
    for cls in all_classes:
        # 收集所有预测框（按置信度排序）
        all_preds = []
        for img_name, boxes in pred_labels.items():
            for box in boxes:
                if box[4] == cls:
                    # 预测框格式：(x1, y1, x2, y2, cls_id, confidence)
                    all_preds.append((box[5], img_name, box[:4]))  # (置信度, 图片名, 框坐标)
        
        # 按置信度降序排序
        all_preds.sort(reverse=True, key=lambda x: x[0])
        
        # 收集所有真实框
        all_gts = {}
        for img_name, boxes in gt_labels.items():
            cls_gts = [box[:4] for box in boxes if box[4] == cls]
            if cls_gts:
                all_gts[img_name] = {
                    "boxes": cls_gts,
                    "used": [False] * len(cls_gts)  # 标记是否已匹配
                }
        
        # 计算每个IoU阈值下的AP
        for iou_thresh in iou_thresholds:
            tp = []  # 真正例
            fp = []  # 假正例
            num_gt = sum(len(gts["boxes"]) for gts in all_gts.values())  # 真实框总数
            
            if num_gt == 0:
                ap_results[cls][iou_thresh] = 0.0
                continue
            
            for conf, img_name, pred_box in all_preds:
                if img_name not in all_gts:
                    fp.append(1)
                    tp.append(0)
                    continue
                
                gts = all_gts[img_name]
                best_iou = 0.0
                best_idx = -1
                
                # 寻找最佳匹配的真实框
                for idx, gt_box in enumerate(gts["boxes"]):
                    if not gts["used"][idx]:
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = idx
                
                # 判断是否为真正例
                if best_iou >= iou_thresh and best_idx != -1:
                    gts["used"][best_idx] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            
            # 计算精度和召回率
            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            precision = tp / (tp + fp + 1e-10)
            recall = tp / num_gt
            
            # 计算AP
            ap = compute_ap(precision, recall)
            ap_results[cls][iou_thresh] = ap
    
    # 计算mAP
    map_results = {}
    for iou in iou_thresholds:
        class_aps = [ap_results[cls][iou] for cls in all_classes]
        map_results[f"mAP@{iou:.2f}"] = np.mean(class_aps)
    
    # 计算mAP50和mAP50-95
    map50 = map_results.get("mAP@0.50", 0.0)
    map50_95 = np.mean([v for k, v in map_results.items()])
    
    return {
        "classes": all_classes,
        "per_class_ap": ap_results,
        "map50": map50,
        "map50-95": map50_95,
        "per_iou_map": map_results
    }

def main():
    # 加载标签（区分真实标签和预测标签）
    print("加载真实标签...")
    gt_labels = load_labels(GT_LABELS_DIR, IMAGES_DIR, is_prediction=False)
    print(f"加载完成，共 {len(gt_labels)} 个真实标签文件")
    
    print("加载预测标签（包含置信度）...")
    pred_labels = load_labels(PRED_LABELS_DIR, IMAGES_DIR, is_prediction=True)
    print(f"加载完成，共 {len(pred_labels)} 个预测标签文件")
    
    # 检查图片匹配
    common_images = set(gt_labels.keys()) & set(pred_labels.keys())
    if not common_images:
        print("错误：真实标签和预测标签没有共同的图片")
        return
    
    print(f"找到 {len(common_images)} 个共同图片用于评估")
    
    # 评估mAP
    print("开始计算mAP指标...")
    results = evaluate_map(gt_labels, pred_labels)
    
    # 输出结果
    print("\n" + "="*60)
    print("📊 预测结果mAP评估对比：")
    print(f"  mAP50: {results['map50']:.4f}")
    print(f"  mAP50-95: {results['map50-95']:.4f}")
    
    print("\n各IoU阈值下的mAP：")
    for iou, map_val in sorted(results['per_iou_map'].items()):
        print(f"  {iou}: {map_val:.4f}")
    
    print("\n各类别的AP@0.50：")
    for cls in results['classes']:
        print(f"  类别 {cls}: {results['per_class_ap'][cls][0.5]:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
