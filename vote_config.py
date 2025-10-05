import os
import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from ViT5_4k import ViTDetector  # 导入ViT检测器类

# 配置参数（与训练时保持一致）
IMG_SIZE = (224, 224)
VIT_MEAN = [0.485, 0.456, 0.406]
VIT_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 4
class_names = ['ship', 'people', 'car', 'motor']  # 类别名称需与训练一致

# 检测后处理参数
CONF_THRESHOLD = 0.5  # 置信度阈值
NMS_IOU_THRESHOLD = 0.5  # NMS的IoU阈值


def calculate_iou(box1, box2):
    """计算两个边界框的IoU"""
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


def nms_boxes(boxes, iou_threshold=0.5):
    """对预测框进行非极大值抑制"""
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    # 按置信度降序排序
    indices = np.argsort(boxes[:, 4])[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        current_box = boxes[current][:4]
        remaining_indices = []

        for i in indices[1:]:
            other_box = boxes[i][:4]
            if calculate_iou(current_box, other_box) <= iou_threshold:
                remaining_indices.append(i)

        indices = np.array(remaining_indices)

    return boxes[keep].tolist()


def load_vit_model(model_path):
    """加载ViT检测模型"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ViTDetector(num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # 处理不同的保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model, device


def vit_predict_image(model, image_path, device, conf_threshold=0.1):
    """使用ViT模型预测单张图片"""
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=VIT_MEAN, std=VIT_STD)
    ])

    # 加载图像
    img = Image.open(image_path).convert('RGB')
    img_np = cv2.imread(str(image_path))
    if img_np is None:
        raise ValueError(f"无法读取图片: {image_path}")
    h, w = img_np.shape[:2]

    # 转换为模型输入格式
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 模型预测
    with torch.no_grad():
        outputs = model(input_tensor)  # [1, num_patches, 5 + num_classes]

    # 解析输出
    outputs = outputs.squeeze(0).cpu().numpy()  # [num_patches, 5 + num_classes]
    boxes = []

    for pred in outputs:
        # 解析预测结果：[x1, y1, x2, y2, conf, cls0, cls1, ...]
        bbox = pred[:4]
        conf = pred[4]
        cls_scores = pred[6:6+NUM_CLASSES]
        cls_id = np.argmax(cls_scores)
        cls_conf = cls_scores[cls_id]
        
        # 综合置信度计算
        total_conf = conf * cls_conf
        
        if total_conf >= conf_threshold:
            # 将归一化坐标转换为原图坐标
            x1, y1, x2, y2 = bbox
            x1 = x1 * w
            y1 = y1 * h
            x2 = x2 * w
            y2 = y2 * h
            
            boxes.append([x1, y1, x2, y2, total_conf, cls_id])

    # 应用NMS
    if boxes:
        boxes = nms_boxes(boxes, NMS_IOU_THRESHOLD)
    
    return boxes


def create_vit_predictions(model_path, val_images_path, val_labels_path, conf=0.5):
    """生成ViT模型的预测结果"""
    # 转换为绝对路径
    val_images_path = os.path.abspath(val_images_path)
    val_labels_path = os.path.abspath(val_labels_path)
    
    # 查找图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    if os.path.isdir(val_images_path):
        for ext in image_extensions:
            image_paths.extend(Path(val_images_path).glob(f'*{ext}'))
            image_paths.extend(Path(val_images_path).glob(f'*{ext.upper()}'))

    print(f"找到 {len(image_paths)} 张验证图片")

    # 创建预测结果目录
    pred_dir = os.path.abspath("./ViT_Pred")
    pred_labels_dir = os.path.join(pred_dir, 'labels')
    pred_images_dir = os.path.join(pred_dir, 'images')
    os.makedirs(pred_labels_dir, exist_ok=True)
    os.makedirs(pred_images_dir, exist_ok=True)

    print(f"开始生成预测结果，将保存至: {pred_dir}")

    # 加载模型
    print("正在加载ViT模型...")
    model, device = load_vit_model(model_path)
    print(f"模型加载完成，使用设备: {device}")

    successful_predictions = 0
    total_boxes = 0

    for img_path in tqdm(image_paths, desc="生成预测结果"):
        try:
            img_path_str = str(img_path)
            image = cv2.imread(img_path_str)
            if image is None:
                print(f"无法读取图片: {img_path_str}")
                continue
            img_height, img_width = image.shape[:2]

            # 模型预测
            pred_boxes = vit_predict_image(
                model, 
                img_path_str, 
                device, 
                conf_threshold=0.1  # 低阈值筛选，后续再过滤
            )

            # 复制图片到预测目录
            img_name = Path(img_path).name
            shutil.copy2(img_path_str, os.path.join(pred_images_dir, img_name))

            # 保存预测结果
            label_name = Path(img_path).stem + '.txt'
            label_path = os.path.join(pred_labels_dir, label_name)

            with open(label_path, 'w') as f:
                for box in pred_boxes:
                    if len(box) >= 6:
                        x1, y1, x2, y2, conf_score, cls = box[:6]

                        # 过滤低置信度框
                        if conf_score < conf:
                            continue

                        # 修正坐标
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)

                        # 裁剪到图片范围内
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_width, x2)
                        y2 = min(img_height, y2)

                        # 转换为COCO TXT格式（归一化）
                        center_x = ((x1 + x2) / 2) / img_width
                        center_y = ((y1 + y2) / 2) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height

                        # 确保坐标在[0,1]范围内
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        # 写入格式：类别ID 中心x 中心y 宽度 高度 置信度
                        f.write(f"{int(cls)} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {conf_score:.6f}\n")
                        total_boxes += 1

            successful_predictions += 1

        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")
            continue

    print(f"成功处理 {successful_predictions} 张图片，共生成 {total_boxes} 个预测框")

    # 创建数据配置文件
    data_yaml_path = os.path.join(pred_dir, 'data.yaml')
    data_config = {
        'train': '',
        'val': val_images_path,
        'test': '',
        'nc': len(class_names),
        'names': class_names,
        'val_labels': val_labels_path
    }

    with open(data_yaml_path, 'w') as f:
        yaml.dump(data_config, f)

    return pred_dir, data_yaml_path, total_boxes


def load_labels(label_dir, image_dir, is_prediction=False):
    """
    加载标签文件并转换为绝对坐标
    :param is_prediction: 是否为预测标签（6值格式，含置信度）
    :return: 标签字典 {图片名: [(x1, y1, x2, y2, cls_id, [conf])] }
    """
    labels = {}
    label_files = list(Path(label_dir).glob("*.txt"))
    
    for label_file in label_files:
        img_name = label_file.stem
        # 更稳健的图片查找：支持不同扩展名与大小写
        image_dir_path = Path(image_dir)
        img_path = image_dir_path / f"{img_name}.jpg"
        if not img_path.exists():
            # 尝试任意同名扩展（例如 .JPG/.JPEG/.png/.bmp/.tiff）
            candidates = list(image_dir_path.glob(f"{img_name}.*"))
            img_path = None
            valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
            for c in candidates:
                if c.suffix.lower() in valid_exts:
                    img_path = c
                    break
            # 若未找到，尝试全目录不区分大小写匹配 stem
            if img_path is None:
                for p in image_dir_path.iterdir():
                    if p.is_file() and p.stem.lower() == img_name.lower() and p.suffix.lower() in valid_exts:
                        img_path = p
                        break
            # 归一为字符串路径或 None
            img_path = str(img_path) if img_path is not None else None
         
        # 获取图片尺寸
        if img_path is None or not os.path.exists(img_path):
            print(f"警告：未找到图片 {img_name}，跳过该标签文件 (在 {image_dir} 中未发现匹配文件)")
            continue
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
         # 读取标签内容
        with open(label_file, 'r') as f:
            for line in f.readlines():
                if is_prediction:
                    # 预测标签格式：x1 y1 x2 y2 cls_id conf
                    parts = line.strip().split()
                    if len(parts) != 6:
                        continue
                    x1, y1, x2, y2, cls_id, conf = parts
                    conf = float(conf) if conf != 'nan' else 1.0  # 处理nan
                else:
                    # 标准标签格式：cls_id x_center y_center width height
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id, x_center, y_center, width, height = parts
                    # 反归一化
                    x1 = (float(x_center) - float(width) / 2) * w
                    y1 = (float(y_center) - float(height) / 2) * h
                    x2 = (float(x_center) + float(width) / 2) * w
                    y2 = (float(y_center) + float(height) / 2) * h
                
                # 转换为绝对坐标
                cls_id = int(cls_id)
                if cls_id >= len(class_names):
                    print(f"警告：标签中类别ID超出范围（ID: {cls_id}），图片: {img_name}")
                    continue

                if img_name not in labels:
                    labels[img_name] = []
                
                labels[img_name].append((x1, y1, x2, y2, cls_id, [conf]))
    
    return labels


if __name__ == '__main__':
    import shutil  # 延迟导入，避免在不需要时加载

    # 配置路径
    model_path = r"E:\2025CompetitionLog\9.9\Travel3_Transformer\runs\vit_detector\vit_detector\weights\best.pt"  # 替换为你的best.pt路径
    val_images_path = r"E:\2025CompetitionLog\9.9\tRVALCUP_2\ChinaMoblieCup\dataset\images\val" # 验证集图片路径
    val_labels_path = r"E:\2025CompetitionLog\9.9\tRVALCUP_2\ChinaMoblieCup\dataset\labels\val"  # 验证集标签路径

    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        exit()

    # 生成预测结果
    pred_dir, data_yaml_path, total_boxes = create_vit_predictions(
        model_path,
        val_images_path,
        val_labels_path,
        conf=0.5  # 最终置信度阈值
    )

    print("\n" + "=" * 60)
    print("✅ 预测结果生成完成：")
    print(f"  生成路径: {pred_dir}")
    print(f"  包含内容:")
    print(f"    - labels文件夹（预测标签，每行6个值）")
    print(f"    - images文件夹（预测使用的图片副本）")
    print(f"    - data.yaml（数据配置文件）")
    print(f"  统计信息: 成功处理 {len(list(Path(val_images_path).glob('*')))} 张图片中的 {total_boxes} 个目标")
    print("=" * 60)