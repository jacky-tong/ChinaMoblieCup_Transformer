# vote_config.py
import numpy as np

def nms_boxes(boxes, iou_threshold=0.5):
    """
    对预测框进行非极大值抑制（NMS）
    boxes: ndarray, shape=(N, 6), 每行为 [x1, y1, x2, y2, conf, cls]
    iou_threshold: IoU阈值
    返回: 过滤后的boxes
    """
    if len(boxes) == 0:
        return []

    # 提取坐标和置信度
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return boxes[keep]