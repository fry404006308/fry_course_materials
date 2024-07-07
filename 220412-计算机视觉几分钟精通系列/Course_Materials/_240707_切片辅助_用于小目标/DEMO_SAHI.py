import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2

def create_random_image(width, height):
    # 创建一个随机的彩色图像
    assert width > 0, "创建图片的宽必须大于0"
    assert height > 0, "创建图片的高必须大于0"

    ans_img = np.zeros((height, width, 3), dtype=np.uint8)

    # 创建颜色渐变背景
    for y in range(height):
        for x in range(width):
            ans_img[y, x, 0] = (x // 3) % 255  # B通道
            ans_img[y, x, 1] = (y // 2) % 255  # G通道
            ans_img[y, x, 2] = ((x + y) // 4) % 255  # R通道

    width_1_2 = width // 2
    width_1_3 = width // 3
    width_1_4 = width // 4
    width_1_5 = width // 5
    width_1_6 = width // 6
    width_1_7 = width // 7
    width_1_8 = width // 8

    height_1_2 = height // 2
    height_1_3 = height // 3
    height_1_4 = height // 4
    height_1_5 = height // 5
    height_1_6 = height // 6
    height_1_7 = height // 7
    height_1_8 = height // 8

    # 画矩形
    cv2.rectangle(ans_img, (width_1_8, height_1_8), (width_1_8 + width_1_2, height_1_8 + height_1_2), (255, 0, 255),
                  -1)

    # 画圆形
    cv2.circle(ans_img, (width_1_2, height_1_2), min(width_1_3, height_1_3), (0, 255, 255), 2)

    # 画三角形
    triangle = np.array([[width_1_3, height_1_3], [width_1_3 * 2, height_1_3 * 2], [width_1_3, height_1_3 * 2]],
                        np.int32)
    cv2.fillConvexPoly(ans_img, triangle, (255, 255, 0))

    # 在图片中间写字
    cv2.putText(ans_img, "Hello, OpenCV!", (width_1_5, height_1_5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255),
                2)

    return ans_img


def slice_image(image, slice_size, overlap):
    slices = []
    h, w = image.shape[:2]
    stride = int(slice_size * (1 - overlap))

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x_end = min(x + slice_size, w)
            y_end = min(y + slice_size, h)
            x_start = max(0, x_end - slice_size)
            y_start = max(0, y_end - slice_size)

            slice_img = image[y_start:y_end, x_start:x_end]
            slices.append((slice_img, (x_start, y_start)))

    return slices


def detect_objects(slice_img):
    h, w = slice_img.shape[:2]
    num_objects = np.random.randint(1, 4)
    detections = []
    for _ in range(num_objects):
        x1, y1 = np.random.randint(0, w - 20), np.random.randint(0, h - 20)
        x2, y2 = x1 + np.random.randint(40, 80), y1 + np.random.randint(40, 80)
        score = np.random.uniform(0.5, 1.0)
        class_id = np.random.randint(0, 10)
        # 这里是把随机生成的坐标及分数，及分类id当做检测结果输出出去了
        detections.append([x1, y1, x2, y2, score, class_id])
    return np.array(detections)


def merge_results(slices_with_detections, original_size):
    all_detections = []

    for slice_img, (ox, oy), detections in slices_with_detections:
        for det in detections:
            # 拿到检测结果
            x1, y1, x2, y2, score, class_id = det
            # 加上位置偏移
            x1, x2 = x1 + ox, x2 + ox
            y1, y2 = y1 + oy, y2 + oy
            all_detections.append([x1, y1, x2, y2, score, class_id])

    return np.array(all_detections)


def nms(boxes, scores, iou_threshold):
    """
    执行非极大值抑制(NMS)来过滤重叠的边界框。

    参数:
    boxes: 形状为(N, 4)的numpy数组，每行表示一个边界框 [x1, y1, x2, y2]
    scores: 形状为(N,)的numpy数组，表示每个边界框的置信度得分
    iou_threshold: 浮点数，IOU阈值，用于决定是否抑制重叠的框

    返回:
    keep: 列表，包含保留的边界框的索引
    """
    # 按分数降序排序
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        # 选择分数最高的边界框
        i = order[0]
        keep.append(i)

        # 计算该框与其他所有框的IOU
        ious = compute_iou(boxes[i], boxes[order[1:]])

        # 保留IOU小于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        # 更新order，+1是因为我们跳过了第一个元素（当前处理的框）
        order = order[inds + 1]

    return keep

def compute_iou(box, boxes):
    """
    计算一个框与多个框的IOU（交并比）。

    参数:
    box: 形状为(4,)的numpy数组，表示单个边界框 [x1, y1, x2, y2]
    boxes: 形状为(N, 4)的numpy数组，表示N个边界框 [x1, y1, x2, y2]

    返回:
    iou: 形状为(N,)的numpy数组，表示box与boxes中每个框的IOU
    """
    # 计算交集区域
    inter_area = np.maximum(0, np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])) * \
                 np.maximum(0, np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]))

    # 计算每个框的面积
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 计算并集区域
    union_area = box_area + boxes_area - inter_area

    # 计算IOU
    iou = inter_area / union_area

    return iou


def visualize_detections(image, detections):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1+5, y1+5), f"{int(class_id)}: {score:.2f}", fill="green")
    return np.array(img_pil)


# 主流程
image_width, image_height = 800, 600
image = create_random_image(image_width, image_height)
slice_size = 416
overlap = 0.2

# 1. 切片
# _240707_1401_ 切图没问题
slices = slice_image(image, slice_size, overlap)

# 2. 对每个切片进行检测
slices_with_detections = []
for slice_img, coords in slices:
    detections = detect_objects(slice_img)
    slices_with_detections.append((slice_img, coords, detections))

# 3. 合并结果
merged_detections = merge_results(slices_with_detections, image.shape[:2])

# 4. 后处理(NMS)
if len(merged_detections) > 0:
    boxes = merged_detections[:, :4]
    scores = merged_detections[:, 4]
    keep = nms(boxes, scores, iou_threshold=0.5)
    final_detections = merged_detections[keep]
else:
    final_detections = []

# 可视化结果
result_image = visualize_detections(image, final_detections)

plt.figure(figsize=(12, 8))
plt.imshow(result_image)
plt.axis('off')
plt.show()
plt.imsave("answer.jpg", result_image)

print(f"检测到 {len(final_detections)} 个目标")