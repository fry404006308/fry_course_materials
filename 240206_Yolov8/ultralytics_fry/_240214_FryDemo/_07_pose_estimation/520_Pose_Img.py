from ultralytics import YOLO
import cv2
from pathlib import Path


# 模型路径
model_path = Path('../model').resolve()
# 资源路径
asset_path = Path('../assets').resolve()

# Load a model
model = YOLO(model_path / 'yolov8n-pose.pt')  # load a pretrained model (recommended for training)
results = model(asset_path / 'bus.jpg')
res = results[0].plot()
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)