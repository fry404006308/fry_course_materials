from ultralytics import YOLO
import cv2
from pathlib import Path


# 模型路径
model_path = Path('../model').resolve()
# 资源路径
asset_path = Path('../assets').resolve()

# Load a model
model = YOLO(model_path / 'yolov8m-seg.pt')
# Predict with the model
results = model(asset_path / 'bus.jpg')  # predict on an image
res = results[0].plot(boxes=False) #boxes=False表示不展示预测框，True表示同时展示预测框
# Display the annotated frame
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)