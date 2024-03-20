from ultralytics import YOLO
import cv2
from pathlib import Path


# 模型路径
model_path = Path('../../fry_model').resolve()
# 资源路径
asset_path = Path('../../fry_assets').resolve()



# Load a model
model = YOLO(model_path / 'yolov8n-cls.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model('../ultralytics/assets/bus.jpg')  # predict on an image