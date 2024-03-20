# from ultralytics import YOLO
from ultralytics_fry.ultralytics_fry_base import YOLO
import cv2
from pathlib import Path


# 模型路径
model_path = Path('../../fry_model').resolve()
# 资源路径
asset_path = Path('../../fry_assets').resolve()


print(model_path)


# 加载预训练模型
model = YOLO(r"./yolov8n_CBAM.yaml")
print(model)
# model = YOLO(r".\ultralytics\cfg\models\v8\fry_config\yolov8_CBAM.yaml")

# 检测图片
results = model(asset_path / "bus.jpg")










