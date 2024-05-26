from ultralytics import YOLO
from pathlib import Path
import os

# _240526_2159_ 解决可能出现的OMP错误 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型路径
model_path = Path('../model').resolve()
# 资源路径
asset_path = Path('../assets').resolve()

# Load a model
model = YOLO(model_path / 'yolov8n.pt')
# model = YOLO('yolov8n-seg.pt')

# Track with the model
results = model.track(source=str(asset_path / "01.mp4"), show=True)


"""
ModuleNotFoundError: No module named 'lap'

pip install lapx

Installing collected packages: Cython, lapx
Successfully installed Cython-3.0.2 lapx-0.5.4
"""