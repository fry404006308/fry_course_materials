from ultralytics import YOLO
# from ultralytics_fry.ultralytics_fry_base import YOLO
import cv2
from pathlib import Path


# 模型路径
model_path = Path('../model').resolve()
# 资源路径
asset_path = Path('../assets').resolve()


# 加载预训练模型
model = YOLO(model_path / "yolov8n-obb.pt")
# model = YOLO("yolov8n.pt") task参数也可以不填写，它会根据模型去识别相应任务类别
# 检测图片
results = model(asset_path / "P1053__1024__0___90.jpg")
res = results[0].plot()

def tensor2numpy(now_tensor):
    now_numpy = now_tensor.squeeze().detach().cpu().numpy()
    return now_numpy

obb_ans = results[0].obb

print("obb_ans.conf : \n",tensor2numpy(obb_ans.conf))
print("obb_ans.xywhr : \n",tensor2numpy(obb_ans.xywhr))
print("obb_ans.xyxy : \n",tensor2numpy(obb_ans.xyxy))
print("obb_ans.xyxyxyxy : \n",tensor2numpy(obb_ans.xyxyxyxy))
print("obb_ans.xyxyxyxyn : \n",tensor2numpy(obb_ans.xyxyxyxyn))



cv2.imshow("YOLOv8 Inference", res)
cv2.imwrite("_240621_1341_obb.jpg",res)
cv2.waitKey(0)









