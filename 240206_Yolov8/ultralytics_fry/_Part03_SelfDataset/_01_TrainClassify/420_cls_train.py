from ultralytics import YOLO
import cv2
from pathlib import Path


# 模型路径
model_path = Path('../../fry_model').resolve()
# 资源路径
asset_path = Path('../../fry_assets').resolve()


if __name__ == '__main__':


    # Load a model
    model = YOLO(model_path /'yolov8n-cls.pt',task="classify")  # load a pretrained model (recommended for training)

    # Train the model
    # results = model.train(data='mnist160', epochs=100, imgsz=64)
    # results = model.train(data='./YoloClsDataset4060.yaml', project="./", epochs=100, imgsz=64)
    # 230926_1556_ 分类任务可能并不支持yaml文件的方式
    dataPath = r"D:\_230711_learnArchive\fry_course_materials\240206_Yolov8\ultralytics_fry\_Part03_SelfDataset\_01_TrainClassify\mnist160"
    results = model.train(data=dataPath, project="./", epochs=100, imgsz=64)



