from ultralytics import YOLO

# Load a model
model = YOLO('./train/weights/best.pt')  # load an official model
# model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model('./testImage/1804.png',save=True,project="./")  # predict on an image

"""

说明：下面预测的结果就是8

image 1/1 D:\_230711_learnArchive\LA_ai_main_CV_classify_detect_segment\_200_YOLO\2023_01_YoloV8\920_ultralytics\230926_TrainClassify\TestImg\1804.png: 

64x64 8 1.00, 0 0.00, 7 0.00, 5 0.00, 3 0.00, 1.8ms

Speed: 2.0ms preprocess, 1.8ms inference, 0.0ms postprocess per image at shape (1, 3, 64, 64)

进程已结束,退出代码0
"""
