from ultralytics import YOLO
import cv2


# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('./train/weights/best.pt')  # load a custom model

# Predict with the model
results = model('./TestImage/000000000061.jpg',save=True,project="./")  # predict on an image



res = results[0].plot(boxes=False) #boxes=False表示不展示预测框，True表示同时展示预测框
# Display the annotated frame
cv2.imshow("YOLOv8 Inference", res)
cv2.waitKey(0)


r"""

image 1/1 D:\_230711_learnArchive\fry_course_materials\240206_Yolov8\ultralytics_fry\_Part03_SelfDataset\_02_TrainDetect\TestImage\000000000061.jpg: 512x640 3 persons, 2 elephants, 96.6ms
Speed: 2.3ms preprocess, 96.6ms inference, 2.2ms postprocess per image at shape (1, 3, 512, 640)
Results saved to predict

进程已结束,退出代码0

"""
