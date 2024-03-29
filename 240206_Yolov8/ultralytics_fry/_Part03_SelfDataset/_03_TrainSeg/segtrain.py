
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-seg.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolo_seg_dataset.yaml').load('yolov8m-seg.pt')  # build from YAML and transfer weights

if __name__ == '__main__':
    # Use the model
    # results = model.train(data="coco128.yaml", epochs=3)  # train the model
    # results = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # success = YOLO("yolov8n.pt").export(format="onnx")  # export a model to ONNX format

    model = YOLO('./yolov8m-seg.pt')

    # Train the model
    # 230926_1058_ 亲测 project 参数是一定可以的，也就是 project 参数在某个地方传进去了
    # 因为那个train的目录就直接到了当前目录下面
    results = model.train(data='./yoloSegDataset4060.yaml', project="./",epochs=100, imgsz=640, batch=8)
