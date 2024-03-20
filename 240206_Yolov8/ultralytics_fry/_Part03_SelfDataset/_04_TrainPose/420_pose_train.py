from ultralytics import YOLO


if __name__ == "__main__":

    # Load a model
    model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='./fry-coco8-pose.yaml', epochs=100, imgsz=640)



"""

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     53/100     0.872G       1.01      3.572     0.3692     0.9624      1.164         15        640: 100%|██████████| 1/1 [00:00<00:00,  8.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00, 10.96it/s]
                   all          4         14      0.846      0.429      0.509       0.34      0.821      0.143      0.154     0.0629

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss  Instances       Size
     54/100     0.875G     0.9845      3.756     0.2867      1.017      1.319         14        640: 100%|██████████| 1/1 [00:00<00:00,  6.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00, 10.99it/s]
                   all          4         14       0.83       0.35      0.452      0.284      0.506      0.143      0.156     0.0512
Stopping training early as no improvement observed in last 50 epochs. Best results observed at epoch 4, best model saved as best.pt.
To update EarlyStopping(patience=50) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

54 epochs completed in 0.011 hours.
Optimizer stripped from D:\_230711_learnArchive\LA_ai_main_CV_classify_detect_segment\runs\pose\train11\weights\last.pt, 6.8MB
Optimizer stripped from D:\_230711_learnArchive\LA_ai_main_CV_classify_detect_segment\runs\pose\train11\weights\best.pt, 6.8MB

Validating D:\_230711_learnArchive\LA_ai_main_CV_classify_detect_segment\runs\pose\train11\weights\best.pt...
Ultralytics YOLOv8.0.157  Python-3.10.13 torch-2.0.1+cu118 CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU, 8188MiB)
YOLOv8n-pose summary (fused): 187 layers, 3289964 parameters, 0 gradients
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  6.90it/s]
                   all          4         14        0.9      0.929       0.95      0.677       0.84        0.5      0.535      0.352
Speed: 1.5ms preprocess, 17.4ms inference, 0.0ms loss, 1.6ms postprocess per image
Results saved to D:\_230711_learnArchive\LA_ai_main_CV_classify_detect_segment\runs\pose\train11

进程已结束,退出代码0


"""




