from ultralytics import YOLO
import cv2
from pathlib import Path


# 模型路径
model_path = Path('../../fry_model').resolve()
# 资源路径
asset_path = Path('../../fry_assets').resolve()



if __name__ == "__main__":


    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data='FryCoco128_4060.yaml',project='./', epochs=100, imgsz=640)




"""


      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      2.47G     0.8411     0.6733      1.013         71        640: 100%|██████████| 8/8 [00:00<00:00,  9.06it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:00<00:00,  5.16it/s]
                   all        128        929      0.825      0.835      0.877      0.719

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      2.52G     0.8292      0.724      1.012         74        640: 100%|██████████| 8/8 [00:00<00:00,  9.01it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  3.56it/s]
                   all        128        929      0.826      0.821      0.875      0.718


100 epochs completed in 0.063 hours.
Optimizer stripped from train3\weights\last.pt, 6.5MB
Optimizer stripped from train3\weights\best.pt, 6.5MB

Validating train3\weights\best.pt...
Ultralytics YOLOv8.0.157  Python-3.10.13 torch-2.0.1+cu118 CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU, 8188MiB)
Model summary (fused): 168 layers, 3151904 parameters, 0 gradients
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4/4 [00:01<00:00,  3.37it/s]
                   all        128        929      0.906      0.817      0.887      0.743
                person        128        254      0.994      0.703      0.889      0.709
               bicycle        128          6      0.698        0.5      0.614      0.486
                   car        128         46      0.949      0.348      0.632      0.345
            motorcycle        128          5          1      0.968      0.995      0.995
              airplane        128          6      0.955          1      0.995       0.94
                   bus        128          7          1      0.782      0.944       0.81
                 train        128          3      0.917          1      0.995      0.831
                 truck        128         12      0.934        0.5      0.628      0.494
                  boat        128          6      0.773      0.576      0.808      0.567
         traffic light        128         14      0.923      0.286      0.425      0.242
             stop sign        128          2      0.873          1      0.995      0.846
                 bench        128          9      0.953      0.889      0.975      0.832
                  bird        128         16          1      0.977      0.995      0.782
                   cat        128          4      0.919          1      0.995      0.995
                   dog        128          9      0.896          1      0.995      0.906
                 horse        128          2      0.885          1      0.995      0.897
              elephant        128         17          1      0.926      0.985      0.872
                  bear        128          1        0.8          1      0.995      0.995
                 zebra        128          4      0.923          1      0.995      0.995
               giraffe        128          9      0.965          1      0.995      0.913
              backpack        128          6      0.967      0.667      0.806      0.687
              umbrella        128         18          1       0.81      0.969      0.825
               handbag        128         19      0.908      0.474      0.728      0.524
                   tie        128          7      0.919      0.857      0.862       0.75
              suitcase        128          4          1       0.94      0.995      0.905
               frisbee        128          5      0.906        0.8      0.802      0.712
                  skis        128          1      0.807          1      0.995      0.895
             snowboard        128          7      0.812      0.857      0.849      0.663
           sports ball        128          6          1      0.547      0.674      0.398
                  kite        128         10          1      0.241      0.775      0.361
          baseball bat        128          4      0.599        0.5      0.828      0.424
        baseball glove        128          7      0.937      0.429      0.438      0.362
            skateboard        128          5      0.949        0.8      0.854      0.681
         tennis racket        128          7          1      0.713      0.718      0.489
                bottle        128         18          1      0.519      0.877      0.617
            wine glass        128         16       0.92        0.5      0.837      0.533
                   cup        128         36          1      0.714      0.934      0.665
                  fork        128          6      0.961      0.833      0.883      0.773
                 knife        128         16      0.887       0.75      0.836      0.537
                 spoon        128         22      0.904      0.636      0.751      0.565
                  bowl        128         28          1      0.825       0.91      0.732
                banana        128          1      0.785          1      0.995      0.995
              sandwich        128          2      0.772          1      0.995      0.995
                orange        128          4      0.757          1      0.995      0.782
              broccoli        128         11      0.882      0.455      0.665      0.489
                carrot        128         24      0.905      0.798      0.949      0.721
               hot dog        128          2      0.865          1      0.995      0.995
                 pizza        128          5      0.936          1      0.995      0.976
                 donut        128         14      0.955          1      0.995      0.934
                  cake        128          4      0.902          1      0.995      0.931
                 chair        128         35      0.889      0.714      0.915      0.707
                 couch        128          6      0.957          1      0.995       0.81
          potted plant        128         14          1      0.975      0.995      0.876
                   bed        128          3      0.899          1      0.995      0.995
          dining table        128         13          1      0.973      0.995       0.87
                toilet        128          2      0.867          1      0.995      0.898
                    tv        128          2      0.875          1      0.995      0.995
                laptop        128          3      0.883          1      0.995      0.952
                 mouse        128          2      0.934          1      0.995        0.4
                remote        128          8      0.954       0.75      0.751      0.666
            cell phone        128          8      0.929        0.5      0.623      0.453
             microwave        128          3      0.936          1      0.995        0.9
                  oven        128          5      0.769        0.8        0.8      0.676
                  sink        128          6      0.882      0.833      0.972      0.634
          refrigerator        128          5      0.914          1      0.995      0.995
                  book        128         29          1      0.478      0.788      0.562
                 clock        128          9      0.955      0.889       0.94      0.826
                  vase        128          2      0.806          1      0.995      0.946
              scissors        128          1       0.47          1      0.497      0.448
            teddy bear        128         21      0.953      0.958      0.993      0.834
            toothbrush        128          5       0.88          1      0.995      0.948
Speed: 0.6ms preprocess, 2.6ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to train3

进程已结束,退出代码0


"""