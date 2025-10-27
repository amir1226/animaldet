# RF-DETR experiment

Animal detection process based on [RF-DETR](https://github.com/roboflow/rf-detr) model.

All the experiment steps are described in the [experiment_1](./experiment_1) folder.

## Expected files after running the patcher

```
.
├── data
│   ├── herdnet/raw/groundtruth/json/big_size
│   │   ├── train_big_size_A_B_E_K_WH_WB.json
│   │   ├── test_big_size_A_B_E_K_WH_WB.json
│   │   ├── val_big_size_A_B_E_K_WH_WB.json
│   ├── herdnet/raw/train
│   │   ├── image_1.jpg
│   ├── herdnet/raw/test
│   │   ├── image_1.jpg
│   ├── herdnet/raw/val
│   │   ├── image_1.jpg
│   ├── test
│   │   ├── image_1.jpg
│   │   ├── _annotations.coco.json
|   ├── train
│   │   ├── image_1.jpg
│   │   ├── _annotations.coco.json
|   ├── val
│   │   ├── image_1.jpg
│   ├── ...
```