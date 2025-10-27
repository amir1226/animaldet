# Patcher execution

In order to generate patches we use `simple_coco_patcher.py`, script located in `RF-DETR/simple_coco_patcher.py`.

```bash
uv run simple_coco_patcher.py --images_dir data/herdnet/raw/train --json_file data/herdnet/raw/groundtruth/json/big_size/train_big_size_A_B_E_K_WH_WB.json  --output_dir data/train --patch_hei
ght 512 --patch_width 512 --overlap 160 --min_visibility 0.8

uv run simple_coco_patcher.py --images_dir data/herdnet/raw/test --json_file data/herdnet/raw/groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB.json  --output_dir data/test --patch_hei
ght 512 --patch_width 512 --overlap 160 --min_visibility 0.8

uv run simple_coco_patcher.py --images_dir data/herdnet/raw/val --json_file data/herdnet/raw/groundtruth/json/big_size/val_big_size_A_B_E_K_WH_WB.json  --output_dir data/valid --patch_hei
ght 512 --patch_width 512 --overlap 160 --min_visibility 0.8
```