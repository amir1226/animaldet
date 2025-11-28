# Hard negative patches

Generate patches:
```bash 
uv run simple_coco_patcher.py --images_dir data/herdnet/raw/train --json_file eval_results_coco.json  --output_dir data/hnp --patch_hei
ght 512 --patch_width 512 --overlap 160 --min_visibility 0.8
```
