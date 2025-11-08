from rfdetr import RFDETRSmall

model = RFDETRSmall()

model.train(
    dataset_dir='data',
    dataset_file="roboflow",
    train_annotations='data/train/_annotations.coco.json',
    val_annotations='data/val/_annotations.coco.json',
    img_size=512,
    epochs=100,
    batch_size=2,
    grad_accum_steps=8,
    output_dir='output',
    wandb=True,
)
