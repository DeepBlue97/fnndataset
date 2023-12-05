from torch.utils.data import DataLoader

from fnndataset.coco.det import COCOYOLOXDataset
from fnnaug.augment.yolox import TrainTransform

dataset_root = '/mnt/d/Share/datasets/coco'

preproc = TrainTransform(
            max_labels=50,
            flip_prob=0.5,
            hsv_prob=1.0,)

dataset = COCOYOLOXDataset(
        data_dir=dataset_root,
        json_file="train_ADAS.json",
        name="train2017",
        img_size=(416, 416),
        preproc=preproc,
        cache=False,
        cache_type="ram",
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

for i, data in enumerate(loader):
    print(data)
