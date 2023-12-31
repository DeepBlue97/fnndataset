import os
import warnings
import random
import copy

import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F

from fnndataset.coco import *
from .base import CacheDataset, cache_read_img


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = []
        for path in self.img_files:
            image_dir = os.path.dirname(path)
            label_dir = "labels".join(image_dir.rsplit("images", 1))
            assert label_dir != image_dir, \
                f"Image path must contain a folder named 'images'! \n'{image_dir}'"
            label_file = os.path.join(label_dir, os.path.basename(path))
            label_file = os.path.splitext(label_file)[0] + '.txt'
            self.label_files.append(label_file)

        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        try:

            img_path = self.img_files[index % len(self.img_files)].rstrip()

            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        except Exception:
            print(f"Could not read image '{img_path}'.")
            return

        # ---------
        #  Label
        # ---------
        try:
            label_path = self.label_files[index % len(self.img_files)].rstrip()

            # Ignore warning if file is empty
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                boxes = np.loadtxt(label_path).reshape(-1, 5)
        except Exception:
            print(f"Could not read label '{label_path}'.")
            return

        # -----------
        #  Transform
        # -----------
        if self.transform:
            try:
                img, bb_targets = self.transform((img, boxes))
            except Exception:
                print("Could not apply transform.")
                return

        return img_path, img, bb_targets

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets

    def __len__(self):
        return len(self.img_files)


class CocoDataset(Dataset):
    def __init__(self, root_dir, ann_file, img_path, transform=None, names_file=None):
        """
        初始化COCO数据集。

        Args:
            root_dir (str): COCO数据集根目录。
            ann_file (str): 数据集名称, 可以是train2017、val2017、test2017中的任意一个。
            img_path: 图片所在目录
            transform (callable, optional): 用于数据转换的函数。
            names_file: 类别名称, 由于COCO的类别ID大于实际类别个数, 所以需要自己写个包含类别名的文件, 以此设置对应ID

        """
        self.root_dir = root_dir
        # self.set_name = set_name
        self.ann_file = ann_file
        self.transform = transform
        self.img_path = img_path
        # 由于
        self.names_file = names_file
        if self.names_file is not None:
            self.names = open(os.path.join(root_dir, self.names_file), 'r').read().split('\n')
            if '' in self.names:
                self.names.remove('')

        # 加载COCO注释文件
        self.coco = COCO(os.path.join(self.root_dir, self.ann_file))

        # 获取所有图像ID和注释ID
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        # 过滤没有标签的图像
        self.img_ids = [i for i in self.img_ids if len(self.coco.getAnnIds(imgIds=i))]

        # self.ann_ids = self.coco.getAnnIds(imgIds=self.img_ids)

    def __getitem__(self, idx):
        """
        获取COCO数据集中的一张图像和其注释。

        Args:
            idx (int): 图像索引。

        Returns:
            tuple: 包含图像和注释的元组。
        """
        # 获取图像ID和注释ID
        img_id = self.img_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        # 加载图像
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.root_dir, self.img_path, img_info['file_name'])
        # img = Image.open(img_path).convert('RGB')
        img = cv2.imread(img_path)
        img_width_org = img.shape[1]
        img_height_org = img.shape[0]
        # img_width_org = img.width
        # img_height_org = img.height
        # img = cv2.imread(img_path)

        # 加载注释
        anns = self.coco.loadAnns(ann_ids)
        # bbox = [[ann['bbox'], ann['category_id']] for ann in anns]
        bbox = np.array([ann['bbox'] for ann in anns])
        # 左上角变为中心点
        bbox[:,0] = bbox[:,0] + bbox[:,2] / 2
        bbox[:,1] = bbox[:,1] + bbox[:,3] / 2
        # 像素坐标归一化
        bbox[:,0] = bbox[:,0] / img_width_org
        bbox[:,2] = bbox[:,2] / img_width_org
        bbox[:,1] = bbox[:,1] / img_height_org
        bbox[:,3] = bbox[:,3] / img_height_org

        # 1开始，不是0开始
        if self.names_file is not None:
            category_id = [self.names.index(self.coco.cats[ann['category_id']]['name'])+1 for ann in anns]
        else:
            category_id = [ann['category_id'] for ann in anns]

        # if len(bbox)<200:
        #     bbox += [[0]*5]*(200-len(bbox))

        # bbox = np.array(bbox)

        # 对图像和注释进行转换
        if self.transform is not None:
            # img, anns = self.transform(img)
            img, bbox = self.transform([img, bbox])


        return img, bbox, category_id
        # return img, np.cat(category_id, bbox)


    def __len__(self):
        """
        返回数据集中图像的数量。

        Returns:
            int: 数据集中图像的数量。
        """
        return len(self.img_ids)


class COCOYOLOXDataset(CacheDataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
        cache_type="ram",
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        # if data_dir is None:
        #     data_dir = os.path.join(get_yolox_datadir(), "COCO")
        self.data_dir = data_dir
        self.json_file = json_file

        self.coco = COCO(os.path.join(self.data_dir, self.json_file))
        remove_useless_info(self.coco)
        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_coco_annotations()

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id
