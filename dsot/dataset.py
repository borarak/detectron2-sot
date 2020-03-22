import random
from pathlib import Path
from skimage import io
import os
import torch
import json
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, ColorJitter
from dsot.bbox_utils import get_bbox, center2corner, Center

random.seed(42)

x_transforms = Compose([
    ColorJitter(brightness=0.25, contrast=0.25),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
z_transforms = Compose([
    ColorJitter(brightness=0.25, contrast=0.25),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class COCODataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        train_dir="/home/rex/datasets/coco2017/SiamFCCrop511/train2017",
        anno_file="/home/rex/datasets/coco2017/SiamFCCrop511_anno/train2017.json",
        eval_mode=False):
        """

        Args:
            config: Path to detectron2 style config
            train_dir: Path to train images
            anno_file: Path to annotation file
            eval_mode: True when "testing"
        """
        self.anno_file = anno_file
        self.cfg = config
        self.anno_data = json.load(open(self.anno_file, "r"))

        self.root_dir = train_dir
        assert os.path.isdir(self.root_dir), "train_dir must be a valid dir"
        self.image_folders = os.listdir(self.root_dir)

        # limit some folders with less annotations
        self.image_folders = [
            folder for folder in self.image_folders
            if len(os.listdir(os.path.join(self.root_dir, folder))) >= 16
        ]
        random.shuffle(self.image_folders)

        if not eval_mode:
            self.image_folders = self.image_folders[:self.cfg.
                                                    NUM_IMAGE_FOLDERS]
            self.x_images = [
                self.generate_combinations(folder)
                for folder in self.image_folders
            ]
            self.x_images = [y for x in self.x_images for y in x]

            xx_images = np.asarray(self.x_images).reshape(
                (-1, self.cfg.MAGIC_NUMBER))
            new_images = []
            for yy in range(xx_images.shape[0] // 2):
                bat = []
                bat += list(xx_images[yy])
                bat += list(xx_images[yy + (xx_images.shape[0] // 2)])
                new_images.append(bat)

            random.shuffle(new_images)
            new_images = [y for x in new_images for y in x]
            self.x_images = [str(Path(x)) for x in new_images]
            print(f"Training dataset has {len(self.x_images)} images")
        else:
            self.x_images = [
                self.generate_combinations(folder)
                for folder in self.image_folders
            ]
            self.x_images = [y for x in self.x_images for y in x]
            print(f"Validation dataset has {len(self.x_images)} images")

        self.z_images = [
            str(x).replace("x.jpg", "z.jpg") for x in self.x_images
        ]

    def generate_combinations(self, folder_path):
        num_combinations = self.cfg.EPOCHS_PER_FOLDER * self.cfg.MAGIC_NUMBER

        folder_path = os.path.join(self.root_dir, folder_path)
        x_images = [
            str(x) for x in Path(folder_path).glob('*x.jpg')
            if os.path.exists(str(x).replace("x.jpg", "z.jpg"))
        ]
        x_images = [random.choice(x_images) for _ in range(num_combinations)]
        return x_images

    def _get_bbox(self, image, shape):

        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = 127
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __getitem__(self, idx):
        pos = self.random.choice([True, False])
        x_img = self.x_images[idx]
        object_id = str(x_img.split("/")[-1]).split(".")[-3]
        frame = str(x_img.split("/")[-2])
        bbox = self.anno_data[str(self.root_dir.split("/")[-1]) + "/" +
                              str(frame)][object_id]["000000"]
        if pos:
            z_img = self.z_images[idx]
        else:
            z_img = self.z_images[self.random.choice(range(len(
                self.z_images)))]

        x_img_cv = x_transforms(Image.open(x_img))
        z_img_cv = z_transforms(Image.open(z_img))
        bbox = self._get_bbox(io.imread(x_img), bbox)

        return {
            'search': x_img_cv,
            'exemplar': z_img_cv,
            'label': 1 if pos else -1,
            'bbox': bbox
        }

    def __len__(self):
        return len([x for x in self.x_images if x.split(".")[-2] == "x"])
