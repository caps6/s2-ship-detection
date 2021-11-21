# -*- coding: utf-8 -*-
"""Custom dataset for Airbus images."""
from torch.utils.data import Dataset
from os import path
from PIL import Image
import numpy as np
import torch

class AirbusShipDetection(Dataset):

    def __init__(self, image_ids, dir_images, dir_masks, transforms = None):

        self.image_ids = image_ids
        self.dir_images = dir_images
        self.dir_masks = dir_masks
        self._transforms = transforms

    def __getitem__(self, idx):

        # Read the RGB image.
        fn_image = f'{self.image_ids[idx]}.jpg'
        path_image = path.join(self.dir_images, fn_image)
        image = Image.open(path_image).convert("RGB")

        # Read the integer-based mask.
        fn_mask = f'{self.image_ids[idx]}_mask.png'
        path_mask = path.join(self.dir_masks, fn_mask)
        mask = np.array(Image.open(path_mask))

        # Instances are encoded with different integers.
        obj_ids = np.unique(mask)

        # We remove the background (id=0) from the mask.
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # Split the mask into a set of binary masks
        # masks.shape[0] = number of istances
        masks = mask == obj_ids[:, None, None]

        # Get bounding box of each mask.
        boxes = []
        for mask in masks:

            pos = np.where(mask)

            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            # Enforce a positive area.
            if xmax - xmin < 1:
                xmax += 1
            if ymax - ymin < 1:
                ymax += 1

            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype = torch.float32)

        # Compute the area.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Only one class (ships).
        labels = torch.ones((num_objs,), dtype = torch.int64)
        masks = torch.as_tensor(masks, dtype = torch.uint8)

        # Crowd flag not applicable here.
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = image_id
        target['area'] = area
        target['iscrowd'] = iscrowd

        # Apply image augmentation.
        if self._transforms:
            image, target = self._transforms(image, target)

        return image, target

    def __len__(self):
        # return length of
        return len(self.image_ids)
