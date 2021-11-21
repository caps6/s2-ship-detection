#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model validation for ship detection."""
from os import path, makedirs
import pandas as pd
import numpy as np
import torch
from datasets import AirbusShipDetection
from imageutils import draw_image_with_boxes
from nnutils import get_transform

DATA_DIR = '../Data/airbus'
DIR_TRAIN_IMAGES = path.join(DATA_DIR, 'train_images')
DIR_TRAIN_MASKS = path.join(DATA_DIR, 'train_masks')
VALID_DIR = path.join(DATA_DIR, 'validation')

if not path.exists(VALID_DIR):
    makedirs(VALID_DIR)

# Corrupted images.
exclude_list = ['6384c3e78.jpg', '13703f040.jpg', '14715c06d.jpg',
    '33e0ff2d5.jpg', '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg',
    'a8d99130e.jpg', 'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg',
    'dc3e7c901.jpg', 'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']


if __name__ == '__main__':

    # Set num of epochs
    n_epochs = 5

    # Set device: `cuda` or `cpu`
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #
    # DATA LOADING
    #

    # Load training data.
    metadata = path.join(DATA_DIR, 'train_ship_segmentations_v2.csv')
    df_metadata = pd.read_csv(metadata)
    df_metadata['ImageId'] = df_metadata.ImageId.apply(lambda img_id: img_id.replace('.jpg', ''))

    for corrupted in exclude_list:
        df_metadata = df_metadata[~df_metadata.ImageId.str.contains(corrupted)]

    # Remove images without ships.
    df_metadata.dropna(inplace = True)

    # Remove duplicate image ids.
    df_metadata = df_metadata.drop_duplicates(subset = ['ImageId'])

    # Shuffle dataframe.
    df_metadata = df_metadata.sample(frac = 1).reset_index(drop = True)
    #df_metadata = df_metadata.sample(frac = 0.05, random_state = 42)

    # Perform train / validation split from labeled data.
    df_valid = df_metadata.sample(frac = 0.2, random_state = 42)
    print(f'Size of validation data: {len(df_valid)}')

    # Create validation dataset.
    dataset_valid = AirbusShipDetection(df_valid.ImageId.tolist(),
        DIR_TRAIN_IMAGES, DIR_TRAIN_MASKS, get_transform(train = False))

    #
    # MODEL VALIDATION
    #

    # load best saved model.
    if path.exists('./best_model.pth'):
        model = torch.load('./best_model.pth', map_location = device)
        print('Loaded best model available.')
    else:
        raise ValueError('No model found.')

    print('Starting to classify images...')

    for idx in range(len(dataset_valid)):

        if idx % 100 == 0:
            print(f'Processing image num. {idx}')

        image_id = df_valid.iloc[idx].ImageId

        image, gt_tgts = dataset_valid[idx]

        x_tensor = image.to(device).unsqueeze(0)

        # Predict test image.
        tgt_pred = model(x_tensor)
        tgt_pred = tgt_pred[0]

        # Get the boxes.
        boxes = tgt_pred['boxes'].detach().cpu().numpy()

        # Convert image from `CHW` format to `HWC` format.
        image = np.transpose(image.detach().cpu().numpy(), (1,2,0))

        fn = path.join(VALID_DIR, f'{image_id}_pred.png')
        draw_image_with_boxes(fn, image, boxes)

    print('Done.')
