#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model training for ship detection.
Derived from TorchVision Object Detection Finetuning Tutorial
http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
from os import path
import pandas as pd
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
from utils import collate_fn
from nnutils import get_transform
from datasets import AirbusShipDetection

DATA_DIR = '../Data/airbus'
DIR_TRAIN_IMAGES = path.join(DATA_DIR, 'train_images')
DIR_TRAIN_MASKS = path.join(DATA_DIR, 'train_masks')

# Corrupted images (credits to Kaggle notebooks).
exclude_list = ['6384c3e78.jpg', '13703f040.jpg', '14715c06d.jpg',
    '33e0ff2d5.jpg', '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg',
    'a8d99130e.jpg', 'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg',
    'dc3e7c901.jpg', 'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']

def get_model_instance_segmentation(num_classes):

    # Load an instance segmentation model pre-trained pre-trained on COCO.
    model = maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
        hidden_layer, num_classes)

    return model


if __name__ == "__main__":

    # Train on the GPU or on the CPU, if a GPU is not available.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    n_epochs = 10

    #
    # DATA LOADING
    #

    # Load training metadata.
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
    df_train = df_metadata.drop(df_valid.index)
    print(f'Size of training data: {len(df_train)}')
    print(f'Size of validation data: {len(df_valid)}')

    # Create training and validation datasets.
    dataset_train = AirbusShipDetection(df_train.ImageId.tolist(),
        DIR_TRAIN_IMAGES, DIR_TRAIN_MASKS, get_transform(train = True))
    dataset_valid = AirbusShipDetection(df_valid.ImageId.tolist(),
        DIR_TRAIN_IMAGES, DIR_TRAIN_MASKS, get_transform(train = False))

    # Define training and validation data loaders.
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = 4,
        shuffle = True, num_workers = 4, collate_fn = collate_fn)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size = 4,
        shuffle = False, num_workers = 4, collate_fn = collate_fn)

    #
    # MODEL TRAINING
    #

    # Get the model pretrained with COCO dataset.
    num_classes = 2
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # Optimizer and learning rate scheduler.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = 0.005, momentum = 0.9,
        weight_decay = 0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3,
        gamma = 0.1)

    # Start training epochs.
    for epoch in range(n_epochs):

        # Train for one epoch, printing every N iterations.
        train_one_epoch(model, optimizer, data_loader_train, device, epoch,
            print_freq = 100)

        # Update the learning rate.
        lr_scheduler.step()

        # Evaluate on the test dataset.
        evaluate(model, data_loader_valid, device = device)

    # Save the final model to disk.
    torch.save(model, './best_model.pth')
    print('Training done and model saved.')
