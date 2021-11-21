# -*- coding: utf-8 -*-
"""Utils for pytorch models."""
import albumentations as album
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_training_augmentation():

    train_transform = [
        album.PadIfNeeded(min_height = 128, min_width = 128, p = 1),
        #album.RandomCrop(height=64, width=64, always_apply=True),
        album.HorizontalFlip(p = 0.5),
        album.VerticalFlip(p = 0.5),
        album.Lambda(image = preprocessing_fn),
        album.Lambda(image = to_tensor, mask = to_tensor)
    ]

    return album.Compose(train_transform)

def get_validation_augmentation():

    valid_transform = [
        album.PadIfNeeded(min_height = 128, min_width = 128, p = 1)
        #album.CenterCrop(height=64, width=64, always_apply=True),
    ]

    return album.Compose(valid_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    """Construct a preprocessing transform.
    Args:
        preprocessing_fn (callable): data normalization function (can be
            specific for each pretrained neural network)
    Returns:
        transform: albumentations.Compose
    """
    _transform = [
        album.PadIfNeeded(min_height = 128, min_width = 128, p = 1)
    ]
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)
