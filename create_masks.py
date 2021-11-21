# -*- coding: utf-8 -*-
"""Creates integer-based masks from RLE data."""
from os import path
import pandas as pd
import numpy as np
from PIL import Image

DATA_DIR = '../Data/airbus'
MASK_DIR = path.join(DATA_DIR, 'train_masks')

def get_mask(encoded_pixels, dims):
    """Get integer-based mask from multiple run length encoded ships."""

    # Init mask with all 0-class pixels.
    mask = np.zeros(dims[0] * dims[1], dtype = np.uint8)

    for obj_id, curr_encoded in enumerate(encoded_pixels):

        s = curr_encoded.split()

        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            mask[start : start + length] = obj_id + 1

    return mask.reshape(dims).T


if __name__ == '__main__':

    # Load training metadata.
    metadata = path.join(DATA_DIR, 'train_ship_segmentations_v2.csv')
    df_metadata = pd.read_csv(metadata)

    # Corrupted images.
    exclude_list = ['6384c3e78.jpg', '13703f040.jpg', '14715c06d.jpg',
        '33e0ff2d5.jpg', '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg',
        'a8d99130e.jpg', 'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg',
        'dc3e7c901.jpg', 'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']

    for corrupted in exclude_list:
        df_metadata = df_metadata[~df_metadata.ImageId.str.contains(corrupted)]

    # Remove images without ships.
    df_metadata.dropna(inplace = True)

    n_ships = len(df_metadata)
    image_ids = np.unique(df_metadata.ImageId.tolist())
    print(f'There are {n_ships} ships across {len(image_ids)} images.')

    print('Starting to create masks...')

    for image_id in image_ids:

        fn_mask = path.join(MASK_DIR, image_id.replace('.jpg', '_mask.png'))

        samples = df_metadata[df_metadata['ImageId'] == image_id]
        encoded_pixels = samples.EncodedPixels.tolist()
        mask = get_mask(encoded_pixels, (768, 768))

        im = Image.fromarray(mask)
        im.save(fn_mask)

    print('Done.')
