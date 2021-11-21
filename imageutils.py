# -*- coding: utf-8 -*-
"""Utility functions for image processing."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def draw_image_with_boxes(filename, image, boxes):
    """Draws an image with boxes of detected objects."""

    # plot the image
    plt.figure()
    plt.imshow(image)

    # get the context for drawing boxes
    ax = plt.gca()

    # plot each box
    for box in boxes:

        # get coordinates
        x1, y1, x2, y2 = box

        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1

        # create the shape
        rect = Rectangle((x1, y1), width, height, fill = False, color = 'red')

        # draw the box
        ax.add_patch(rect)

    # Save the figure
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename, dpi = 300)
    plt.close()

def one_hot_encode(label, label_values):
    """ Converts a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes.
    """

    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

def reverse_one_hot(image):
    """Transforms a one-hot format to a 2D array with only 1 channel where each
    pixel value is the classified class key.
    """

    x = np.argmax(image, axis = -1)

    return x

def colour_code_segmentation(image, label_values):
    """Given a 1-channel array of class keys assigns colour codes."""

    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

def save_fig(figname, **images):
    """Saves a list of images to disk."""

    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]);
        plt.yticks([])
        plt.title(name.replace('_',' ').title(), fontsize = 20)
        plt.imshow(image)
    plt.savefig(f'{figname}.png')
    plt.close()
