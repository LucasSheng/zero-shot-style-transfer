from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import scipy.misc
import yaml
import os

from PIL import Image

slim = tf.contrib.slim


def tf_image_reader(filename):
    """help fn that provides tensorflow image coding utilities"""
    if not tf.gfile.Exists(filename):
        raise ValueError('The style image [%s] does not exist' % filename)
    raw_bytes = tf.gfile.FastGFile(filename, 'r').read()
    decoded_image = tf.image.decode_image(raw_bytes, channels=3)
    decoded_image = tf.to_float(decoded_image)
    return decoded_image


def image_reader(filename):
    """help fn that provides numpy image coding utilities"""
    img = scipy.misc.imread(filename).astype(np.float)
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def imsave(filename, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(filename, quality=95)


def get_image_filenames(dataset_dir):
    """helper fn that provides the full image filenames from the dataset_dir"""
    image_filenames = []
    for filename in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, filename)
        image_filenames.append(file_path)
    return image_filenames


def get_options_from_config(filename):
    """helper fn that parses options from a yml file"""
    if not tf.gfile.Exists(filename):
        raise ValueError('The config file [%s] does not exist' % filename)
    with open(filename, 'rb') as f:
        options = yaml.load(f)
    return options
