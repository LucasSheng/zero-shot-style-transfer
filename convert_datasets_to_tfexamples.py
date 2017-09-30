from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# number of shards per dataset split
_NUM_SHARDS = 32

# seed for repeatability
_RANDOM_SEED = 0

# the ratio of images in the train sets
_TRAIN_RATIO = 0.7


tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'The directory where the source files are stored.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'simple',
    'The name of the dataset as the prefix for the tfexample files, Eg., [simple] or [wikiart]')

tf.app.flags.DEFINE_float(
    'dataset_train_ratio', _TRAIN_RATIO,
    'The ratio of the dataset that is split into train subsets.')

tf.app.flags.DEFINE_integer(
    'num_shards', _NUM_SHARDS,
    'The number of shards for each split.')

tf.app.flags.DEFINE_string(
    'output_dir', None,
    'The directory where the outputs TFRecords and temporary files are saved.')

FLAGS = tf.app.flags.FLAGS


def _get_filenames():
    image_filenames = []
    if FLAGS.dataset_name == 'simple':
        # there are no subfolders in the dataset
        local_dataset_dir = os.path.join(FLAGS.dataset_dir)
        for filename in os.listdir(local_dataset_dir):
            file_path = os.path.join(local_dataset_dir, filename)
            image_filenames.append(file_path)
    elif FLAGS.dataset_name == 'wikiart':
        # there are subfolders of styles
        local_dataset_dir = os.path.join(FLAGS.dataset_dir)
        for style_dir in os.listdir(local_dataset_dir):
            style_path = os.path.join(local_dataset_dir, style_dir)
            for filename in os.listdir(style_path):
                file_path = os.path.join(style_path, filename)
                image_filenames.append(file_path)
    print('There are %d images in the [%s] dataset' % (len(image_filenames), FLAGS.dataset_name))
    return image_filenames


def _get_dataset_filename(split_name, shard_id):
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
        FLAGS.dataset_name, split_name, shard_id, FLAGS.num_shards)
    return os.path.join(FLAGS.output_dir, output_filename)


def _convert_dataset(split_name, image_filenames):
    assert split_name in ['train', 'validation', 'test']
    num_per_shard = int(math.ceil(len(image_filenames) / float(FLAGS.num_shards)))

    with tf.Graph().as_default():
        # image_reader = dataset_utils.ImageReader()
        image_reader = dataset_utils.ImageCoder()
        with tf.Session('') as sess:
            for shard_id in range(FLAGS.num_shards):
                output_filename = _get_dataset_filename(split_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(image_filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(image_filenames), shard_id))
                        sys.stdout.flush()
                        # read the image
                        img_filename = image_filenames[i]
                        img_data = tf.gfile.FastGFile(img_filename, 'r').read()
                        img_status, img_data, img_shape = image_reader.decode_image(sess, img_data)
                        if img_status and img_data is not None:
                            example = dataset_utils.image_to_tfexample(
                                img_data, 'jpg', img_shape, img_filename)
                            tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def _dataset_exists():
    for split_name in ['train', 'validation', 'test']:
        for shard_id in range(FLAGS.num_shards):
            output_filename = _get_dataset_filename(split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run():
    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    if _dataset_exists():
        print('Dataset files already exist. Exiting without re-creating them.')

    image_filenames = _get_filenames()

    # randomize the datasets
    random.seed(_RANDOM_SEED)
    random.shuffle(image_filenames)

    num_train = int(math.ceil(FLAGS.dataset_train_ratio * len(image_filenames)))
    num_test = len(image_filenames) - num_train

    train_image_filenames = image_filenames[:num_train]
    test_image_filenames = image_filenames[num_train:]

    # store the dataset meta data
    dataset_meta_data = {
        'dataset_name': FLAGS.dataset_name,
        'source_dataset_dir': FLAGS.dataset_dir,
        'num_of_samples': len(image_filenames),
        'num_of_train': num_train,
        'num_of_test': num_test,
        'train_image_filenames': train_image_filenames,
        'test_image_filenames': test_image_filenames,
    }
    dataset_utils.write_dataset_meta_data(FLAGS.output_dir, dataset_meta_data)

    _convert_dataset('train', train_image_filenames)
    _convert_dataset('test', test_image_filenames)


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    if not FLAGS.output_dir:
        # use the default output dirs
        FLAGS.output_dir = FLAGS.dataset_dir
    run()


if __name__ == '__main__':
    tf.app.run()
