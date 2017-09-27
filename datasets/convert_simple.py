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
_NUM_SHARDS = 1

# seed for repeatability
_RANDOM_SEED = 0

# the ratio of images in the train sets
_RATIO_TRAIN = 1.0


tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'The directory where the outputs TFRecords and temporary files are saved.')

FLAGS = tf.app.flags.FLAGS


def _get_filenames(dataset_dir):
    dataset_dir = os.path.join(dataset_dir, 'sources')
    image_filenames = []
    for filename in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, filename)
        image_filenames.append(file_path)
    print('Statistics in the [SIMPLE] dataset...')
    print('There are %d images in the dataset' % len(image_filenames))
    return image_filenames


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'SIMPLE_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, image_filenames, dataset_dir):
    assert split_name in ['train', 'validation', 'test']

    num_per_shard = int(math.ceil(len(image_filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = dataset_utils.ImageReader()

        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)
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
                        img_shape = image_reader.read_image_dims(sess, img_data)
                        example = dataset_utils.image_to_tfexample(
                            img_data, img_filename[-3:], img_shape, img_filename)
                        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation', 'test']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run(dataset_dir):
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')

    image_filenames = _get_filenames(dataset_dir)

    # randomize the datasets
    random.seed(_RANDOM_SEED)
    random.shuffle(image_filenames)

    num_train = int(math.ceil(_RATIO_TRAIN * len(image_filenames)))
    num_test = len(image_filenames) - num_train

    train_image_filenames = image_filenames[:num_train]
    test_image_filenames = image_filenames[num_train:]

    # store the dataset meta data
    dataset_meta_data = {
        'dataset_name': 'SIMPLE',
        'source_dataset_dir': dataset_dir,
        'num_of_samples': len(image_filenames),
        'num_of_train': num_train,
        'num_of_test': num_test,
        'train_image_filenames': train_image_filenames,
        'test_image_filenames': test_image_filenames,
    }
    dataset_utils.write_dataset_meta_data(dataset_dir, dataset_meta_data)

    _convert_dataset('train', train_image_filenames, dataset_dir)
    _convert_dataset('test', test_image_filenames, dataset_dir)


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    run(FLAGS.dataset_dir)


if __name__ == '__main__':
    tf.app.run()
