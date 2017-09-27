from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import os
import tensorflow as tf

from models import models_factory
from models import preprocessing
from utils import utils

slim = tf.contrib.slim


tf.app.flags.DEFINE_string(
    'checkpoint_dir', 'tmp/tfmodel',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', 'tmp/tfmodel',
    'Directory where the results are saved to.')

tf.app.flags.DEFINE_string(
    'style_dataset_dir', None,
    'The style directorty where the style images are stored.')

# choose the model configuration file
tf.app.flags.DEFINE_string(
    'model_config_path', None,
    'The path of the model configuration file.')

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.style_dataset_dir:
        raise ValueError('You must supply the style dataset directory '
                         'with --style_dataset_dir')

    if not FLAGS.checkpoint_dir:
        raise ValueError('You must supply the checkpoints directory with '
                         '--checkpoint_dir')

    if tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        checkpoint_dir = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    else:
        checkpoint_dir = FLAGS.checkpoint_dir

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # define the model
        style_model, options = models_factory.get_model(FLAGS.model_config_path)

        # predict the stylized image
        inp_style_image = tf.placeholder(tf.float32, shape=(None, None, 3))
        # style_image = preprocessing.mean_image_subtraction(inp_style_image)
        style_image = preprocessing.preprocessing_image(
            inp_style_image, 224, 224, resize_side=256, is_training=False)
        style_image = tf.expand_dims(style_image, axis=0)

        # style transfer with the randomized content
        stylized_image, _ = style_model.texture_generation(style_image)
        stylized_image = tf.squeeze(stylized_image, axis=0)

        # gather the test image filenames and style image filenames
        styles_categories = utils.get_image_filenames(FLAGS.style_dataset_dir)

        # starting inference of the images
        init_fn = slim.assign_from_checkpoint_fn(
            checkpoint_dir, slim.get_model_variables(), ignore_missing_vars=True)
        with tf.Session() as sess:
            # initialize the graph
            init_fn(sess)

            # style transfer for each image based on one style image
            for style_category in styles_categories:
                # gather the storage folder for the style transfer
                category = style_category.split('/')[-1]
                style_dir = os.path.join(FLAGS.eval_dir, category)
                if not tf.gfile.Exists(style_dir):
                    tf.gfile.MakeDirs(style_dir)
                sys.stdout.write('>> Transferring the texture [%s] ' % category)

                style_image_filenames = utils.get_image_filenames(style_category)
                for i in range(len(style_image_filenames)):
                    np_style_image = utils.image_reader(style_image_filenames[i])

                    np_stylized_image = sess.run(
                      stylized_image, feed_dict={inp_style_image: np_style_image})
                    output_filename = os.path.join(
                      style_dir, style_image_filenames[i].split('/')[-1])
                    utils.imsave(output_filename, np_stylized_image)
                    sys.stdout.write('\r>> Transferring image %d/%d' % (
                      i+1, len(style_image_filenames)))
                    sys.stdout.flush()


if __name__ == '__main__':
    tf.app.run()
