from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from models import losses
from models import preprocessing
from models import vgg
from models import vgg_decoder

slim = tf.contrib.slim

network_map = {
    'vgg_16': vgg.vgg_16,
    'vgg_19': vgg.vgg_19,
  }


class PatchSwapper(object):
    def __init__(self, options):
        self.default_size = options.get('default_size')
        self.content_size = options.get('content_size')
        self.style_size = options.get('style_size')

        # network architecture
        self.network_name = options.get('network_name')

        # the loss layers for content and style similarity
        self.style_loss_layers = options.get('style_loss_layers')

        # window size for the patch swapper
        self.patch_size = options.get('patch_size')

    def style_transfer(self, inputs, styles, style_layers=(2,)):
        """style transfer via patch swapping

        Args:
            inputs: input images [batch_size, height, width, channel]
            styles: input styles [1, height, width, channel]
            style_layers: the list of layers to perform style swapping, default is None
                that applied all style layers as self.style_loss_layers

        Returns:
            outputs: the stylized images [batch_size, height, width, channel]s
        """
        styles_image_features = losses.extract_image_features(
            styles, self.network_name)
        styles_features = losses.compute_content_features(
            styles_image_features, self.style_loss_layers)

        # construct the recurrent modules
        selected_style_layers = self.style_loss_layers
        if style_layers:
            selected_style_layers = [selected_style_layers[i] for i in style_layers]
        else:
            style_layers = range(len(selected_style_layers))

        # input preprocessing
        outputs = tf.identity(inputs)
        # start style transfer
        num_modules = len(selected_style_layers)
        for i in range(num_modules, 0, -1):
            starting_layer = selected_style_layers[i-1]
            # encoding the inputs
            contents_image_features = losses.extract_image_features(
                outputs, self.network_name)
            contents_features = losses.compute_content_features(
                contents_image_features, [starting_layer])
            # feature transformation
            transformed_features = feature_transform(
                contents_features[starting_layer],
                styles_features[starting_layer],
                patch_size=self.patch_size)
            # decoding the contents
            with slim.arg_scope(vgg_decoder.vgg_decoder_arg_scope()):
                outputs = vgg_decoder.vgg_decoder(
                    transformed_features, self.network_name, starting_layer,
                    scope='decoder_%d' % style_layers[i-1])
            outputs = preprocessing.batch_mean_image_subtraction(outputs)
            print('Finish the module of [%s]' % starting_layer)

        # recover the outputs
        return preprocessing.batch_mean_image_summation(outputs)


def feature_transform(content_features, style_features, patch_size=3):
    """feature transformation by patch swapping"""
    # channels for both the content and style, must be the same
    c_shape = tf.shape(content_features)
    s_shape = tf.shape(style_features)
    channel_assertion = tf.Assert(
        tf.equal(c_shape[3], s_shape[3]), ['number of channels  must be the same'])

    with tf.control_dependencies([channel_assertion]):
        # spatial shapes for style and content features
        c_height, c_width, c_channel = c_shape[1], c_shape[2], c_shape[3]

        # convert the style features into convolutional kernels
        style_kernels = tf.extract_image_patches(
            style_features, ksizes=[1, patch_size, patch_size, 1],
            strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='VALID')
        style_kernels = tf.squeeze(style_kernels, axis=0)
        style_kernels = tf.transpose(style_kernels, perm=[2, 0, 1])

        # gather the conv and deconv kernels
        v_height, v_width = style_kernels.get_shape().as_list()[1:3]
        deconv_kernels = tf.reshape(
            style_kernels, shape=(patch_size, patch_size, c_channel, v_height*v_width))

        kernels_norm = tf.norm(style_kernels, axis=0, keep_dims=True)
        kernels_norm = tf.reshape(kernels_norm, shape=(1, 1, 1, v_height*v_width))

        # calculate the normalization factor
        mask = tf.ones((c_height, c_width), tf.float32)
        fullmask = tf.zeros((c_height+patch_size-1, c_width+patch_size-1), tf.float32)
        for x in range(patch_size):
            for y in range(patch_size):
                paddings = [[x, patch_size-x-1], [y, patch_size-y-1]]
                padded_mask = tf.pad(mask, paddings=paddings, mode="CONSTANT")
                fullmask += padded_mask
        pad_width = int((patch_size-1)/2)
        deconv_norm = tf.slice(fullmask, [pad_width, pad_width], [c_height, c_width])
        deconv_norm = tf.reshape(deconv_norm, shape=(1, c_height, c_width, 1))

        ########################
        # starting convolution #
        ########################
        # convolutional operations
        net = tf.nn.conv2d(
            content_features,
            tf.div(deconv_kernels, kernels_norm+1e-7),
            strides=[1, 1, 1, 1],
            padding='SAME')
        # find the maximum locations
        best_match_ids = tf.argmax(net, axis=3)
        best_match_ids = tf.cast(
            tf.one_hot(best_match_ids, depth=v_height*v_width), dtype=tf.float32)

        # find the patches and warping the output
        unnormalized_output = tf.nn.conv2d_transpose(
            value=best_match_ids,
            filter=deconv_kernels,
            output_shape=c_shape,
            strides=[1, 1, 1, 1],
            padding='SAME')
        output = tf.div(unnormalized_output, deconv_norm)
        output = tf.reshape(output, shape=c_shape)

        # output the swapped feature maps
        return output
