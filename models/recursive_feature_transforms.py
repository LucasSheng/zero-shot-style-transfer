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


class RecursiveFeatureTransforms(object):
    def __init__(self, options):
        self.default_size = options.get('default_size')
        self.content_size = options.get('content_size')
        self.style_size = options.get('style_size')

        # network architecture
        self.network_name = options.get('network_name')

        # the loss layers for content and style similarity
        self.style_loss_layers = options.get('style_loss_layers')

        # additional arguments
        self.is_unbiased_transform = options.get('is_unbiased_transform')
        self.blending_weight = options.get('blending_weight')

    def style_transfer(self, inputs, styles, style_layers=None):
        """style transfer via recursive feature transforms

        Args:
            inputs: input images [batch_size, height, width, channel]
            styles: input styles [1 or batch_size, height, width, channel]
            style_layers: a list of enforced style layer ids, default is None that
                applies all style layers as self.style_loss_layers

        Returns:
            outputs: the stylized images [batch_size, height, width, channel]
        """
        # get the style features for the styles
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

        outputs = tf.identity(inputs)
        intermediate_results = []
        num_modules = len(selected_style_layers)
        for i in range(num_modules, 0, -1):
            starting_layer = selected_style_layers[i-1]
            # encoding the inputs
            contents_image_features = losses.extract_image_features(
                outputs, self.network_name)
            content_features = losses.compute_content_features(
                contents_image_features, [starting_layer])

            # feature transformation
            if self.is_unbiased_transform:
                transformed_features = unbiased_feature_transform(
                    content_features[starting_layer], styles_features[starting_layer])
            else:
                transformed_features = feature_transform(
                    content_features[starting_layer], styles_features[starting_layer])
            if self.blending_weight:
                transformed_features = self.blending_weight * transformed_features + \
                                       (1-self.blending_weight) * content_features[starting_layer]

            # decoding the contents
            with slim.arg_scope(vgg_decoder.vgg_decoder_arg_scope(self.weight_decay)):
                outputs = vgg_decoder.vgg_decoder(
                    transformed_features, self.network_name, starting_layer,
                    reuse=True, scope='decoder_%d' % style_layers[i-1])
            intermediate_results.append(outputs)
            outputs = preprocessing.batch_mean_image_subtraction(outputs)
            print('Finish the module of [%s]' % starting_layer)

        # recover the outputs
        return preprocessing.batch_mean_image_summation(outputs), intermediate_results

    def texture_generation(self, styles, style_layers=None):
        """texture generation via recursive feature transforms

        Args:
            styles: input styles [1 or batch_size, height, width, channel]
            style_layers: a list of enforced style layer ids, default is None that
              applies all style layers as self.style_loss_layers

        Returns:
            outputs: the generated textures [batch_size, height, width, channel]
        """
        # get the style features for the styles
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

        outputs = tf.random_normal(shape=tf.shape(styles), stddev=1.0)*127.5
        intermediate_results = []
        num_modules = len(selected_style_layers)
        for i in range(num_modules, 0, -1):
            starting_layer = selected_style_layers[i-1]
            # encoding the inputs
            contents_image_features = losses.extract_image_features(
                outputs, self.network_name)
            content_features = losses.compute_content_features(
                contents_image_features, [starting_layer])

            # feature transformation
            if self.is_unbiased_transform:
                transformed_features = unbiased_feature_transform(
                    content_features[starting_layer], styles_features[starting_layer])
            else:
                transformed_features = feature_transform(
                    content_features[starting_layer], styles_features[starting_layer])

            # decoding the contents
            with slim.arg_scope(vgg_decoder.vgg_decoder_arg_scope(self.weight_decay)):
                outputs = vgg_decoder.vgg_decoder(
                    transformed_features, self.network_name, starting_layer,
                    reuse=True, scope='decoder_%d' % style_layers[i-1])
            intermediate_results.append(outputs)
            outputs = preprocessing.batch_mean_image_subtraction(outputs)
            print('Finish the module of [%s]' % starting_layer)

        # recover the outputs
        return preprocessing.batch_mean_image_summation(outputs), intermediate_results


def unbiased_feature_transform(content_features, style_features):
    """unbiased feature transformation from content to style"""
    content_shape = tf.shape(content_features)
    style_shape = tf.shape(style_features)

    # mean features for style and content
    mean_content_features = tf.reduce_mean(
        content_features, axis=[1, 2], keep_dims=True)
    mean_style_features = tf.reduce_mean(
        style_features, axis=[1, 2], keep_dims=True)

    # get the unbiased content and style features
    unbiased_content_features = content_features - mean_content_features
    unbiased_content_features = tf.reshape(
        unbiased_content_features, shape=(content_shape[0], -1, content_shape[3]))
    unbiased_style_features = style_features - mean_style_features
    unbiased_style_features = tf.reshape(
        unbiased_style_features, shape=(style_shape[0], -1, style_shape[3]))

    # get the covariance matrices
    unbiased_content_gram = tf.matmul(
        unbiased_content_features, unbiased_content_features, transpose_a=True)
    unbiased_content_gram /= tf.reduce_prod(tf.cast(content_shape[1:], tf.float32))
    unbiased_style_gram = tf.matmul(
        unbiased_style_features, unbiased_style_features, transpose_a=True)
    unbiased_style_gram /= tf.reduce_prod(tf.cast(style_shape[1:], tf.float32))

    #################################
    # converting the feature spaces #
    #################################
    s_c, u_c, v_c = tf.svd(unbiased_content_gram, compute_uv=True)
    s_c = tf.expand_dims(s_c, axis=1)
    s_s, u_s, v_s = tf.svd(unbiased_style_gram, compute_uv=True)
    s_s = tf.expand_dims(s_s, axis=1)

    # normalized features
    normalized_features = tf.matmul(unbiased_content_features, u_c)
    normalized_features = tf.multiply(normalized_features, 1.0/(tf.sqrt(s_c+1e-5)))
    normalized_features = tf.matmul(normalized_features, v_c, transpose_b=True)

    # colorized features
    # broadcasting the tensors for matrix multiplication
    content_batch = tf.shape(u_c)[0]
    style_batch = tf.shape(u_s)[0]
    batch_multiplier = tf.cast(content_batch/style_batch, tf.int32)
    u_s = tf.tile(u_s, multiples=tf.stack([batch_multiplier, 1, 1]))
    v_s = tf.tile(v_s, multiples=tf.stack([batch_multiplier, 1, 1]))
    colorized_features = tf.matmul(normalized_features, u_s)
    colorized_features = tf.multiply(colorized_features, tf.sqrt(s_s+1e-5))
    colorized_features = tf.matmul(colorized_features, v_s, transpose_b=True)

    # reshape the colorized features
    colorized_features = tf.reshape(colorized_features, shape=content_shape)
    colorized_features += mean_style_features
    return colorized_features


def feature_transform(content_features, style_features):
    """feature transformation from content to style"""
    content_shape = tf.shape(content_features)
    style_shape = tf.shape(style_features)

    # get the unbiased content and style features
    content_features = tf.reshape(
        content_features, shape=(content_shape[0], -1, content_shape[3]))
    style_features = tf.reshape(
        style_features, shape=(style_shape[0], -1, style_shape[3]))

    # get the covariance matrices
    content_gram = tf.matmul(content_features, content_features, transpose_a=True)
    content_gram /= tf.reduce_prod(tf.cast(content_shape[1:], tf.float32))
    style_gram = tf.matmul(style_features, style_features, transpose_a=True)
    style_gram /= tf.reduce_prod(tf.cast(style_shape[1:], tf.float32))

    #################################
    # converting the feature spaces #
    #################################
    s_c, u_c, v_c = tf.svd(content_gram, compute_uv=True)
    s_c = tf.expand_dims(s_c, axis=1)
    s_s, u_s, v_s = tf.svd(style_gram, compute_uv=True)
    s_s = tf.expand_dims(s_s, axis=1)

    # normalized features
    normalized_features = tf.matmul(content_features, u_c)
    normalized_features = tf.multiply(normalized_features, 1.0/(tf.sqrt(s_c+1e-5)))
    normalized_features = tf.matmul(normalized_features, v_c, transpose_b=True)

    # colorized features
    # broadcasting the tensors for matrix multiplication
    content_batch = tf.shape(u_c)[0]
    style_batch = tf.shape(u_s)[0]
    batch_multiplier = tf.cast(content_batch/style_batch, tf.int32)
    u_s = tf.tile(u_s, multiples=tf.stack([batch_multiplier, 1, 1]))
    v_s = tf.tile(v_s, multiples=tf.stack([batch_multiplier, 1, 1]))
    colorized_features = tf.matmul(normalized_features, u_s)
    colorized_features = tf.multiply(colorized_features, tf.sqrt(s_s+1e-5))
    colorized_features = tf.matmul(colorized_features, v_s, transpose_b=True)

    # reshape the colorized features
    colorized_features = tf.reshape(colorized_features, shape=content_shape)
    return colorized_features
