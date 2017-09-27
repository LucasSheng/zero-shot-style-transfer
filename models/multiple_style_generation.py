from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models import custom_ops
from models import losses
from models import preprocessing

slim = tf.contrib.slim


class MSG(object):
    def __init__(self, options):
        """gather necessary parameters for the multiple style generation"""
        self.weight_decay = options.get('weight_decay')
        self.default_size = options.get('default_size')
        self.content_size = options.get('content_size')
        self.style_size = options.get('style_size')

        # loss network architecture
        self.network_name = options.get('network_name')

        # the loss layers for the content and style similarity
        self.style_loss_layers = options.get('style_loss_layers')
        self.content_loss_layers = options.get('content_loss_layers')

        # the weights for losses
        self.content_weight = options.get('content_weight')
        self.style_weight = options.get('style_weight')
        self.tv_weight = options.get('tv_weight')

        # initialize the training quantities
        self.summaries = None
        self.train_op = None

        self.total_loss = 0.0
        self.content_loss = None
        self.style_loss = None
        self.tv_loss = None

    def style_transfer(self, inputs, styles, reuse=False):
        """style transfer via inspiration layers"""
        # encode the input content images
        content_net = self.feature_extractor(inputs, reuse=reuse)
        style_net = self.feature_extractor(styles, reuse=True)
        # add the inspiration
        net = inspiration(content_net, style_net, kernel_size=256, reuse=reuse)
        # add the preprocessing
        net = self.internal_processing(net, 6, reuse=reuse)
        # add the decoder
        outputs = self.feature_decoder(net, reuse=reuse)
        return outputs * 150 + 127.5

    def feature_extractor(self, inputs, reuse=False):
        """feature extractor for the inputs"""
        with tf.variable_scope('feature_extractor', reuse=reuse):
            with slim.arg_scope(self.msg_arg_scope()):
                net = tf.div(inputs, 127.5)  # normalize the input features
                net = custom_ops.conv2d_same(net, 64, 7, 1, scope='enc_0')
                net = custom_ops.residual_block_downsample(net, 128, 2, scope='enc_1')
                net = custom_ops.residual_block_downsample(net, 256, 2, scope='enc_2')
                return net

    def feature_decoder(self, inputs, reuse=False):
        """feature decoder of the inputs"""
        with tf.variable_scope('feature_decoder', reuse=reuse):
            with slim.arg_scope(self.msg_arg_scope()):
                net = custom_ops.residual_block_upsample(inputs, 128, 2, scope='dec_2')
                net = custom_ops.residual_block_upsample(net, 64, 2, scope='dec_1')
                # output processing
                outputs = slim.layer_norm(
                    net, activation_fn=tf.nn.relu, scope='dec_0_preact')
                with slim.arg_scope([slim.conv2d],
                                    normalizer_fn=None, activation_fn=tf.tanh):
                    outputs = custom_ops.conv2d_same(outputs, 3, 7, 1, scope='dec_0')
                    return outputs

    def internal_processing(self, inputs, num_blocks, reuse=False):
        """feature processing of the inputs"""
        with tf.variable_scope('feature_processing', reuse=reuse):
            with slim.arg_scope(self.msg_arg_scope()):
                return slim.repeat(inputs, num_blocks,
                                   custom_ops.residual_block_downsample,
                                   256, 1, scope='residual_block')

    def build_model(self, inputs, styles, reuse=False):
        """build the graph for the MSG model

        Args:
        inputs: the inputs [batch_size, height, width, channel]
        styles: the styles [1, height, width, channel]
        reuse: whether to reuse the parameters

        Returns:
        total_loss: the total loss for the style transfer
        """
        # extract the content features for the inputs
        inputs_image_features = losses.extract_image_features(
            inputs, self.network_name)
        inputs_content_features = losses.compute_content_features(
            inputs_image_features, self.content_loss_layers)

        # extract styles style features
        styles_image_features = losses.extract_image_features(
            styles, self.network_name)
        styles_style_features = losses.compute_style_features(
            styles_image_features, self.style_loss_layers)

        # transfer the styles from the inputs
        outputs = self.style_transfer(inputs, styles, reuse=reuse)

        # preprocessing the outputs to avoid biases and calculate the features
        outputs = preprocessing.batch_mean_image_subtraction(outputs)
        outputs_content_features, outputs_style_features = \
            losses.compute_content_and_style_features(
                outputs, self.network_name,
                self.content_loss_layers, self.style_loss_layers)

        # gather the summary operations
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # calculate the losses
        # the content loss
        if self.content_weight > 0.0:
            self.content_loss = losses.compute_content_loss(
                inputs_content_features, outputs_content_features, self.content_loss_layers)
            self.total_loss += self.content_weight * self.content_loss
            summaries.add(tf.summary.scalar('losses/content_loss', self.content_loss))
        # the style loss
        if self.style_weight > 0.0:
            self.style_loss = losses.compute_style_loss(
                styles_style_features, outputs_style_features, self.style_loss_layers)
            self.total_loss += self.style_weight * self.style_loss
            summaries.add(tf.summary.scalar('losses/style_loss', self.style_loss))
        # the total variation loss
        if self.tv_weight > 0.0:
            self.tv_loss = losses.compute_total_variation_loss(outputs)
            self.total_loss += self.tv_weight * self.tv_loss
            summaries.add(tf.summary.scalar('losses/tv_loss', self.tv_loss))

        summaries.add(tf.summary.scalar('total_loss', self.total_loss))

        # gather the image tiles for style transfer
        image_tiles = tf.concat([inputs, outputs], axis=2)
        image_tiles = preprocessing.batch_mean_image_summation(image_tiles)
        image_tiles = tf.cast(tf.clip_by_value(image_tiles, 0.0, 255.0), tf.uint8)
        summaries.add(tf.summary.image('style_results', image_tiles, max_outputs=8))

        # gather the styles
        summaries.add(tf.summary.image('styles',
                                       preprocessing.batch_mean_image_summation(styles),
                                       max_outputs=8))
        # gather the summaries
        self.summaries = summaries
        return self.total_loss

    def get_training_operations(self, optimizer, global_step,
                                variables_to_train=tf.trainable_variables()):
        # gather the variable summaries
        variables_summaries = \
            custom_ops.add_trainable_variables_summaries(variables_to_train)

        # add the training operations
        train_ops = []

        grads_and_vars = optimizer.compute_gradients(
            self.total_loss, var_list=variables_to_train)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)
        train_ops.append(train_op)

        # gather the summaries
        self.summaries |= variables_summaries
        self.train_op = tf.group(*train_ops)
        return self.train_op

    def msg_arg_scope(self):
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.layer_norm):
            with slim.arg_scope([custom_ops.residual_block_downsample, custom_ops.residual_block_upsample],
                                normalizer_fn=slim.layer_norm, activation_fn=tf.nn.relu) as arg_sc:
                return arg_sc


def inspiration(inputs_features, style_features, kernel_size, reuse=False):
    """inspiration layer for the msg-net"""
    with tf.variable_scope('inspiration', reuse=reuse):
        # affine transform of the input content feature
        with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
            outputs = custom_ops.conv2d_same(inputs_features, kernel_size, 1, 1, scope='affine_matrix')

        # multiply with the style statistics
        style_feature = losses.compute_gram_matrix(style_features)
        style_feature = tf.expand_dims(style_feature, axis=0)
        outputs = tf.nn.conv2d(outputs, style_feature, [1, 1, 1, 1], padding='SAME')
        outputs.set_shape(shape=inputs_features.get_shape())
        return outputs
