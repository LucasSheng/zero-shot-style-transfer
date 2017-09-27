from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from models import custom_ops
from models import losses
from models import preprocessing
from models import vgg

slim = tf.contrib.slim

network_map = {
    'vgg_16': vgg.vgg_16,
    'vgg_19': vgg.vgg_19,
}


class AdaIN(object):
    def __init__(self, options):
        """gather necessary parameters for the AdaIN architecture"""
        self.weight_decay = options.get('weight_decay')
        self.default_size = options.get('default_size')
        self.content_size = options.get('content_size')
        self.style_size = options.get('style_size')

        # loss network architecture
        self.network_name = options.get('network_name')

        # the loss layers for content and style similarity
        self.style_loss_layers = options.get('style_loss_layers')
        self.content_loss_layers = options.get('content_loss_layers')

        # the weights for losses
        self.style_weight = options.get('style_weight')
        self.content_weight = options.get('content_weight')
        self.tv_weight = options.get('tv_weight')

        # initialize some training quantities
        self.total_loss = 0
        self.content_loss = None
        self.style_loss = None
        self.tv_loss = None
        self.weight_loss = None

        # initialize the summaries and training_op
        self.summaries = None
        self.train_op = None

    def style_transfer(self, inputs, styles, reuse=False):
        """style transfer with the pipeline: Encode -> AdaIn -> Decode"""
        # image encoder
        with slim.arg_scope(vgg.vgg_arg_scope(self.weight_decay)):
            _, content_points = network_map[self.network_name](
                inputs, spatial_squeeze=False, is_training=False, reuse=reuse)
            content_label = content_points.keys()[0].split('/')[0]
            content_net = content_points[content_label + '/' + self.content_loss_layers[-1]]

        with slim.arg_scope(vgg.vgg_arg_scope(self.weight_decay)):
            _, style_points = network_map[self.network_name](
                styles, spatial_squeeze=False, is_training=False, reuse=True)
            style_label = style_points.keys()[0].split('/')[0]
            style_net = style_points[style_label + '/' + self.content_loss_layers[-1]]

        # adaptive instance normalization
        content_feature = adaptive_instance_normalization(content_net, style_net)
        content_features = {self.content_loss_layers[-1]: content_feature}

        # image decoder
        with tf.variable_scope('image_decoder', values=[content_feature], reuse=reuse):
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_regularizer=slim.l2_regularizer(self.weight_decay),
                    normalizer_fn=slim.layer_norm,
                    activation_fn=tf.nn.relu):
                # mimic conv4_1
                net = custom_ops.conv2d_same(content_feature, 512, 3, 1, scope='conv1')
                # mimic conv3_4 + max_pool2d
                net = custom_ops.conv2d_resize(net, 256, 3, 2, scope='deconv1')
                # mimic conv3_3 + conv3_2 + conv3_1
                net = slim.repeat(
                    net, 3, custom_ops.conv2d_same, 256, 3, 1, scope='conv2')
                # mimic conv2_2 + max_pool2d
                net = custom_ops.conv2d_resize(net, 128, 3, 2, scope='deconv2')
                # mimic conv2_1
                net = custom_ops.conv2d_same(net, 128, 3, 1, scope='conv3')
                # mimic conv1_2 + max_pool2d
                net = custom_ops.conv2d_resize(net, 64, 3, 2, scope='deconv3')
                # mimic conv1_1
                net = custom_ops.conv2d_same(net, 64, 3, 1, scope='conv4')
                # get the output
                with slim.arg_scope([slim.conv2d], activation_fn=tf.tanh, normalizer_fn=None):
                    outputs = custom_ops.conv2d_same(net, 3, 3, 1, scope='output')

        # output the image and hidden variables
        return outputs * 150.0 + 127.5, content_features

    def build_model(self, inputs, styles):
        # style transfer to the inputs
        outputs, inputs_content_features = self.style_transfer(inputs, styles)

        # calculate the style features for the outputs
        outputs = preprocessing.batch_mean_image_subtraction(outputs)
        # use approximated style loss instead
        # outputs_content_features, outputs_style_features = \
        #     losses.compute_content_and_style_features(
        #         outputs, self.network_name,
        #         self.content_loss_layers, self.style_loss_layers)
        outputs_image_features = losses.extract_image_features(outputs, self.network_name)
        outputs_content_features = losses.compute_content_features(
            outputs_image_features, self.content_loss_layers)
        outputs_style_features = losses.compute_approximate_style_features(
            outputs_image_features, self.style_loss_layers)

        # styles style features
        styles_image_features = losses.extract_image_features(
            styles, self.network_name)

        # use approximated style features instead
        # styles_style_features = losses.compute_style_features(
        #     styles_image_features, self.style_loss_layers)
        styles_style_features = losses.compute_approximate_style_features(
            styles_image_features, self.style_loss_layers)

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
            # use approximated style features instead
            # self.style_loss = losses.compute_style_loss(
            #     styles_style_features, outputs_style_features, self.style_loss_layers)
            self.style_loss = losses.compute_approximate_style_loss(
                styles_style_features, outputs_style_features, self.style_loss_layers)
            self.total_loss += self.style_weight * self.style_loss
            summaries.add(tf.summary.scalar('losses/style_loss', self.style_loss))
        # the total weight loss
        if self.tv_weight > 0.0:
            self.tv_loss = losses.compute_total_variation_loss(outputs)
            self.total_loss += self.tv_weight * self.tv_loss
            summaries.add(tf.summary.scalar('losses/tv_loss', self.tv_loss))
        # the weight regularization loss
        if self.weight_decay > 0.0:
            self.weight_loss = tf.add_n(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
                name='weight_loss')
            self.total_loss += self.weight_loss
            summaries.add(tf.summary.scalar('losses/weight_loss', self.weight_loss))

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

        # gather gradient summaries
        grads_summaries = custom_ops.add_gradients_summaries(grads_and_vars)

        # gather the summaries
        self.summaries = (variables_summaries | grads_summaries)
        self.train_op = tf.group(*train_ops)
        return self.train_op


def adaptive_instance_normalization(inputs, styles, epsilon=1e-5):
    """adaptive instance normalization given the inputs and the styles

    Args:
        inputs: input feature maps [batch_size, height, width, channel]
        styles: style feature maps [1, height, width, channel]
                                or [batch_size, height, width, channel]
        epsilon: 1e-5 to prevent bad values
    Returns:
        the adaptive instance normalized features
    """
    input_mean, input_var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
    style_mean, style_var = tf.nn.moments(styles, [1, 2], keep_dims=True)
    normalized_inputs = tf.div(
        tf.subtract(inputs, input_mean), tf.sqrt(tf.add(input_var, epsilon)))
    adaptive_inputs = tf.add(
        tf.multiply(tf.sqrt(style_var), normalized_inputs), style_mean)
    return adaptive_inputs
