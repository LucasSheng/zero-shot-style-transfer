from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import yaml

from models import adaptive_instance_normalization
from models import multiple_style_generation
from models import recursive_feature_transforms
from models import patch_swapper
from models import auto_encoder

slim = tf.contrib.slim

models_map = {
    'AdaIN': adaptive_instance_normalization.AdaIN,
    'MSG': multiple_style_generation.MSG,
    'RFT': recursive_feature_transforms.RecursiveFeatureTransforms,
    'PS': patch_swapper.PatchSwapper,
    'AE': auto_encoder.AutoEncoder,
}


def get_model(filename):
    if not tf.gfile.Exists(filename):
        raise ValueError('The config file [%s] does not exist.' % filename)

    with open(filename, 'rb') as f:
        options = yaml.load(f)
        model_name = options.get('model_name')
        print('Finish loading the model [%s] configuration' % model_name)
        if model_name not in models_map:
            raise ValueError('Name of model [%s] unknown' % model_name)
        model = models_map[model_name](options)
        return model, options
