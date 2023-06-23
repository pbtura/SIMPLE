from typing import Optional, List, Union, Dict, Type, Any

import gym
import numpy as np
import tensorflow as tf
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.layers import BatchNormalization, Activation, Flatten, Conv2D, Add, Dense, Dropout

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution


class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,):
        super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn, ortho_init, use_sde, log_std_init, full_std, sde_net_arch, use_expln, squash_output, features_extractor_class, features_extractor_kwargs, normalize_images, optimizer_class, optimizer_kwargs)

        with tf.variable_scope("model"):
            
            extracted_features = resnet_extractor(self.processed_obs)
            self._policy = policy_head(extracted_features)
            self._value_fn, self.q_value = value_head(extracted_features)

            self._proba_distribution  = CategoricalDistribution(self._policy)

            
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})



def value_head(y):
    y = convolutional(y, 4, 1)
    y = Flatten()(y)
    y = dense(y, 128, batch_norm = False)
    vf = dense(y, 1, batch_norm = False, activation = 'tanh', name='vf')
    q = dense(y, 7, batch_norm = False, activation = 'tanh', name='q')
    return vf, q


def policy_head(y):
    y = convolutional(y, 4, 1)
    y = Flatten()(y)
    policy = dense(y, 7, batch_norm = False, activation = None, name='pi')
    return policy


def resnet_extractor(y, **kwargs):

    y = convolutional(y, 32, 4)
    y = residual(y, 32, 4)
    y = residual(y, 32, 4)
    y = residual(y, 32, 4)

    return y



def convolutional(y, filters, kernel_size):
    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization(momentum = 0.9)(y)
    y = Activation('relu')(y)
    return y

def residual(y, filters, kernel_size):
    shortcut = y

    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization(momentum = 0.9)(y)
    y = Activation('relu')(y)

    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization(momentum = 0.9)(y)
    y = Add()([shortcut, y])
    y = Activation('relu')(y)

    return y


def dense(y, filters, batch_norm = True, activation = 'relu', name = None):

    if batch_norm or activation:
        y = Dense(filters)(y)
    else:
        y = Dense(filters, name = name)(y)
    
    if batch_norm:
        if activation:
            y = BatchNormalization(momentum = 0.9)(y)
        else:
            y = BatchNormalization(momentum = 0.9, name = name)(y)

    if activation:
        y = Activation(activation, name = name)(y)
    
    return y


