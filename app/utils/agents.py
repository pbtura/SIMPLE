import sys

import gym
import numpy as np
import torch
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor

np.set_printoptions(threshold=sys.maxsize)
import random
import string

import app.config as Config

from stable_baselines3.common import logger as lg

def sample_action(action_probs):
    action = np.random.choice(len(action_probs), p = action_probs)
    return action


def mask_actions(legal_actions, action_probs):
    masked_action_probs = np.multiply(legal_actions, action_probs)
    masked_action_probs = masked_action_probs / np.sum(masked_action_probs)
    return masked_action_probs





class Agent():
  model: OnPolicyAlgorithm

  def __init__(self, name, model: OnPolicyAlgorithm = None):
      self.name = name
      self.id = self.name + '_' + ''.join(random.choice(string.ascii_lowercase) for x in range(5))
      self.model = model
      self.points = 0

  def print_top_actions(self, action_probs):
    top5_action: tuple = torch.topk(action_probs, 5, sorted=True)
    top5_action_idx = top5_action[1]
    top5_action_values = top5_action[0]
    for i, a in zip(top5_action_idx, top5_action_values):
        Config.logger.debug(f"Top 5 actions: {i.item()} ': ' {a.item():.2f}")

  def choose_action(self, env: gym.Env, choose_best_action, mask_invalid_actions):
      if self.name == 'rules':
        action_probs = np.array(env.rules_move())
        value = None
      else:
        # get a tensor of the observation space
        obs = self.model.policy.obs_to_tensor(env.observation)[0]
        dist: Distribution = self.model.policy.get_distribution(obs)
        probs = dist.distribution.probs
        # value = self.model.policy.value(np.array([env.observation]))[0]
        # Config.logger.debug(f'Value {value:.2f}')
        action_probs = probs.flatten()

      self.print_top_actions(action_probs)
      
      if mask_invalid_actions:
        action_probs = mask_actions(env.legal_actions, action_probs)
        Config.logger.debug('Masked ->')
        self.print_top_actions(action_probs)
        
      action = torch.max(action_probs).item()
      Config.logger.debug(f'Best action {action}')

      if not choose_best_action:
          action = env.action_space.sample()
          Config.logger.debug(f'Sampled action {action} chosen')

      return action



