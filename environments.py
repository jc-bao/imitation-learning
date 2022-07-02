from logging import ERROR

import gym
import numpy as np
import torch
import h5py

from training import TransitionDataset

gym.logger.set_level(ERROR)  # Ignore warnings from Gym logger

D4RL_ENV_NAMES = ['ant-bullet-medium-v0', 'halfcheetah-bullet-medium-v0',
                  'hopper-bullet-medium-v0', 'walker2d-bullet-medium-v0']


def get_keys(h5file):
  keys = []
  def visitor(name, item):
    if isinstance(item, h5py.Dataset):
      keys.append(name)
  h5file.visititems(visitor)
  return keys


# Test environment for testing the code
class PendulumEnv():
  def __init__(self, env_name=''):
    self.env = gym.make('Pendulum-v0')
    self.env.action_space.high, self.env.action_space.low = torch.as_tensor(
      self.env.action_space.high), torch.as_tensor(self.env.action_space.low)  # Convert action space for action clipping

  def reset(self):
    state = self.env.reset()
    # Add batch dimension to state
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)

  def step(self, action):
    action = action.clamp(min=self.env.action_space.low,
                          max=self.env.action_space.high)  # Clip actions
    state, reward, terminal, _ = self.env.step(
      action[0].detach().numpy())  # Remove batch dimension from action
    # Add batch dimension to state
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, terminal

  def seed(self, seed):
    return self.env.seed(seed)

  def render(self, **kwargs):
    return self.env.render(**kwargs)

  def close(self):
    self.env.close()

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space

  def get_dataset(self, size=0, dtype=torch.float):
    return []


class D4RLEnv():
  def __init__(self, env):
    self.env = env
    self.env.action_space.high, self.env.action_space.low = torch.as_tensor(
      self.env.action_space.high), torch.as_tensor(self.env.action_space.low)  # Convert action space for action clipping

  def reset(self):
    state = self.env.reset()
    # Add batch dimension to state
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)

  def step(self, action):
    action = action.clamp(min=self.env.action_space.low,
                          max=self.env.action_space.high)  # Clip actions
    state, reward, terminal, _ = self.env.step(
      action[0].detach().numpy())  # Remove batch dimension from action
    # Add batch dimension to state
    return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0), reward, terminal

  def seed(self, seed):
    return self.env.seed(seed)

  def render(self, **kwargs):
    return self.env.render(**kwargs)

  def close(self):
    self.env.close()

  @property
  def observation_space(self):
    return self.env.observation_space

  @property
  def action_space(self):
    return self.env.action_space

  def get_dataset(self, size=0, subsample=20):
    dataset_file = h5py.File(
      '/juno/u/chaoyi/rl/dexterity-fork/examples/logs/10k_re.hdf5', 'r')
    data_dict = {k: dataset_file[k][:] for k in get_keys(dataset_file)}
    dataset_file.close()
    # Run a few quick sanity checks
    for key in ['observations', 'actions', 'rewards', 'terminals']:
      assert key in data_dict, f'Dataset is missing key {key}'
    N_samples = data_dict['observations'].shape[0]
    if self.observation_space.shape is not None:
      assert data_dict['observations'].shape[
          1:] == self.observation_space.shape, f"Observation shape does not match env: {str(data_dict['observations'].shape[1:])} vs {str(self.observation_space.shape)}"
    assert data_dict['actions'].shape[
        1:] == self.action_space.shape, f"Action shape does not match env: {str(data_dict['actions'].shape[1:])} vs {str(self.action_space.shape)}"
    if data_dict['rewards'].shape == (N_samples, 1):
      data_dict['rewards'] = data_dict['rewards'][:, 0]
    assert data_dict['rewards'].shape == (
        N_samples,
    ), f"Reward has wrong shape: {str(data_dict['rewards'].shape)}"
    if data_dict['terminals'].shape == (N_samples, 1):
      data_dict['terminals'] = data_dict['terminals'][:, 0]
    assert data_dict['terminals'].shape == (
        N_samples,
    ), f"Terminals has wrong shape: {str(data_dict['rewards'].shape)}"
    dataset = data_dict
    N = dataset['rewards'].shape[0]
    dataset_out = {'states': torch.as_tensor(dataset['observations'][:-1], dtype=torch.float32),
                   'actions': torch.as_tensor(dataset['actions'][:-1], dtype=torch.float32),
                   'rewards': torch.as_tensor(dataset['rewards'][:-1], dtype=torch.float32),
                   'next_states': torch.as_tensor(dataset['observations'][1:], dtype=torch.float32),
                   'terminals': torch.as_tensor(dataset['terminals'][:-1], dtype=torch.float32)}
    # Postprocess
    if size > 0 and size < N:
      for key in dataset_out:
        dataset_out[key] = dataset_out[key][:size]
    if subsample > 0:
      for key in dataset_out:
        dataset_out[key] = dataset_out[key][0::subsample]

    return TransitionDataset(dataset_out)


ENVS = {'ant': D4RLEnv, 'halfcheetah': D4RLEnv,
        'hopper': D4RLEnv, 'pendulum': PendulumEnv, 'walker2d': D4RLEnv}
