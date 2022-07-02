import torch
import numpy as np

from environments import PendulumEnv, D4RLEnv


# Evaluate agent with deterministic policy π
def evaluate_agent(agent, num_episodes, env, return_trajectories=False, render=False):
  returns, trajectories = [], []

  if render:
    imgs = []

  with torch.inference_mode():
    for _ in range(num_episodes):
      states, actions, rewards = [], [], []
      state, terminal = env.reset(), False
      while not terminal:
        action = agent.get_greedy_action(state)  # Take greedy action
        next_state, reward, terminal = env.step(action)

        if return_trajectories:
          states.append(state)
          actions.append(action)
        rewards.append(reward)
        state = next_state
        if render:
          imgs.append(env.render(mode='rgb_array'))
      returns.append(sum(rewards))

      if return_trajectories:
        # Collect trajectory data (including terminal signal, which may be needed for offline learning)
        terminals = torch.cat([torch.zeros(len(rewards) - 1), torch.ones(1)])
        trajectories.append({'states': torch.cat(states), 'actions': torch.cat(
          actions), 'rewards': torch.tensor(rewards, dtype=torch.float32), 'terminals': terminals})
      if render:
        import skvideo.io
        skvideo.io.vwrite('example.mp4', np.array(imgs))

  env.close()
  return (returns, trajectories) if return_trajectories else returns
