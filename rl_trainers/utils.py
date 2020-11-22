import gym
import os
import gym_chrome_dino
import numpy as np
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy
from gym_chrome_dino.utils.wrappers import make_dino
from stable_baselines.bench import Monitor
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import VecNormalize
import random

def make_dino_env(env_id, log_path):
    env = gym.make(env_id)
    env = Monitor(env, log_path)
    env = make_dino(env, timer=True, frame_stack=True)
    return env

def make_dino_vec_env(env_id, log_path, nbr_env = 5):
    env = make_vec_env(env_id, n_envs=nbr_env, monitor_dir=log_path, wrapper_class=make_dino)
    env = VecNormalize(env)
    return env

def evaluate_model(env, model, num_steps=10000):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_steps: (int) number of timesteps to evaluate it
    :return: (float) Mean reward for the last 100 episodes
    """
    episode_rewards = [0.0]
    episode_gamescore = []

    obs = env.reset()
    for i in range(num_steps):
        # _states are only useful when using LSTM policies
        action_prob = model.action_probability(obs)
        action = 1 if action_prob[1] > 0.999 else 0 # for no accleration
        # action, _states = model.predict(obs, deterministic=False)

        obs, reward, done, info = env.step(action)

        # Stats
        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
            # episode_gamescore.append(info['score'])
    # Compute mean reward for the last 100 episodes
    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    mean_100ep_score = round(np.mean(episode_gamescore[-100:]), 1)
    print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))
    print("Mean score:", mean_100ep_score, "Num complete games:", len(episode_gamescore))

    return episode_rewards, episode_gamescore

class save_model_callback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(save_model_callback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.gamescore = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        for env in self.training_env.envs:
            if env.game.is_crashed():
                self.gamescore.append(env.game.get_score())

        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                # Mean in-game score over past episodes
                mean_score = np.mean(self.gamescore)
                print("Mean score over last {:.2f} episodes: {:.2f}".format(len(self.gamescore), mean_score))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True