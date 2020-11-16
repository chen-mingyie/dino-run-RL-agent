import gym
import os
import gym_chrome_dino
from stable_baselines import DQN
from stable_baselines import ACKTR
from gym_chrome_dino.utils.wrappers import make_dino
from stable_baselines.bench import Monitor
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import TRPO
from stable_baselines import PPO1
from rl_trainers.utils import save_model_callback
from rl_trainers.utils import evaluate_model

# Set log directory
LOG_DIR = "Logs"
ENV_ID = 'ChromeDino-v0'

def dqn(pretrained_model_name: str = ""):
    logpath = os.path.join(LOG_DIR, 'dqn')

    # Create env
    env = gym.make(ENV_ID)
    env = make_dino(env, timer=True, frame_stack=True)
    env = Monitor(env, logpath)

    # Train DQN
    if len(pretrained_model_name) == 0:
        model = DQN('CnnPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)
    else:
        model = DQN.load(os.path.join(logpath, pretrained_model_name), exploration_initial_eps=0.02)
        model.env = env
    callback = save_model_callback(check_freq=1000, log_dir=logpath)
    model.learn(total_timesteps=25000, log_interval=10, callback=callback) # Train the agent
    evaluate_model(model)

def trpo(pretrained_model_name: str = ""):
    logpath = os.path.join(LOG_DIR, 'trpo')

    # Create env
    env = gym.make(ENV_ID)
    env = make_dino(env, timer=True, frame_stack=True)
    env = Monitor(env, logpath)

    # Train model from scratch

    if len(pretrained_model_name) == 0:
        model = TRPO('CnnPolicy', env, verbose=1)
    else:
        model = TRPO.load(os.path.join(logpath, pretrained_model_name))
        model.env = env
    callback = save_model_callback(check_freq=1000, log_dir=logpath)
    model.learn(total_timesteps=25000, log_interval=10, callback=callback) # Train the agent
    evaluate_model(model)

def ppo1(pretrained_model_name: str = ""):
    logpath = os.path.join(LOG_DIR, 'ppo1')

    # Create env
    env = gym.make(ENV_ID)
    env = make_dino(env, timer=True, frame_stack=True)
    env = Monitor(env, logpath)

    # Train model from scratch
    if len(pretrained_model_name) == 0:
        model = PPO1('CnnPolicy', env, verbose=1)
    else:
        model = PPO1.load(os.path.join(logpath, pretrained_model_name))
        model.env = env
    callback = save_model_callback(check_freq=1000, log_dir=logpath)
    model.learn(total_timesteps=25000, log_interval=10, callback=callback) # Train the agent
    evaluate_model(model)