import os
import gym
import gym_chrome_dino
from gym_chrome_dino.utils.wrappers import make_dino
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import TRPO, HER, PPO1, PPO2, DQN, ACKTR, A2C
from rl_trainers.utils import save_model_callback, evaluate_model, make_dino_env, make_dino_vec_env
from stable_baselines.common.evaluation import evaluate_policy

# Set log directory
LOG_DIR = "Logs"
ENV_ID = 'ChromeDino-v0'
# ENV_ID = 'ChromeDinoNoBrowser-v0'
# ENV_ID = 'ChromeDinoHandrafted-v0'

def dqn_dinorunner():
    logpath = os.path.join(LOG_DIR, 'dqn_dinorunner')

    # Create env
    env = make_dino_env(ENV_ID, logpath)

    # Train DQN
    model = DQN.load(os.path.join(logpath, 'best_model.zip'), env=env, exploration_initial_eps=0.02)
    model.learn(total_timesteps=25000, log_interval=10) # Train the agent

def dqn_flappybird():
    logpath = os.path.join(LOG_DIR, 'flappybird_dinorunner')

    # Create env
    env = make_dino_env(ENV_ID, logpath)

    # Train DQN
    model = DQN.load(os.path.join(logpath, 'best_model.zip'), env=env, exploration_initial_eps=0.02)
    model.learn(total_timesteps=25000, log_interval=10) # Train the agent

def ddqn_dinorunner():
    logpath = os.path.join(LOG_DIR, 'ddqn_dinorunner')

    # Create env
    env = make_dino_env(ENV_ID, logpath)

    # Train DQN
    model = DQN.load(os.path.join(logpath, 'best_model.zip'), env=env, exploration_initial_eps=0.02, double_q=True)
    model.learn(total_timesteps=25000, log_interval=10) # Train the agent

def ppo_dinorunner():
    logpath = os.path.join(LOG_DIR, 'ppo_dinorunner')

    # Create Env
    # env = make_dino_vec_env(ENV_ID, logpath, nbr_env=5)
    env = make_vec_env('ChromeDinoHandrafted-v0', n_envs=5, monitor_dir=logpath)

    # Train model
    model = PPO2.load(os.path.join(logpath, 'best_model.zip'), env=env)
    model.learn(total_timesteps=int(1e6), log_interval=50) # Train the agent

def ppo_dinorunner_fullgame():
    logpath = os.path.join(LOG_DIR, 'ppo_dinorunner_fullgame')

    # Create Env
    env = make_vec_env('ChromeDinoHandrafted_accl-v0', n_envs=5, monitor_dir=logpath)

    # Train model
    model = PPO2.load(os.path.join(logpath, 'best_model.zip'), env=env)
    model.learn(total_timesteps=int(1e6), log_interval=50) # Train the agent

def ppo2(pretrained_model_name: str = ""):
    logpath = os.path.join(LOG_DIR, 'ppo2')

    # Create Env
    # env = make_dino_vec_env(ENV_ID, logpath, nbr_env=5)
    env = make_vec_env('ChromeDinoHandrafted_accl-v0', n_envs=5, monitor_dir=logpath)
    # env = make_vec_env('ChromeDinoHandrafted-v0', n_envs=5, monitor_dir=logpath)

    # Train model from scratch
    if len(pretrained_model_name) == 0:
        # model = PPO2('CnnPolicy', env, verbose=1)
        model = PPO2('MlpPolicy', env, verbose=1)
    else:
        model = PPO2.load(os.path.join(logpath, pretrained_model_name), env=env)
        # model.env = env
    callback = save_model_callback(check_freq=1000, log_dir=logpath)
    model.learn(total_timesteps=int(1e6), log_interval=50, callback=callback) # Train the agent
    # evaluate_model(env, model)

def trpo(pretrained_model_name: str = ""):
    logpath = os.path.join(LOG_DIR, 'trpo')

    # Create env
    env = make_dino_env(ENV_ID, logpath)

    # Train model from scratch
    if len(pretrained_model_name) == 0:
        model = TRPO('CnnPolicy', env, verbose=1)
    else:
        model = TRPO.load(os.path.join(logpath, pretrained_model_name))
        model.env = env
    callback = save_model_callback(check_freq=1000, log_dir=logpath)
    model.learn(total_timesteps=25000, log_interval=10, callback=callback) # Train the agent
    evaluate_model(env, model)

def ppo1(pretrained_model_name: str = ""):
    logpath = os.path.join(LOG_DIR, 'ppo1')

    # Create env
    env = make_dino_env(ENV_ID, logpath)

    # Train model from scratch
    if len(pretrained_model_name) == 0:
        model = PPO1('CnnPolicy', env, verbose=1)
    else:
        model = PPO1.load(os.path.join(logpath, pretrained_model_name))
        model.env = env
    callback = save_model_callback(check_freq=1000, log_dir=logpath)
    model.learn(total_timesteps=25000, log_interval=10, callback=callback) # Train the agent
    evaluate_model(env, model)

def a2c(pretrained_model_name: str = ""):
    logpath = os.path.join(LOG_DIR, 'a2c')

    # Create Env
    env = make_dino_vec_env(ENV_ID, logpath)

    # Train model from scratch
    if len(pretrained_model_name) == 0:
        model = A2C('CnnPolicy', env, verbose=1)
    else:
        model = A2C.load(os.path.join(logpath, pretrained_model_name))
        model.env = env
    callback = save_model_callback(check_freq=1000, log_dir=logpath)
    model.learn(total_timesteps=100000, log_interval=50, callback=callback) # Train the agent
    # evaluate_model(env, model)

def acktr_notworking(pretrained_model_name: str = ""):
    logpath = os.path.join(LOG_DIR, 'acktr')

    # Create Env
    env = make_dino_vec_env(ENV_ID, logpath)

    # Train model from scratch
    if len(pretrained_model_name) == 0:
        model = ACKTR('CnnPolicy', env, verbose=1)
    else:
        model = ACKTR.load(os.path.join(logpath, pretrained_model_name))
        model.env = env
    callback = save_model_callback(check_freq=1000, log_dir=logpath)
    model.learn(total_timesteps=100000, log_interval=50, callback=callback) # Train the agent
    # evaluate_model(env, model)

def ppo2_evaluate(pretrained_model_name: str, nbr_timesteps = 10000):
    logpath = os.path.join(LOG_DIR, 'ppo2')

    # Create Env
    # env = make_dino_vec_env(ENV_ID, logpath, 1)
    env = gym.make('ChromeDinoHandrafted-v0')
    # env = make_vec_env('ChromeDinoHandrafted_accl-v0', n_envs=1)

    # Evaluate model
    model = PPO2.load(os.path.join(logpath, pretrained_model_name), env)
    evaluate_model(env, model, nbr_timesteps)
    # evaluate_policy(model, env, deterministic=True)