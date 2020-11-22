import gym_ple

import gym
import os
from gym_chrome_dino.utils.wrappers import make_dino
from stable_baselines.bench import Monitor
from stable_baselines import DQN
from rl_trainers.utils import save_model_callback
from rl_trainers.utils import evaluate_model

# Set log directory
LOG_DIR = "Logs"


def train_dqn(saved_model_name: str, env_id: str = "ChromeDino-v0", env_suit_dino: bool = False):
    logpath = os.path.join(LOG_DIR, 'dqn_flappybird')

    # Create env
    env = gym.make(env_id)

    if env_suit_dino:
        env = make_dino(env, timer=True, frame_stack=True)

    env = Monitor(env, logpath)
    env.reset()

    model = DQN('CnnPolicy', env,
                gamma=0.99,
                learning_rate=1e-3,
                exploration_fraction=0.1,
                prioritized_replay=True, verbose=1)
    print("Untrained model: ")
    evaluate_model(env, model)

    callback = save_model_callback(check_freq=1000, log_dir=logpath)
    model.learn(total_timesteps=20000, log_interval=10, callback=callback)  # Train the agent
    model.save(os.path.join(logpath, saved_model_name))

    print("Trained model: ")
    evaluate_model(env, model)

    env.close()


def transfer_learning_dqn(original_model_name: str, saved_model_name: str, env_id: str = "ChromeDino-v0"):
    logpath = os.path.join(LOG_DIR, 'dqn_flappybird')

    # Create env
    env = gym.make(env_id)
    env = make_dino(env, timer=True, frame_stack=True)

    env = Monitor(env, logpath)

    model = DQN.load(os.path.join(logpath, original_model_name))
    model.set_env(env)

    print("Untrained model: ")
    evaluate_model(env, model)

    callback = save_model_callback(check_freq=1000, log_dir=logpath)
    model.learn(total_timesteps=20000, log_interval=10, callback=callback)  # Train the agent
    model.save(os.path.join(logpath, saved_model_name))

    print("Trained model: ")
    evaluate_model(env, model)

    env.close()


def render_model(path: str = None, env_suit_dino: bool = False):
    logpath = os.path.join(LOG_DIR, 'dqn_flappybird')

    env = gym.make("FlappyBird-v0")

    env = Monitor(env, logpath)

    if env_suit_dino:
        env = make_dino(env, timer=True, frame_stack=True)

    if path is not None:
        model = DQN.load(path)
        model.set_env(env)
    else:
        model = DQN('CnnPolicy', env,
                    gamma=0.99,
                    learning_rate=1e-3,
                    exploration_fraction=0.1,
                    prioritized_replay=True, verbose=1)

    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        while True:
            action, _states = model.predict(ob)
            ob, reward, done, _ = env.step(action)
            env.render()
            if done:
                break

    env.close()


if __name__ == '__main__':
    # Training models with random weights and RGB input
    train_dqn(saved_model_name="dqn_wo_mod_flappybird.zip", env_id="FlappyBird-v0", env_suit_dino=False)
    # Training models with random weights and grayscale input
    train_dqn(saved_model_name="dqn_w_mod_flappybird.zip", env_id="FlappyBird-v0", env_suit_dino=True)

    # Training models with pretrained weights
    transfer_learning_dqn("dqn_dinorun.zip", "dqn_tl_flappybird.zip", env_id="FlappyBird-v0")

    # View untrained model with raw RGB inputs
    render_model(path=None,
                 env_suit_dino=False)
    # View untrained model with grayscale inputs
    render_model(path=None,
                 env_suit_dino=True)
    # View pretrained model with grayscale inputs
    # TODO: For the zipfiles below, please download the files using links from
    # TODO: rl_trainers/Logs/dqn_flappybird/Flappybrid trained using dqn model.txt
    # TODO: and leave it in rl_trainers/Logs/dqn_flappybird
    render_model(path="Logs/dqn_flappybird/dqn_dinorun.zip",
                 env_suit_dino=True)
    # View trained model with raw RGB inputs
    render_model(path="Logs/dqn_flappybird/dqn_wo_mod_flappybird.zip",
                 env_suit_dino=False)
    # View trained model with grayscale inputs
    render_model(path="Logs/dqn_flappybird/dqn_w_mod_flappybird.zip",
                 env_suit_dino=True)
    # View pretrained trained model with grayscale inputs
    render_model(path="Logs/dqn_flappybird/dqn_tl_flappybird.zip",
                 env_suit_dino=True)
