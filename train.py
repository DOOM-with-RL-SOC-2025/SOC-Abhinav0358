from vizdoomenv import ViZDoomGym
from trainNlog import TrainAndLoggingCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import PPO

env = ViZDoomGym()

CHECKPOINT_DIR = './train/train_defend'
LOG_DIR = './logs/log_defend'
model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=4096)
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
model.learn(total_timesteps=100000, callback=callback)


