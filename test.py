from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from vizdoomenv import ViZDoomGym
import time
model = PPO.load('./train/train_defend/best_model_40000')

env = ViZDoomGym(render=True)

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100)

for episode in range(5): 
    obs = env.reset()
    done = False
    total_reward = 0
    while not done: 
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # time.sleep(0.50)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(total_reward, episode))
    time.sleep(2)