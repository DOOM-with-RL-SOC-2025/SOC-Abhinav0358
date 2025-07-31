from gymnasium import Env
from vizdoom import *
from vizdoom import DoomGame
import random
import time
import numpy as np
from gymnasium.spaces import Discrete, Box
import cv2

class ViZDoomGym(Env):
    """
    ViZDoom Gym Environment Wrapper
    """
    
    def __init__(self, render=False):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config("ViZDoom/scenarios/basic.cfg")
        
        if render == False:
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)

        self.game.init()

        self.observation_space = Box(low=0, high=255, shape=(100,160, 1), dtype=np.uint8)
        self.action_space = Discrete(3) 

    def reset(self, seed=None, options=None):
        # Handle seeding if provided
        if seed is not None:
            np.random.seed(seed)
            
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        observation = self.grayscale(state)
        
        # Return observation and info dict (modern Gymnasium format)
        info = {}
        return observation, info

    def step(self, action):
        # Specify action and take step 
        actions = np.identity(3)
        reward = self.game.make_action(actions[action], 4) 

        # Get all the other stuff we need to return 
        if self.game.get_state(): 
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            ammo = self.game.get_state().game_variables[0]
            info = ammo
        else: 
            state = np.zeros(self.observation_space.shape)
            info = 0 
        
        info = {"info":info}
        
        # Modern Gymnasium API: separate terminated and truncated
        terminated = self.game.is_episode_finished()
        truncated = False  # ViZDoom doesn't typically truncate episodes
        
        return state, reward, terminated, truncated, info 
    def render(self, mode='human'):
        if self.game is None:
            raise RuntimeError("Environment not initialized.")
        return self.game.render(mode)

    def close(self):
        self.game.close()

    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_RGB2GRAY)
        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (100, 160, 1))
        return state