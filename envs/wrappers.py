import os
import gym
import cv2
import numpy as np
from utils.misc import resize, IMG_DIM

try:
	import vizdoom as vzd
except:
	print("Unable to import vizdoom")

configs = sorted([s.replace(".cfg","") for s in sorted(os.listdir("./envs/ViZDoom/scenarios/")) if s.endswith(".cfg")])

class GymEnv(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)

	def reset(self, **kwargs):
		self.time = 0
		return self.env.reset()

	def step(self, action, train=False):
		self.time += 1
		return super().step(action)

class VizDoomEnv():
	def __init__(self, env_name, resize=IMG_DIM, transpose=[1,2,0], render=False):
		self.transpose = transpose
		self.env = vzd.DoomGame()
		self.env.load_config(os.path.abspath(f"./envs/ViZDoom/scenarios/{env_name}.cfg"))
		self.action_space = gym.spaces.Discrete(self.env.get_available_buttons_size())
		self.size = [self.env.get_screen_channels()] + ([resize, resize] if resize else [self.env.get_screen_height(), self.env.get_screen_width()])
		self.sizet = [self.size[x] for x in transpose] if self.transpose else self.size
		self.observation_space = gym.spaces.Box(0,255,shape=self.sizet)
		self.env.set_window_visible(render)
		self.env.init()

	def reset(self):
		self.time = 0
		self.done = False
		self.env.new_episode()
		state = self.env.get_state().screen_buffer
		if self.transpose: state = np.transpose(state, self.transpose)
		return resize(state.astype(np.uint8))

	def step(self, action, train=False):
		self.time += 1
		action_oh = self.one_hot(action)
		reward = self.env.make_action(action_oh)
		self.done = self.env.is_episode_finished() or self.done
		state = np.zeros(self.size) if self.done else self.env.get_state().screen_buffer
		if self.transpose: state = np.transpose(state, self.transpose)
		return resize(state.astype(np.uint8)), reward, self.done, None

	def render(self):
		pass

	def one_hot(self, action):
		action_oh = [0]*self.action_space.n
		action_oh[int(action)] = 1
		return action_oh

	def close(self):
		self.env.close()

	def __del__(self):
		self.close()
