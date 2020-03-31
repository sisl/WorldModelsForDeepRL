import os
import gym
import cv2
import time
import torch
import random
import argparse
import numpy as np
import utils.misc as misc
from envs import make_env, env_name
from utils.rand import RandomAgent
from data.loaders import ROOT
from models.worldmodel.controller import ControlAgent
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Rollout Generator")
parser.add_argument("--nsamples", type=int, default=0, help="How many rollouts to save")
parser.add_argument("--iternum", type=int, default=0, choices=[0,1], help="Which iteration of trained World Model to load (0 or 1)")
parser.add_argument("--datadir", type=str, default=ROOT, help="The directory path to save the sampled rollouts")
args = parser.parse_args()

class RolloutCollector():
	def __init__(self, save_path):
		self.reset_rollout()
		self.save_path = save_path

	def reset_rollout(self):
		self.a_rollout = []
		self.s_rollout = []
		self.r_rollout = []
		self.d_rollout = []

	def step(self, env_action, next_state, reward, done, number=None):
		self.a_rollout.append(env_action)
		self.s_rollout.append(misc.resize(next_state))
		self.r_rollout.append(reward)
		self.d_rollout.append(done)
		if done: self.save_rollout(number=number)

	def save_rollout(self, number=None):
		if len(self.a_rollout) + len(self.s_rollout) + len(self.r_rollout) + len(self.d_rollout) == 0: return
		if number is None: number = len([n for n in os.listdir(self.save_path)])
		a = np.array(self.a_rollout)
		s = np.array(self.s_rollout)
		r = np.array(self.r_rollout)
		d = np.array(self.d_rollout)
		np.savez(os.path.join(self.save_path, f"rollout_{number}"), actions=a, states=s, rewards=r, dones=d)
		self.reset_rollout()

def sample(runs, iternum, root=ROOT, number=None):
	dirname = os.path.join(root, f"iter{iternum}/")
	os.makedirs(os.path.dirname(dirname), exist_ok=True)
	rollout = RolloutCollector(dirname)
	env = make_env()
	state_size = env.observation_space.shape
	action_size = [env.action_space.n] if hasattr(env.action_space, 'n') else env.action_space.shape
	agent = RandomAgent(state_size, action_size) if iternum <= 0 else ControlAgent(state_size, action_size, gpu=False, load=f"{env_name}/iter{iternum-1}")
	for ep in range(runs):
		state = env.reset()
		total_reward = 0
		done = False
		while not done:
			env_action = agent.get_env_action(env, state)[0]
			state, reward, done, _ = env.step(env_action)
			rollout.step(env_action, state, reward, done, number)
			agent.train(state, env_action, state, reward, done)
			total_reward += reward
		print(f"Ep: {ep}, Reward: {total_reward}")
	env.close()

def check_samples(iternum, root=ROOT):
	dirname = os.path.join(root, f"iter{iternum}/")
	if os.path.exists(dirname):
		files = [os.path.join(root, dirname, n) for n in sorted(os.listdir(dirname), key=lambda x: str(len(x))+x)]
		for i,f in enumerate(files):
			with np.load(f) as data:
				size = data["rewards"].shape[0]
				print(data["states"].shape, data["actions"].shape, data["rewards"].shape)
			if size <= 1:
				print(f"{f}: {size}")
				sample(1, iternum, number=i)
				check_samples(iternum)

if __name__ == "__main__":
	if args.nsamples > 0:
		sample(args.nsamples, args.iternum, args.datadir)
	# check_samples(args.iternum, args.datadir)