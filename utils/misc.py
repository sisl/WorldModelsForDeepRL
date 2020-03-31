import os
import re
import gym
import cv2
import sys
import time
import torch
import random
import inspect
import datetime
import subprocess
import numpy as np
import psutil as psu
import GPUtil as gputil
import platform as pfm
import matplotlib.pyplot as plt
from models import all_models
from models.worldmodel.vae import VAE
from models.worldmodel.mdrnn import MDRNNCell
np.set_printoptions(precision=3, sign=' ', floatmode="fixed")

IMG_DIM = 64					# The height and width to scale the environment image to

def rgb2gray(image):
	gray = np.dot(image, [0.299, 0.587, 0.114]).astype(np.float32)
	return gray

def resize(image, dim=IMG_DIM):
	img = cv2.resize(image, dsize=(dim,dim), interpolation=cv2.INTER_CUBIC)
	return img

def show_image(img, filename="test.png", save=True):
	if save: plt.imsave(filename, img)
	plt.imshow(img, cmap=plt.get_cmap('gray'))
	plt.show()

def make_video(imgs, dim, filename):
	video = cv2.VideoWriter(filename, 0, 60, dim)
	for img in imgs:
		video.write(img.astype(np.uint8))
	video.release()

def to_env(env, action):
	action_normal = (1+action)/2
	action_range = env.action_space.high - env.action_space.low
	env_action = env.action_space.low + np.multiply(action_normal, action_range)
	return env_action

def from_env(env, env_action):
	action_range = env.action_space.high - env.action_space.low
	action_normal = np.divide(env_action - env.action_space.low, action_range)
	action = 2*action_normal - 1
	return action

def rollout(env, agent, eps=None, render=False, sample=False, log_dir=None):
	state = env.reset()
	total_reward = None
	done = None
	with torch.no_grad():
		while not np.all(done):
			if render: env.render()
			env_action = agent.get_env_action(env, state, eps, sample)[0]
			state, reward, ndone, _ = env.step(env_action)
			reward = np.equal(done,False).astype(np.float32)*reward if done is not None else reward
			done = np.array(ndone) if done is None else np.logical_or(done, ndone)
			total_reward = reward if total_reward is None else total_reward + reward
	return total_reward

LOG_DIR = "logs"

class Logger():
	def __init__(self, model_class, env_name, **kwconfig):
		self.git_info = self.get_git_info()
		self.config = kwconfig
		self.env_name = env_name
		self.model_class = model_class
		self.model_name = inspect.getmodule(model_class).__name__.split(".")[-1]
		os.makedirs(f"{LOG_DIR}/{self.model_name}/{env_name}/", exist_ok=True)
		self.run_num = len([n for n in os.listdir(f"{LOG_DIR}/{self.model_name}/{env_name}/")])
		self.model_src = [line for line in open(inspect.getabsfile(self.model_class))]
		self.net_src = [line for line in open(f"utils/network.py") if re.match("^[A-Z]", line)] if self.model_name in all_models.keys() else None
		self.trn_src = [line for line in open(f"train_a3c.py")] if self.model_name in all_models.keys() else None
		self.log_path = f"{LOG_DIR}/{self.model_name}/{env_name}/logs_{self.run_num}.txt"
		self.log_num = 0

	def log(self, string, debug=True):
		with open(self.log_path, "a+") as f:
			if self.log_num == 0: 
				self.start_time = time.time()
				f.write(f"Model: {self.model_class}, Env: {self.env_name}, Date: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
				f.write(self.get_hardware_info() + "\n")
				f.write(self.git_info + "\n\n")
				if self.config: f.writelines("\n".join([f"{k}: {v}," for k,v in self.config.items()]) + "\n\n")
				if self.model_src: f.writelines(self.model_src + ["\n"])
				if self.net_src: f.writelines(self.net_src + ["\n"])
				if self.trn_src: f.writelines(self.trn_src + ["\n"])
				f.write("\n")
			f.write(f"{string} <{self.get_time()}> \n")
		if debug: print(string)
		self.log_num += 1

	def get_git_info(self):
		git_status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], universal_newlines=True)
		# assert len(git_status)==0, "Uncommitted changes need to be committed to log current codebase state"
		git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], universal_newlines=True).strip()
		git_url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], universal_newlines=True).strip()
		git_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], universal_newlines=True).strip()
		return "\n".join([f"{k}: {v}" for v,k in zip([git_url, git_hash, git_branch], ["Git URL", "Hash", "Branch"])])

	def get_time(self):
		delta = time.gmtime(time.time()-self.start_time)
		return f"{delta.tm_mday-1}-{time.strftime('%H:%M:%S', delta)}"

	def get_hardware_info(self):
		cpu_info = f"CPU: {psu.cpu_count(logical=False)} Core, {psu.cpu_freq().max/1000}GHz, {np.round(psu.virtual_memory().total/1024**3,2)} GB, {pfm.platform()}"
		gpu_info = [f"GPU {gpu.id}: {gpu.name}, {np.round(gpu.memoryTotal/1000,2)} GB (Driver: {gpu.driver})" for gpu in gputil.getGPUs()]
		return "\n".join([cpu_info, *gpu_info])

	def get_classes(self, model):
		return [v for k,v in model.__dict__.items() if inspect.getmembers(v)[0][0] == "__class__"]