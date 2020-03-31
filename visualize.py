import os
import gym
import cv2
import torch
import argparse
import numpy as np
from torchvision import transforms
from models.worldmodel.vae import VAE, LATENT_SIZE
from models.worldmodel.mdrnn import MDRNNCell, HIDDEN_SIZE
from models.worldmodel.controller import ControlAgent
from models.ddpg import DDPGAgent, EPS_MIN
from models.ppo import PPOAgent
from utils.misc import IMG_DIM, resize, make_video
from utils.envs import ImgStack, WorldModel
from utils.wrappers import WorldACAgent, rollout
from data.loaders import ROOT

parser = argparse.ArgumentParser(description="Visualizer")
parser.add_argument("--iternum", type=int, default=1, help="Whether to use world model (0 or 1) or raw pixels (-1) [default]")
parser.add_argument("--runs", type=int, default=1, help="Number of episodes to train the agent")
args = parser.parse_args()

def evaluate_best(runs=1, gpu=True, iternums=[-1, 0, 1]):
	env = gym.make("CarRacing-v0")
	env.env.verbose = 0
	for iternum in iternums:
		dirname = "pytorch" if iternum < 0 else f"iter{iternum}/"
		for model in [PPOAgent]:#, PPOAgent]:
			statemodel = ImgStack if iternum < 0 else WorldModel
			agent = WorldACAgent(env.action_space.shape, 1, model, statemodel, load=dirname, gpu=gpu, train=False)
			# scores = [rollout(env, agent.reset(), eps=EPS_MIN, render=True) for _ in range(runs)]
			scores = []
			for ep in range(runs):
				scores.append(rollout(env, agent.reset(), eps=EPS_MIN, sample=False, render=False))
				print(f"   Ep: {ep}, Score: {scores[-1]}")
			mean = np.mean(scores)
			std = np.std(scores)
			print(f"It: {iternum}, Model: {model.__name__}, Mean: {mean}, Std: {std} ({EPS_MIN:.4f})")
			# for ep,score in enumerate(scores): print(f"   Ep: {ep}, Score: {score}")

		# agent = ControlAgent(env.action_space.shape, load=dirname, gpu=True)
		# scores = [rollout(env, agent, render=True) for _ in range(runs)]
		# mean = np.mean(scores)
		# std = np.std(scores)
		# print(f"It: {iternum}, Model: Controller, Mean: {mean}, Std: {std}")
		# for ep,score in enumerate(scores): print(f"   Ep: {ep}, Score: {score}")
	env.close()

def visualize_vae(rollout, path="/home/shawn/Documents/world-models/datasets/carracing/openai/", save="./tests/videos/vae.avi"):
	images_file = os.path.join(path, f"rollout_{rollout}.npz")
	vae = VAE()
	imgs = []
	with np.load(images_file) as data:
		rollout = {k: np.copy(v) for k, v in data.items()}
		for img in rollout["states"]:
			rec = vae.forward(torch.Tensor(img/255).permute(2,0,1).unsqueeze(0))[0]
			rec = rec.squeeze().permute(1,2,0).cpu().detach().numpy()*255
			img = np.concatenate((img,rec.astype(np.uint8)), axis=1)
			imgs.append(img)
	make_video(imgs, (2*IMG_DIM, IMG_DIM), save)

def visualize_mdrnn(rollout, path="/home/shawn/Documents/world-models/datasets/carracing/openai/", save="./tests/videos/mdrnn.avi"):
	images_file = os.path.join(path, f"rollout_{rollout}.npz")
	vae = VAE()
	mdrnn = MDRNNCell()
	with np.load(images_file) as data:
		rollout = {k: np.copy(v) for k, v in data.items()}
		hid = mdrnn.init_hidden()
		imgs = []
		for img, action in zip(rollout["states"], rollout["actions"]):
			lat = vae.get_latents(torch.Tensor(img/255).permute(2,0,1).unsqueeze(0))
			hid = mdrnn(np.expand_dims(action, 0).astype(np.float32), lat, hid)
			nex = mdrnn.get_latents(hid[0])[0]
			rec = vae.decoder(nex).squeeze().permute(1,2,0).cpu().detach().numpy()*255
			img = np.concatenate((img,rec.astype(np.uint8)), axis=1)
			imgs.append(img)
	make_video(imgs, (2*IMG_DIM, IMG_DIM), save)

def visualize_controller(load_dir="pytorch", gpu=False, save="./tests/videos/ctrl.avi"):
	env = gym.make("CarRacing-v0")
	agent = ControlAgent(env.action_space.shape, gpu=False, load=load_dir)
	img = env.reset()
	imgs = []
	done = False
	with torch.no_grad():
		while not done:
			imgs.append(resize(img))
			action = agent.get_env_action(env, img)
			img, reward, done, _ = env.step(action)
	make_video(imgs, (IMG_DIM, IMG_DIM), save)
	env.close()

def visualize_dream(load_dir="pytorch", gpu=False, save="./tests/videos/dream.avi"):
	env = gym.make("CarRacing-v0")
	agent = ControlAgent(env.action_space.shape, gpu=False, load=load_dir)
	rec = agent.world_model.vae.sample()[0]
	imgs = []
	with torch.no_grad():
		for _ in range(1000):
			agent.get_env_action(env, rec.astype(np.uint8))
			nex = agent.world_model.mdrnn.get_latents(agent.world_model.hidden[0])[0]
			rec = agent.world_model.vae.decoder(nex).squeeze().permute(1,2,0).cpu().detach().numpy()*255
			imgs.append(rec)
	make_video(imgs, (IMG_DIM, IMG_DIM), save)

def visualize_qlearning(gpu=False, save="./tests/videos/qlearning.avi"):
	env = gym.make("CarRacing-v0")
	vae = VAE()
	mdrnn = MDRNNCell()
	agent = DDPGAgent([LATENT_SIZE + HIDDEN_SIZE], env.action_space.shape, gpu=False, load=True)
	transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IMG_DIM, IMG_DIM)), transforms.ToTensor()])
	img = env.reset()
	imgs = []
	done = False
	with torch.no_grad():
		hid = mdrnn.init_hidden()
		while not done:
			env.render()
			img = resize(np.array(img))
			imgs.append(img)
			lat = vae.get_latents(transform(img).unsqueeze(0))
			lat_hid = np.concatenate([lat.cpu().numpy(), hid[0].cpu().numpy()], axis=1)[0]
			action = agent.get_env_action(env, lat_hid, eps=0.1)
			img, reward, done, _ = env.step(action)
			hid = mdrnn(np.expand_dims(action, 0).astype(np.float32), lat, hid)
	make_video(imgs, (IMG_DIM, IMG_DIM), save)
	env.close()

if __name__ == "__main__":
	dirname = os.path.join(ROOT, f"iter{args.iternum}/")
	evaluate_best(args.runs, iternums=[args.iternum])
	# visualize_vae(100, dirname)
	# visualize_mdrnn(200, dirname)
	# visualize_controller(f"iter{args.iternum}/")
	# visualize_dream(f"iter{args.iternum}/")
	# visualize_qlearning()