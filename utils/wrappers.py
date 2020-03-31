import torch
import numpy as np
from collections import deque
from torchvision import transforms
from models.worldmodel.vae import VAE, LATENT_SIZE
from models.worldmodel.mdrnn import MDRNNCell, HIDDEN_SIZE
from utils.rand import RandomAgent
from utils.misc import IMG_DIM

FRAME_STACK = 3					# The number of consecutive image states to combine for training a3c on raw images
NUM_ENVS = 16					# The default number of environments to simultaneously train the a3c in parallel

class RawStack():
	def __init__(self, state_size, action_size, num_envs=1, stack_len=FRAME_STACK, load="", gpu=True):
		self.state_size = state_size
		self.action_size = action_size
		self.stack_len = stack_len
		self.reset(num_envs)

	def reset(self, num_envs, restore=False):
		pass

	def get_state(self, state):
		return state, None

	def step(self, state, env_action):
		pass

	def load_model(self, dirname="pytorch", name="best"):
		return self

class ImgStack(RawStack):
	def __init__(self, state_size, action_size, num_envs=1, stack_len=FRAME_STACK, load="", gpu=True):
		super().__init__(action_size, num_envs, stack_len)
		self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize((IMG_DIM, IMG_DIM)), transforms.ToTensor()])
		self.process = lambda x: self.transform(x.astype(np.uint8)).unsqueeze(0).numpy()
		self.state_size = [IMG_DIM, IMG_DIM, stack_len]
		self.stack_len = stack_len
		self.reset(num_envs)

	def reset(self, num_envs, restore=False):
		self.num_envs = num_envs
		self.stack = deque(maxlen=self.stack_len)

	def get_state(self, state):
		state = np.concatenate([self.process(s) for s in state]) if self.num_envs > 1 else self.process(state)
		while len(self.stack) < self.stack_len: self.stack.append(state)
		self.stack.append(state)
		return np.concatenate(self.stack, axis=1), None

class WorldModel():
	def __init__(self, state_size, action_size, num_envs=1, load="", gpu=True):
		self.vae = VAE(load=load, gpu=gpu)
		self.mdrnn = MDRNNCell(action_size, load=load, gpu=gpu)
		self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IMG_DIM, IMG_DIM)), transforms.ToTensor()])
		self.state_size = [LATENT_SIZE + HIDDEN_SIZE]
		self.hiddens = {}
		self.reset(num_envs)
		if load: self.load_model()

	def reset(self, num_envs, restore=False):
		self.num_envs = num_envs
		self.hidden = self.hiddens[num_envs] if restore and num_envs in self.hiddens else self.mdrnn.init_hidden(num_envs)
		self.hiddens[num_envs] = self.hidden

	def get_state(self, state, numpy=True):
		state = torch.cat([self.transform(s).unsqueeze(0) for s in state]) if len(state.shape) > 3 else self.transform(state).unsqueeze(0)
		latent = self.vae.get_latents(state)
		lat_hid = torch.cat([latent, self.hidden[0]], dim=1)
		return lat_hid.cpu().numpy() if numpy else lat_hid, latent

	def step(self, latent, env_action):
		self.hidden = self.mdrnn(env_action.astype(np.float32), latent, self.hidden)

	def load_model(self, dirname="pytorch", name="best"):
		self.vae.load_model(dirname, name)
		self.mdrnn.load_model(dirname, name)
		return self

class WorldACAgent(RandomAgent):
	def __init__(self, state_size, action_size, acagent, num_envs, load="", gpu=True, train=True, worldmodel=True):
		super().__init__(state_size, action_size)
		statemodel = RawStack if len(state_size)!=3 else WorldModel if worldmodel else ImgStack
		self.state_model = statemodel(state_size, action_size, num_envs=num_envs, load=load, gpu=gpu)
		self.acagent = acagent(self.state_model.state_size, action_size, load="" if train else load, gpu=gpu)

	def get_env_action(self, env, state, eps=None, sample=True):
		state, latent = self.state_model.get_state(state)
		env_action, action = self.acagent.get_env_action(env, state, eps, sample)
		self.state_model.step(latent, env_action)
		return env_action, action, state

	def train(self, state, action, next_state, reward, done):
		next_state = self.state_model.get_state(next_state)[0]
		self.acagent.train(state, action, next_state, reward, done)

	def reset(self, num_envs=None):
		num_envs = self.state_model.num_envs if num_envs is None else num_envs
		self.state_model.reset(num_envs, restore=False)
		return self

	def save_model(self, dirname="pytorch", name="checkpoint"):
		self.acagent.network.save_model(dirname, name)

	def load(self, dirname="pytorch", name="checkpoint"):
		self.acagent.network.load_model(dirname, name)
		return self
