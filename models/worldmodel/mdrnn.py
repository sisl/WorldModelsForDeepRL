import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from models.worldmodel.vae import LATENT_SIZE
from utils.network import one_hot_from_indices

ACTION_SIZE = None					# The number of continuous action values required by the CarRacing-v0 environment
HIDDEN_SIZE = 256				# The number of nodes in the hidden layer output by the MDRNN
N_GAUSS = 5						# The number of Gaussian parameters to output for the mixture of Gaussians
LEARNING_RATE = 0.001			# Sets how much we want to update the network weights at each training step
ALPHA = 0.9						# The decay rate of the learning rate

class MDRNN(torch.nn.Module):
	def __init__(self, action_size, latent_size=LATENT_SIZE, hidden_size=HIDDEN_SIZE, n_gauss=N_GAUSS, load="", gpu=True):
		super().__init__()
		self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
		self.action_size = action_size
		self.latent_size = latent_size
		self.hidden_size = hidden_size
		self.n_gauss = n_gauss
		self.discrete = type(self.action_size) != tuple
		self.lstm = torch.nn.LSTM(action_size[-1] + latent_size, hidden_size).to(self.device)
		self.gmm = torch.nn.Linear(hidden_size, (2*latent_size+1)*n_gauss + 2).to(self.device)
		self.optimizer = torch.optim.RMSprop(self.parameters(), lr=LEARNING_RATE, alpha=ALPHA)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5)
		if load: self.load_model(load)

	def forward(self, actions, latents):
		if self.discrete: actions = one_hot_from_indices(actions, self.action_size[-1], keepdims=True).view(*latents.shape[:2], -1)
		if len(actions.shape) != len(latents.shape):
			print(actions.shape, latents.shape)
		lstm_inputs = torch.cat([actions, latents], dim=-1)
		lstm_hiddens, _ = self.lstm(lstm_inputs)
		gmm_outputs = self.gmm(lstm_hiddens)
		stride = self.n_gauss*self.latent_size
		mus = gmm_outputs[:,:,:stride]
		sigs = gmm_outputs[:,:,stride:2*stride]
		pi = gmm_outputs[:,:,2*stride:2*stride+self.n_gauss]
		rs = gmm_outputs[:,:,2*stride+self.n_gauss]
		ds = gmm_outputs[:,:,2*stride+self.n_gauss+1]
		mus = mus.view(mus.size(0), mus.size(1), self.n_gauss, self.latent_size)
		sigs = sigs.view(sigs.size(0), sigs.size(1), self.n_gauss, self.latent_size).exp()
		logpi = pi.view(pi.size(0), pi.size(1), self.n_gauss).log_softmax(dim=-1)
		return mus, sigs, logpi, rs, ds

	def get_gmm_loss(self, mus, sigs, logpi, next_latents):
		dist = torch.distributions.normal.Normal(mus, sigs)
		log_probs = dist.log_prob(next_latents.unsqueeze(-2))
		log_probs = logpi + torch.sum(log_probs, dim=-1)
		max_log_probs = torch.max(log_probs, dim=-1, keepdim=True)[0]
		g_log_probs = log_probs - max_log_probs
		g_probs = torch.sum(torch.exp(g_log_probs), dim=-1)
		log_prob = max_log_probs.squeeze() + torch.log(g_probs)
		return -torch.mean(log_prob)

	def get_loss(self, latents, actions, next_latents, rewards, dones):
		l, a, nl, r, d = [x.to(self.device) for x in (latents, actions, next_latents, rewards, dones)]
		mus, sigs, logpi, rs, ds = self.forward(a, l)
		mse = torch.nn.functional.mse_loss(rs, r)
		bce = torch.nn.functional.binary_cross_entropy_with_logits(ds, d)
		gmm = self.get_gmm_loss(mus, sigs, logpi, nl)
		return (gmm + mse + bce) / (self.latent_size + 2)

	def optimize(self, latents, actions, next_latents, rewards, dones):
		loss = self.get_loss(latents, actions, next_latents, rewards, dones)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()

	def schedule(self, test_loss):
		self.scheduler.step(test_loss)

	def save_model(self, dirname="pytorch", name="best"):
		filepath = get_checkpoint_path(dirname, name)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.state_dict(), filepath)
		
	def load_model(self, dirname="pytorch", name="best"):
		filepath = get_checkpoint_path(dirname, name)
		if os.path.exists(filepath):
			self.load_state_dict(torch.load(filepath, map_location=self.device))
			print(f"Loaded MDRNN model at {filepath}")
		return self

class MDRNNCell(torch.nn.Module):
	def __init__(self, action_size, latent_size=LATENT_SIZE, hidden_size=HIDDEN_SIZE, n_gauss=N_GAUSS, load="", gpu=True):
		super().__init__()
		self.n_gauss = N_GAUSS
		self.action_size = action_size
		self.latent_size = latent_size
		self.discrete = type(self.action_size) == list
		self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
		self.lstm = torch.nn.LSTMCell(action_size[-1] + latent_size, hidden_size).to(self.device)
		self.gmm = torch.nn.Linear(hidden_size, (2*latent_size+1)*n_gauss + 2).to(self.device)
		if load: self.load_model(load)

	def forward(self, actions, latents, hiddens):
		with torch.no_grad():
			actions, latents = [x.to(self.device) for x in (torch.from_numpy(actions), latents)]
			if self.discrete: actions = one_hot_from_indices(actions, self.action_size[-1], keepdims=True).view(*latents.shape[:-1], -1)
			lstm_inputs = torch.cat([actions, latents], dim=-1)
			lstm_hidden = self.lstm(lstm_inputs, hiddens)
			return lstm_hidden

	def get_latents(self, hiddens):
		with torch.no_grad():
			gmm_out = self.gmm(hiddens)
			stride = self.n_gauss*self.latent_size
			mus = gmm_out[:,:stride]
			sigs = gmm_out[:,stride:2*stride].exp()
			pi = gmm_out[:,2*stride:2*stride+self.n_gauss].softmax(dim=-1)
			rs = gmm_out[:,2*stride+self.n_gauss]
			ds = gmm_out[:,2*stride+self.n_gauss+1].sigmoid()
			mus = mus.view(-1, self.n_gauss, self.latent_size)
			sigs = sigs.view(-1, self.n_gauss, self.latent_size)
			dist = torch.distributions.categorical.Categorical(pi)
			indices = dist.sample()
			mus = mus[:,indices,:].squeeze(1)
			sigs = sigs[:,indices,:].squeeze(1)
			next_latents = mus + torch.randn_like(sigs).mul(sigs)
			return next_latents, rs, ds

	def init_hidden(self, batch_size=1):
		return [torch.zeros(batch_size, HIDDEN_SIZE).to(self.device) for _ in range(2)]

	def load_model(self, dirname="pytorch", name="best"):
		filepath = get_checkpoint_path(dirname, name)
		if os.path.exists(filepath):
			self.load_state_dict({k.replace("_l0",""):v for k,v in torch.load(filepath, map_location=self.device).items()})
			print(f"Loaded MDRNNCell model at {filepath}")
		return self

def get_checkpoint_path(dirname="pytorch", name="best"):
	return f"./saved_models/mdrnn/{dirname}/{name}.pth"