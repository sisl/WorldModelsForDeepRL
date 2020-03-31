import os
import math
import torch
import random
import inspect
import numpy as np
from utils.rand import RandomAgent, ReplayBuffer

REG_LAMBDA = 1e-6             	# Penalty multiplier to apply for the size of the network weights
LEARN_RATE = 0.0001           	# Sets how much we want to update the network weights at each training step
TARGET_UPDATE_RATE = 0.0004   	# How frequently we want to copy the local network to the target network (for double DQNs)
INPUT_LAYER = 512				# The number of output nodes from the first layer to Actor and Critic networks
ACTOR_HIDDEN = 256				# The number of nodes in the hidden layers of the Actor network
CRITIC_HIDDEN = 1024			# The number of nodes in the hidden layers of the Critic networks

DISCOUNT_RATE = 0.99			# The discount rate to use in the Bellman Equation
NUM_STEPS = 500					# The number of steps to collect experience in sequence for each GAE calculation
EPS_MAX = 1.0                 	# The starting proportion of random to greedy actions to take
EPS_MIN = 0.020               	# The lower limit proportion of random to greedy actions to take
EPS_DECAY = 0.980             	# The rate at which eps decays from EPS_MAX to EPS_MIN
ADVANTAGE_DECAY = 0.95			# The discount factor for the cumulative GAE calculation
REPLAY_BATCH_SIZE = 32        	# How many experience tuples to sample from the buffer for each train step
MAX_BUFFER_SIZE = 100000      	# Sets the maximum length of the replay buffer

SAVE_DIR = "./saved_models"

class Conv(torch.nn.Module):
	def __init__(self, state_size, output_size):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(state_size[-1], 32, kernel_size=4, stride=2)
		self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
		self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2)
		self.linear1 = torch.nn.Linear(self.get_conv_output(state_size), output_size)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state):
		out_dims = state.size()[:-3]
		state = state.view(-1, *state.size()[-3:])
		state = self.conv1(state).tanh()
		state = self.conv2(state).tanh() 
		state = self.conv3(state).tanh() 
		state = self.conv4(state).tanh() 
		state = state.view(state.size(0), -1)
		state = self.linear1(state).tanh()
		state = state.view(*out_dims, -1)
		return state

	def get_conv_output(self, state_size):
		inputs = torch.randn(1, state_size[-1], *state_size[:-1])
		output = self.conv4(self.conv3(self.conv2(self.conv1(inputs))))
		return np.prod(output.size())

class PTActor(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.state_fc1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)!=3 else Conv(state_size, INPUT_LAYER)
		self.state_fc2 = torch.nn.Linear(INPUT_LAYER, ACTOR_HIDDEN)
		self.state_fc3 = torch.nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN)
		self.action_mu = torch.nn.Linear(ACTOR_HIDDEN, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state):
		state = self.state_fc1(state).relu() 
		state = self.state_fc2(state).relu() 
		state = self.state_fc3(state).relu() 
		action_mu = self.action_mu(state)
		return action_mu

class PTCritic(torch.nn.Module):
	def __init__(self, state_size, action_size=[1]):
		super().__init__()
		self.state_fc1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)!=3 else Conv(state_size, INPUT_LAYER)
		self.state_fc2 = torch.nn.Linear(INPUT_LAYER, CRITIC_HIDDEN)
		self.state_fc3 = torch.nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN)
		self.value = torch.nn.Linear(CRITIC_HIDDEN, action_size[-1])
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state, action=None):
		state = self.state_fc1(state).relu()
		state = self.state_fc2(state).relu()
		state = self.state_fc3(state).relu()
		value = self.value(state)
		return value

class PTNetwork():
	def __init__(self, tau=TARGET_UPDATE_RATE, gpu=True, name="pt"): 
		self.tau = tau
		self.name = name
		self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')

	def init_weights(self, model=None):
		model = self if model is None else model
		model.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)
		
	def step(self, optimizer, loss, param_norm=None, retain=False, norm=0.5):
		optimizer.zero_grad()
		loss.backward(retain_graph=retain)
		if param_norm is not None: torch.nn.utils.clip_grad_norm_(param_norm, norm)
		optimizer.step()

	def soft_copy(self, local, target, tau=None):
		tau = self.tau if tau is None else tau
		for t,l in zip(target.parameters(), local.parameters()):
			t.data.copy_(t.data + tau*(l.data - t.data))

	def get_checkpoint_path(self, dirname="pytorch", name="checkpoint", net=None):
		filepath = os.path.join(SAVE_DIR, self.name if net is None else net, dirname, f"{name}.pth")
		return filepath, os.path.dirname(filepath)

class PTQNetwork(PTNetwork):
	def __init__(self, state_size, action_size, critic=PTCritic, lr=LEARN_RATE, tau=TARGET_UPDATE_RATE, gpu=True, load="", name="ptq"): 
		super().__init__(tau, gpu, name)
		self.critic_local = critic(state_size, action_size).to(self.device)
		self.critic_target = critic(state_size, action_size).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=lr, weight_decay=REG_LAMBDA)
		if load: self.load_model(load)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.critic_local.state_dict(), filepath)
		with open(filepath.replace(".pth", ".txt"), "w") as f:
			f.write(inspect.getsource(self.critic_local.__class__))
		return os.path.dirname(filepath)
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		if os.path.exists(filepath):
			try:
				self.critic_local.load_state_dict(torch.load(filepath, map_location=self.device))
				self.critic_target.load_state_dict(torch.load(filepath, map_location=self.device))
			except:
				print(f"WARN: Error loading model from {filepath}")

class PTACNetwork(PTNetwork):
	def __init__(self, state_size, action_size, actor=PTActor, critic=PTCritic, lr=LEARN_RATE, tau=TARGET_UPDATE_RATE, gpu=True, load="", name="ptac"): 
		super().__init__(tau, gpu, name)
		self.actor_local = actor(state_size, action_size).to(self.device)
		self.actor_target = actor(state_size, action_size).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=lr, weight_decay=REG_LAMBDA)
		
		self.critic_local = critic(state_size, action_size).to(self.device)
		self.critic_target = critic(state_size, action_size).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=lr, weight_decay=REG_LAMBDA)
		if load: self.load_model(load)

	def save_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.actor_local.state_dict(), filepath.replace(".pth", "_a.pth"))
		torch.save(self.critic_local.state_dict(), filepath.replace(".pth", "_c.pth"))
		with open(filepath.replace(".pth", ".txt"), "w") as f:
			f.write("\n".join([inspect.getsource(model.__class__) for model in [self.actor_local, self.critic_local]]))
		return os.path.dirname(filepath)
		
	def load_model(self, dirname="pytorch", name="checkpoint", net=None):
		filepath, _ = self.get_checkpoint_path(dirname, name, net)
		if os.path.exists(filepath.replace(".pth", "_a.pth")):
			try:
				self.actor_local.load_state_dict(torch.load(filepath.replace(".pth", "_a.pth"), map_location=self.device))
				self.actor_target.load_state_dict(torch.load(filepath.replace(".pth", "_a.pth"), map_location=self.device))
				self.critic_local.load_state_dict(torch.load(filepath.replace(".pth", "_c.pth"), map_location=self.device))
				self.critic_target.load_state_dict(torch.load(filepath.replace(".pth", "_c.pth"), map_location=self.device))
				print(f"Loaded model from {filepath}")
			except:
				print(f"WARN: Error loading model from {filepath}")

class PTACAgent(RandomAgent):
	def __init__(self, state_size, action_size, network=PTACNetwork, lr=LEARN_RATE, update_freq=NUM_STEPS, eps=EPS_MAX, decay=EPS_DECAY, tau=TARGET_UPDATE_RATE, gpu=True, load=None):
		super().__init__(state_size, action_size, eps)
		self.network = network(state_size, action_size, lr=lr, tau=tau, gpu=gpu, load=load)
		self.replay_buffer = ReplayBuffer(MAX_BUFFER_SIZE)
		self.update_freq = update_freq
		self.buffer = []
		self.decay = decay
		self.eps = eps

	def to_tensor(self, arr):
		if isinstance(arr, np.ndarray): return torch.tensor(arr, requires_grad=False).float().to(self.network.device)
		return self.to_tensor(np.array(arr))

	def get_action(self, state, eps=None, e_greedy=False):
		action_random = super().get_action(state, eps)
		return action_random

	def compute_gae(self, last_value, rewards, dones, values, gamma=DISCOUNT_RATE, lamda=ADVANTAGE_DECAY):
		with torch.no_grad():
			gae = 0
			targets = torch.zeros_like(values, device=values.device)
			values = torch.cat([values, last_value.unsqueeze(0)])
			for step in reversed(range(len(rewards))):
				delta = rewards[step] + gamma * values[step + 1] * (1-dones[step]) - values[step]
				gae = delta + gamma * lamda * (1-dones[step]) * gae
				targets[step] = gae + values[step]
			advantages = targets - values[:-1]
			return targets, advantages
		
	def train(self, state, action, next_state, reward, done):
		pass

class NoisyLinear(torch.nn.Linear):
	def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
		super().__init__(in_features, out_features, bias=bias)
		self.sigma_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features).fill_(sigma_init))
		self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
		if bias:
			self.sigma_bias = torch.nn.Parameter(torch.Tensor(out_features).fill_(sigma_init))
			self.register_buffer("epsilon_bias", torch.zeros(out_features))
		self.reset_parameters()

	def reset_parameters(self):
		std = math.sqrt(3 / self.in_features)
		torch.nn.init.uniform_(self.weight, -std, std)
		torch.nn.init.uniform_(self.bias, -std, std)

	def forward(self, input):
		torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
		bias = self.bias
		if bias is not None:
			torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
			bias = bias + self.sigma_bias * torch.autograd.Variable(self.epsilon_bias)
		weight = self.weight + self.sigma_weight * torch.autograd.Variable(self.epsilon_weight)
		return torch.nn.functional.linear(input, weight, bias)

def init_weights(m):
	if type(m) == torch.nn.Linear:
		torch.nn.init.normal_(m.weight, mean=0., std=0.1)
		torch.nn.init.constant_(m.bias, 0.1)

def get_checkpoint_path(net="qlearning", dirname="pytorch", name="checkpoint"):
	return f"./saved_models/{net}/{dirname}/{name}.pth"

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gumbel_softmax_sample(logits, temperature):
	""" Draw a sample from the Gumbel-Softmax distribution"""
	y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
	return torch.nn.functional.softmax(y / temperature, dim=-1)

# modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
def gsoftmax(logits, temperature=1.0, hard=True):
	"""Sample from the Gumbel-Softmax distribution and optionally discretize.
	Args:
	logits: [batch_size, n_class] unnormalized log-probs
	temperature: non-negative scalar
	hard: if True, take argmax, but differentiate w.r.t. soft sample y
	Returns:
	[batch_size, n_class] sample from the Gumbel-Softmax distribution.
	If hard=True, then the returned sample will be one-hot, otherwise it will
	be a probabilitiy distribution that sums to 1 across classes
	"""
	y = gumbel_softmax_sample(logits, temperature)
	if hard:
		y_hard = one_hot(y)
		y = (y_hard - y).detach() + y
	return y

def one_hot(logits):
	return (logits == logits.max(-1, keepdim=True)[0]).float().to(logits.device)

def one_hot_from_indices(indices, depth, keepdims=False):
	y_onehot = torch.zeros([*indices.shape, depth]).to(indices.device)
	y_onehot.scatter_(-1, indices.unsqueeze(-1).long(), 1)
	return y_onehot.float() if keepdims else y_onehot.squeeze(-2).float()
