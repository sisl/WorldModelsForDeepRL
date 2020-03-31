import torch
import numpy as np
from utils.rand import ReplayBuffer, PrioritizedReplayBuffer
from utils.network import PTACNetwork, PTACAgent, Conv, INPUT_LAYER, ACTOR_HIDDEN, CRITIC_HIDDEN, LEARN_RATE, DISCOUNT_RATE, NUM_STEPS, one_hot_from_indices

BATCH_SIZE = 32					# Number of samples to train on for each train step
PPO_EPOCHS = 2					# Number of iterations to sample batches for training
ENTROPY_WEIGHT = 0.005			# The weight for the entropy term of the Actor loss
CLIP_PARAM = 0.05				# The limit of the ratio of new action probabilities to old probabilities

class PPOActor(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.discrete = type(action_size) != tuple
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)!=3 else Conv(state_size, INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, ACTOR_HIDDEN)
		self.layer3 = torch.nn.Linear(ACTOR_HIDDEN, ACTOR_HIDDEN)
		self.action_mu = torch.nn.Linear(ACTOR_HIDDEN, action_size[-1])
		self.action_sig = torch.nn.Parameter(torch.zeros(action_size[-1]))
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)
		self.dist = lambda m,s: torch.distributions.Categorical(m.softmax(-1)) if self.discrete else torch.distributions.Normal(m,s)
		
	def forward(self, state, action_in=None, sample=True):
		state = self.layer1(state).relu()
		state = self.layer2(state).relu()
		state = self.layer3(state).relu()
		action_mu = self.action_mu(state)
		action_sig = self.action_sig.exp().expand_as(action_mu)
		dist = self.dist(action_mu, action_sig)
		action = dist.sample() if action_in is None else action_in.argmax(-1) if self.discrete else action_in
		action_out = one_hot_from_indices(action, action_mu.size(-1)) if self.discrete else action
		log_prob = dist.log_prob(action)
		entropy = dist.entropy()
		return action_out, log_prob, entropy

class PPOCritic(torch.nn.Module):
	def __init__(self, state_size, action_size):
		super().__init__()
		self.layer1 = torch.nn.Linear(state_size[-1], INPUT_LAYER) if len(state_size)!=3 else Conv(state_size, INPUT_LAYER)
		self.layer2 = torch.nn.Linear(INPUT_LAYER, CRITIC_HIDDEN)
		self.layer3 = torch.nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN)
		self.value = torch.nn.Linear(CRITIC_HIDDEN, 1)
		self.apply(lambda m: torch.nn.init.xavier_normal_(m.weight) if type(m) in [torch.nn.Conv2d, torch.nn.Linear] else None)

	def forward(self, state):
		state = self.layer1(state).relu()
		state = self.layer2(state).relu()
		state = self.layer3(state).relu()
		value = self.value(state)
		return value

class PPONetwork(PTACNetwork):
	def __init__(self, state_size, action_size, actor=PPOActor, critic=PPOCritic, lr=LEARN_RATE, tau=None, gpu=True, load=None, name="ppo"):
		super().__init__(state_size, action_size, actor=actor, critic=critic, lr=lr, gpu=gpu, load=load, name=name)

	def get_action_probs(self, state, action_in=None, grad=False, numpy=False, sample=True):
		with torch.enable_grad() if grad else torch.no_grad():
			action, log_prob, entropy = self.actor_local(state.to(self.device), action_in, sample)
			action_or_entropy = action if action_in is None else entropy.mean()
			return (x.cpu().numpy() if numpy else x for x in [action_or_entropy, log_prob])

	def get_value(self, state, grad=False, numpy=False):
		with torch.enable_grad() if grad else torch.no_grad():
			return self.critic_local(state.to(self.device)).cpu().numpy() if numpy else self.critic_local(state.to(self.device))

	def optimize(self, states, actions, old_log_probs, targets, advantages, clip_param=CLIP_PARAM, e_weight=ENTROPY_WEIGHT, scale=1):
		values = self.get_value(states, grad=True)
		critic_loss = (values - targets).pow(2) * scale
		self.step(self.critic_optimizer, critic_loss.mean())

		entropy, new_log_probs = self.get_action_probs(states, actions, grad=True)
		ratio = (new_log_probs - old_log_probs).exp()
		ratio_clipped = torch.clamp(ratio, 1.0-clip_param, 1.0+clip_param)
		actor_loss = -(torch.min(ratio*advantages, ratio_clipped*advantages) + e_weight*entropy) * scale
		self.step(self.actor_optimizer, actor_loss.mean())

class PPOAgent(PTACAgent):
	def __init__(self, state_size, action_size, lr=LEARN_RATE, update_freq=NUM_STEPS, gpu=True, load=None):
		super().__init__(state_size, action_size, PPONetwork, lr=lr, update_freq=update_freq, gpu=gpu, load=load)

	def get_action(self, state, eps=None, sample=True):
		self.action, self.log_prob = self.network.get_action_probs(self.to_tensor(state), numpy=True, sample=sample)
		return np.tanh(self.action)

	def train(self, state, action, next_state, reward, done):
		self.buffer.append((state, self.action, self.log_prob, reward, done))
		if np.any(done[0]) or len(self.buffer) >= self.update_freq:
			states, actions, log_probs, rewards, dones = map(self.to_tensor, zip(*self.buffer))
			self.buffer.clear()
			states = torch.cat([states, self.to_tensor(next_state).unsqueeze(0)], dim=0)
			values = self.network.get_value(states)
			targets, advantages = self.compute_gae(values[-1], rewards.unsqueeze(-1), dones.unsqueeze(-1), values[:-1], gamma=DISCOUNT_RATE)
			states, actions, log_probs, targets, advantages = [x.view(x.size(0)*x.size(1), *x.size()[2:]) for x in (states[:-1], actions, log_probs, targets, advantages)]
			self.replay_buffer.clear().extend(list(zip(states, actions, log_probs, targets, advantages)), shuffle=True)
			for _ in range((len(self.replay_buffer)*PPO_EPOCHS)//BATCH_SIZE):
				state, action, log_prob, target, advantage = self.replay_buffer.next_batch(BATCH_SIZE, torch.stack)
				self.network.optimize(state, action, log_prob, target, advantage)
				