import os
import gym
import torch
import argparse
import numpy as np
from envs import make_env, all_envs, env_name
from models import all_models, EPS_MIN
from utils.rand import RandomAgent
from utils.misc import Logger, rollout
from utils.envs import EnsembleEnv, EnvManager, EnvWorker
from utils.wrappers import WorldACAgent
from utils.multiprocess import set_rank_size

TRIAL_AT = 1000
SAVE_AT = 1

def train(make_env, model, ports, steps, checkpoint=None, save_best=True, log=True, render=False, worldmodel=True):
	envs = (EnvManager if len(ports)>0 else EnsembleEnv)(make_env, ports if ports else 4)
	agent = WorldACAgent(envs.state_size, envs.action_size, model, envs.num_envs, load=checkpoint, gpu=True, worldmodel=worldmodel) 
	logger = Logger(model, checkpoint, num_envs=envs.num_envs, state_size=agent.state_size, action_size=envs.action_size, action_space=envs.env.action_space, envs=type(envs), statemodel=agent.state_model)
	states = envs.reset(train=True)
	total_rewards = []
	for s in range(steps+1):
		env_actions, actions, states = agent.get_env_action(envs.env, states)
		next_states, rewards, dones, _ = envs.step(env_actions, train=True)
		agent.train(states, actions, next_states, rewards, dones)
		states = next_states
		if s%TRIAL_AT==0:
			rollouts = rollout(envs, agent, render=render)
			total_rewards.append(np.round(np.mean(rollouts, axis=-1), 3))
			if checkpoint and len(total_rewards)%SAVE_AT==0: agent.save_model(checkpoint)
			if checkpoint and save_best and np.all(total_rewards[-1] >= np.max(total_rewards, axis=-1)): agent.save_model(checkpoint, "best")
			if log: logger.log(f"Step: {s:7d}, Reward: {total_rewards[-1]} [{np.std(rollouts):4.3f}], Avg: {round(np.mean(total_rewards, axis=0),3)} ({agent.acagent.eps:.4f})")
	envs.close()

def trial(make_env, model, ports, steps, checkpoint=None, render=False, worldmodel=True):
	envs = (EnvManager if len(ports)>0 else EnsembleEnv)(make_env, ports if ports else 4)
	agent = WorldACAgent(envs.state_size, envs.action_size, model, envs.num_envs, load=checkpoint, train=False, gpu=True, worldmodel=worldmodel)#.load(checkpoint,"best")
	rollouts = [rollout(envs, agent, eps=EPS_MIN, render=render) for _ in range(10)]
	print(f"Reward: {np.mean(rollouts)} +- {np.std(rollouts)}")
	envs.close()

def parse_args(all_envs, all_models):
	parser = argparse.ArgumentParser(description="A3C Trainer")
	parser.add_argument("--env_name", type=str, default=env_name, choices=all_envs, help="Name of the environment to use. Allowed values are:\n"+', '.join(all_envs), metavar="env_name")
	parser.add_argument("--model", type=str, default="ppo", choices=all_models, help="Which RL algorithm to use. Allowed values are:\n"+', '.join(all_models), metavar="model")
	parser.add_argument("--iternum", type=int, default=-1, choices=[-1,0,1], help="Whether to train using World Model to load (0 or 1) or raw images (-1)")
	parser.add_argument("--tcp_ports", type=int, default=[], nargs="+", help="The list of worker ports to connect to")
	parser.add_argument("--tcp_rank", type=int, default=0, help="Which port to listen on (as a worker server)")
	parser.add_argument("--render", action="store_true", help="Whether to render an environment rollout")
	parser.add_argument("--trial", action="store_true", help="Whether to show a trial run training on the Pendulum-v0 environment")
	parser.add_argument("--steps", type=int, default=100000, help="Number of steps to train the agent")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args(all_envs, all_models.keys())
	checkpoint = f"{args.env_name}/pytorch" if args.iternum < 0 else f"{args.env_name}/iter{args.iternum}"
	rank, size = set_rank_size(args.tcp_rank, args.tcp_ports)
	get_env = lambda: make_env(args.env_name, args.render)
	model = all_models[args.model]
	if rank>0:
		EnvWorker(make_env=get_env).start()
	else:
		function = trial if args.trial else train
		function(make_env=get_env, model=model, ports=list(range(1,size)), steps=args.steps, checkpoint=checkpoint, render=args.render, worldmodel=args.iternum>=0)
