import cma
import gym
import pickle
import argparse
import numpy as np
import socket as Socket
from envs import make_env, env_name
from torchvision import transforms
from models.worldmodel.vae import VAE, LATENT_SIZE
from models.worldmodel.mdrnn import MDRNNCell, HIDDEN_SIZE, ACTION_SIZE
from models.worldmodel.controller import Controller, ControlAgent
from utils.multiprocess import get_server, get_client, set_rank_size
from utils.misc import rollout, IMG_DIM, Logger

parser = argparse.ArgumentParser(description="Controller Trainer")
parser.add_argument("--iternum", type=int, default=0, help="Which iteration of trained World Model to load")
parser.add_argument("--tcp_ports", type=int, default=[], nargs="+", help="The list of worker ports to connect to")
parser.add_argument("--tcp_rank", type=int, default=0, help="Which port to listen on (as a worker server)")
args = parser.parse_args()

def get_env():
	env = make_env()
	state_size = [LATENT_SIZE+HIDDEN_SIZE]
	action_size = [env.action_space.n] if hasattr(env.action_space, 'n') else env.action_space.shape
	return env, state_size, action_size

class ControllerWorker():
	def start(self, load_dirname, gpu=True, iterations=1):
		conn = get_server()
		env, state_size, action_size = get_env()
		agent = ControlAgent(state_size, action_size, gpu=gpu, load=load_dirname)
		episode = 0
		while True:
			data = conn.recv()
			if not "cmd" in data or data["cmd"] != "ROLLOUT": break
			score = np.mean([rollout(env, agent.set_params(data["item"]), render=False) for _ in range(iterations)])
			conn.send(score)
			print(f"Ep: {episode}, Score: {score:.4f}")
			episode += 1
		env.close()

class ControllerManager():
	def start(self, ports, save_dirname, popsize, epochs=1000):
		conn = get_client(ports)
		def get_scores(params):
			scores = []
			for i in range(0, len(params), conn.num_clients):
				conn.broadcast([{"cmd": "ROLLOUT", "item": params[i+j]} for j in range(conn.num_clients)])
				scores.extend(conn.gather())
			return scores
		popsize = (max(popsize//conn.num_clients, 1))*conn.num_clients
		train(save_dirname, get_scores, epochs, popsize=popsize)

def train(save_dirname, get_scores, epochs=250, popsize=4, restarts=1):
	env, state_size, action_size = get_env()
	agent = Controller(state_size, action_size, gpu=False, load=False)
	logger = Logger(Controller, save_dirname, popsize=popsize, restarts=restarts)
	best_solution = (agent.get_params(), -np.inf)
	total_scores = []
	env.close()
	for run in range(restarts):
		start_epochs = epochs//restarts
		es = cma.CMAEvolutionStrategy(best_solution[0], 0.1, {"popsize": popsize})
		while not es.stop() and start_epochs > 0:
			start_epochs -= 1
			params = es.ask()
			scores = get_scores(params)
			total_scores.append(np.mean(scores))
			es.tell(params, [-s for s in scores])
			best_index = np.argmax(scores)
			best_params = (params[best_index], scores[best_index])
			if best_params[1] > best_solution[1]:
				agent.set_params(best_params[0]).save_model(save_dirname)
				best_solution = best_params
			logger.log(f"Ep: {run}-{start_epochs}, Best score: {best_params[1]:3.4f}, Min: {np.min(scores):.4f}, Avg: {total_scores[-1]:.4f}, Rolling: {np.mean(total_scores):.4f}")

def run(load_dirname, gpu=True, iterations=1):
	env, state_size, action_size = get_env()
	agent = ControlAgent(state_size, action_size, gpu=gpu, load=load_dirname)
	get_scores = lambda params: [np.mean([rollout(env, agent.set_params(p)) for _ in range(iterations)]) for p in params]
	train(load_dirname, get_scores, 1000)
	env.close()

if __name__ == "__main__":
	dirname = f"{env_name}/pytorch" if args.iternum < 0 else f"{env_name}/iter{args.iternum}"
	rank, size = set_rank_size(args.tcp_rank, args.tcp_ports)
	if rank>0:
		ControllerWorker().start(load_dirname=dirname, gpu=True, iterations=1)
	elif rank==0 and size>1:
		ControllerManager().start(ports=list(range(1,size)), save_dirname=dirname, epochs=500, popsize=64)
	else:
		run(dirname, gpu=False)