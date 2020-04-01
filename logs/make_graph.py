import os
import re
import numpy as np
from collections import deque
import matplotlib.pylab as plt
import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
matplotlib.rcParams['pdf.fonttype'] = 42


def read_ctrl(path):
	bests = []
	rolling = []
	avgs = deque(maxlen=100)
	with open(path, "r") as f:
		for line in f:
			match = re.match("Ep.*score: (.*), Min: (.*), Avg: ([^,]*)", line.strip('\n'))
			if match:
				bests.append(float(match.groups()[2]))
				avgs.append(float(match.groups()[2]))
				rolling.append(np.mean(avgs))
				while len(avgs)<100: avgs.append(avgs[-1])
	return bests, rolling, [1000*s for s in range(len(rolling))]

def read_a3c(path):
	rewards = []
	avgs = deque(maxlen=100)
	rolling = []
	with open(path, "r") as f:
		for line in f:
			match = re.match("^Ep.*Test: ([^ ]*).*, Avg: ([^ ]*)", line.strip('\n'))
			if match:
				rewards.append(float(match.groups()[0]))
				avgs.append(float(match.groups()[0]))
				rolling.append(np.mean(avgs))
	return rewards, rolling

def read_cdc(path):
	steps = []
	rewards = []
	rolling = []
	avgs = deque(maxlen=100)
	with open(path, "r") as f:
		for line in f:
			match = re.match("^Step:\s*([0-9]+),.* Reward: ([^ ]*) ", line.strip('\n'))
			if match:
				steps.append(int(match.groups()[0]))
				rewards.append(float(match.groups()[1]))
				avgs.append(float(match.groups()[1]))
				rolling.append(np.mean(avgs))
				while len(avgs)<100: avgs.append(avgs[-1])
	return rewards, rolling, steps

def graph_ctrl():
	bests, rolling = zip(*[read_ctrl(f"./logs/controller/{path}") for path in ctrls])
	plt.plot(range(len(bests[-1])), bests[-1], color="#ADFF2F", linewidth=0.5, label="Best of Baseline")
	plt.plot(range(len(bests[0])), bests[0], color="#00BFFF", linewidth=0.5, label="Best of Iteration 1")
	plt.plot(range(len(bests[1])), bests[1], color="#FF1493", linewidth=0.5, label="Best of Iteration 2")
	plt.plot(range(len(rolling[-1])), rolling[-1], color="#008000", label="Avg of Baseline")
	plt.plot(range(len(rolling[0])), rolling[0], color="#0000CD", label="Avg of Iteration 1")
	plt.plot(range(len(rolling[1])), rolling[1], color="#FF0000", label="Avg of Iteration 2")
	print(f"Max-1: {max(bests[-1]):.0f}, Max0: {max(bests[0]):.0f}, Max1: {max(bests[1]):.0f}")
	print(f"Avg-1: {max(rolling[-1]):.0f}, Avg0: {max(rolling[0]):.0f}, Avg1: {max(rolling[1]):.0f}")
	
	plt.legend(loc="best", bbox_to_anchor=(0.6,0.5))
	plt.title("CMA-ES Best Rewards")
	plt.xlabel("Generation")
	plt.ylabel("Best Total Score")
	plt.grid(linewidth=0.3, linestyle='-')

def graph_a3c(model, logs):
	# _, ravgs = read_a3c("./logs/qlearning/random.txt")
	rewards, qavgs = zip(*[read_a3c(f"./logs/{model}/{path}") for path in logs])
	plt.plot(range(len(rewards[-1])), rewards[-1], color="#ADFF2F", linewidth=0.5, label="Baseline")
	plt.plot(range(len(rewards[0])), rewards[0], color="#00BFFF", linewidth=0.5, label="Using Iteration 1 WM")
	plt.plot(range(len(rewards[1])), rewards[1], color="#FF1493", linewidth=0.5, label="Using Iteration 2 WM")
	plt.plot(range(len(qavgs[-1])), qavgs[-1], color="#008000", label="Avg Baseline")
	plt.plot(range(len(qavgs[0])), qavgs[0], color="#0000CD", label="Avg Using Iteration 1 WM")
	plt.plot(range(len(qavgs[1])), qavgs[1], color="#FF0000", label="Avg Using Iteration 2 WM")
	# plt.plot(range(len(ravgs[:len(rewards[0])])), ravgs[:len(rewards[0])], color="#FF0000", label="Avg Random")
	print(f"Max-1: {max(rewards[-1]):.0f}, Max0: {max(rewards[0]):.0f}, Max1: {max(rewards[1]):.0f}")
	print(f"Avg-1: {max(qavgs[-1]):.0f}, Avg0: {max(qavgs[0]):.0f}, Avg1: {max(qavgs[1]):.0f}")
	
	plt.legend(loc="upper left" if model=="ppo" else "best")
	plt.title(f"{model.upper()} Training Rewards")
	plt.xlabel("Rollout")
	plt.ylabel("Total Score")
	plt.grid(linewidth=0.3, linestyle='-')

def graph_CDC():
	logs = {
		"ctrl": {"CarRacing-v0": {"pytorch":-1, "iter0":-1, "iter1":  9}, "take_cover": {"pytorch":-1, "iter0": 3, "iter1":-1}, "defend_the_line": {"pytorch":-1, "iter0":-1, "iter1": 1}},
		"ddpg": {"CarRacing-v0": {"pytorch": 4, "iter0":-1, "iter1": 31}, "take_cover": {"pytorch": 4, "iter0": 0, "iter1":-1}, "defend_the_line": {"pytorch": 4, "iter0":-1, "iter1": 1}},
		"ddqn": {"CarRacing-v0": {"pytorch":-1, "iter0":-1, "iter1": -1}, "take_cover": {"pytorch": 2, "iter0": 2, "iter1":-1}, "defend_the_line": {"pytorch": 0, "iter0":-1, "iter1": 1}},
		"ppo":  {"CarRacing-v0": {"pytorch": 5, "iter0":-1, "iter1": 61}, "take_cover": {"pytorch": 3, "iter0": 3, "iter1":-1}, "defend_the_line": {"pytorch": 4, "iter0":-1, "iter1": 0}},
		"sac":  {"CarRacing-v0": {"pytorch": 4, "iter0":-1, "iter1":  4}, "take_cover": {"pytorch": 1, "iter0": 0, "iter1":-1}, "defend_the_line": {"pytorch": 0, "iter0":-1, "iter1": 0}}
	}
	env_names = ["CarRacing-v0", "take_cover", "defend_the_line"]
	models = ["ctrl", "ddqn", "ddpg", "ppo", "sac"]
	lighter_cols = ["#EEEEEE", "#FFED44", "#44DFFF", "#FF4493", "#BDFF4F"]
	light_cols = ["#CCCCCC", "#FFED00", "#00BFFF", "#FF1493", "#ADFF2F"]
	dark_cols = ["#777777", "#FFA500", "#0000CD", "#FF0000", "#008000"]
	iternums = ["pytorch", "iter0", "iter1"]
	for env_name in env_names:
		fig = plt.figure()
		# fig.set_size_inches(w=3.75, h=2.7)
		for it in iternums:
			for m,model in enumerate(models):
				files = [logs[model][env_name][it]]
				if files[0] >= 0:
					wm = "iter" in it
					tag = "(WM)" if wm else "(Conv)"
					dirname = f"./logs/{model}/{env_name}/{it}/logs_{files[0]}.txt"
					rewards, rolling, steps = zip(*[(read_ctrl if m==0 else read_cdc)(dirname) for j,f in enumerate(files)])
					plt.plot(steps[0], rewards[0], ls="-" if wm else ":", color=light_cols[m], linewidth=0.5, zorder=0)
					plt.plot(steps[0], rolling[0], ls="-" if wm else "--", color=dark_cols[m], label=f"Avg {model.upper()} {tag}", zorder=1)
					print(dirname)
		plt.title(f"Average Training Rewards for {env_name}")
		plt.xlabel("Environment Step")
		plt.ylabel("Evaluation Reward")
		plt.legend(prop={'size': 10 - env_names.index(env_name)})
		plt.savefig(f"./logs/graphs/{env_name}.pdf", bbox_inches='tight')
		plt.show()
		# plt.savefig(f"./logs/graphs/{env_name}.png")


def main():
	# graph_ctrl()
	# plt.figure()
	# graph_a3c("ddpg", ddpgs)
	# plt.figure()
	# graph_a3c("ppo", ppos)
	graph_CDC()

main()