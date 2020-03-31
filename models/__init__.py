from utils.rand import RandomAgent
from models.singleagent.ppo import PPOAgent
from models.singleagent.ddpg import DDPGAgent, EPS_MIN
from models.singleagent.sac import SACAgent
from models.singleagent.ddqn import DDQNAgent

all_models = {"ppo":PPOAgent, "ddpg":DDPGAgent, "sac":SACAgent, "ddqn":DDQNAgent, "rand":RandomAgent}
