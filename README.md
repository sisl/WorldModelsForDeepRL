# WorldModelsForDeepRL

## Introduction

This repository incorporates the World Model generative model architecture developed by Ha & Schmidhuber [here](https://worldmodels.github.io/) to enhance Deep Reinforcement Learning agents by reducing the dimensionality of the raw image states to a compressed feature vector. We consider Deep Deterministic Policy Gradients (DDPG) and Proximal Policy Optimization (PPO).

## Setup

Follow these steps to install the repository dependencies

1. Clone the repository with 
**`git clone https://github.com/shawnmanuel000/WorldModelsForDeepRL.git`**

2. Cd into the repository root directory and install the required packages in Python3 with
**`pip3 install -r requirements.txt`**

3. In case the Box2D fails to build, you may be missing 'swig'. If so, follow the instructions in ./dependencies/swig_{OS}/swig3.0.8/Doc/Manual/preface.html

4. Test the three saved models by running
**`python3 visualize.py`**

## Training

1. To train the RL agents asynchronously, run the following line with either ddpg or ppo
**`bash train_a3c.sh [ddpg|ppo]`**

2. To train the RL agents synchronously, run the following line with either ddpg or ppo
**`python3 train_a3c.py --model [ddpg|ppo] --runs 500`**

2. To train the complete World Model with controller, run the following line
**`bash train_worldmodel.sh`**