# SnakeGym
Gym environment based on the classic Snake game. Based on PyTorch and supports parallel data collection from each core of your CPU.

## Foreword
I have made this project as modular as possible, allowing easy adjustments to network architecture, learning algorithm, and even for quickly switching from one Gym environment to another.

## Contents
1. [Setup Instructions and Dependencies](#1-setup-instructions-and-dependencies)
2. [Training Agents](#2-training-agents)
3. [Testing Agents](#3-testing-agents)
4. [Repository Overview](#4-repository-overview)

## 1. Setup Instructions and Dependencies
You may setup the repository on your local machine by either downloading it or running the following line on `terminal`.

``` Batchfile
git clone https://github.com/Atharv24/SnakeGym.git
```

All dependencies required by this repo are listed in `requirements.txt`.
You can install them manually or setup a virtual environment with Python 3.6 by running

``` Batchfile
pip install -r requirements.txt
```

Make sure to install latest CUDA and cuDNN if you want to use GPU acceleration.

## 2. Training Agents
To train your agent from scratch
+ Setup your parameters in a config file and move it to `configs` folder.
+ Run the following command in `terminal`
```Batchfile
python train.py -config name_of_config
```
For example :
```Batchfile
python train.py -config Default_SP
```
Will start training with the default parameters set by me. 

+ Data for each agent will be stored in `agents` folder.
+ Network and optimizer weights will be saved whenever the agent is evaluated. You can set this frequency with `TEST_FREQ` parameter in the config file.
+ To resume training from last saved checkpoint, add `--resume` in `terminal`.
```Batchfile
python train.py -config myConfig --resume
```
+ For more efficient data collection, increase the `NUM_ENVS` parameter in the config file. (Using more cores than what your CPU has will throw an error)
+ The `RENDER_TRAINING` and `RENDER_TESTING` parameters are self explanatory. Note that disabling rendering will lead to faster data collection and hence faster training.
+ You can also set the `RENDER_WAIT_TIME` to your need. Increasing this parameter will reduce the FPS and vice-versa.
+ TensorboardX logs can be found in the `logs` subdirectory in your agent's folder.
+ The training will conclude when the agent reaches the `TARGET_REWARD` in 4 out of 5 test games.

## 3. Testing Agents
To test a trained agent run
```Batchfile
python test.py -agent_name AGENT_NAME -num_games N
```
For example
```Batchfile
python test.py -agent_name Default_SP -num_games 10
```
+ The `-num_games` parameter is optional and will default to 5 if not provided.
+ At the end of testing scores for each game will be printed in the `terminal`.

## 4. Repository Overview
+ **agents**: Folder containing data for each agent.

  Each agent folder contains:
  + **logs**: TensorboardX logs containing the critic and actor losses, action space entropy and agent advantage estimates.
  + **saved_models**: Network and optimizer weights saved in *.pt format for testing/resuming training.
  + `config.ini`: The same configuration file used for training.
+ **configs**: Folder containing configuration files in *.ini format.
+ **lib**: Folder containing scripts used for training and testing.

  + **algos**: Algorithms used for training the agent.  
  Currently only has Proximal Policy Optimization (PPO).
  
  + **envs**: Gym environments for data collection.  
  Single player Snake is ready and Multiplayer is a WIP.
  + **nets**: Neural network architecture for agent's brain.
  + **utils**: Misc utils required by the project. Currently only has `multiprocessing_env.py` that I took and modified from OpenAI baselines.

+ `train.py`: Script for [training agents](#2-training-agents).
+ `test.py`: Script for [testing agents](#3-testing-agents).