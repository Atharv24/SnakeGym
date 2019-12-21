from lib.algos.ppo import Agent, compute_gae
import torch
from lib.utils.multiprocessing_env import SubprocVecEnv
import numpy as np
from lib.envs.SnakeEnvSP import SnakeEnv
import configparser
import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-config', help = 'Name of config file to be used')
parser.add_argument('-resume', type=int, default=False, help = '0 to start afresh, 1 to load saved model')
args = parser.parse_args()

resume_training = bool(args.resume)

config_path = 'configs/' +  args.config + '.ini'

config = configparser.ConfigParser()
config.read(config_path)

training_parameters = config['TRAINING_PARAMETERS']
env_parameters = config['ENV_PARAMETERS']
ppo_parameters = config['PPO_PARAMETERS']
network_parameters = config['NETWORK_PARAMETERS']

AGENT_NAME = training_parameters['AGENT_NAME']
agent_directory = os.path.join('agents', AGENT_NAME)

if not os.path.exists(agent_directory):
    os.mkdir(agent_directory)
    os.mkdir(os.path.join(agent_directory, 'saved_model'))
    os.mkdir(os.path.join(agent_directory, 'logs'))

shutil.copyfile(config_path, agent_directory + '/config.ini')

NUM_ENVS = int(training_parameters['NUM_ENVS'])

PPO_STEPS = int(ppo_parameters['PPO_STEPS'])
GAE_LAMBDA = float(ppo_parameters['GAE_LAMBDA'])
GAMMA = float(ppo_parameters['GAMMA'])

TEST_FREQ = int(training_parameters['TEST_FREQ'])
TARGET_REWARD = int(training_parameters['TARGET_REWARD'])
TEST_EPOCHS = 5
MIN_TEST_CLEARED = 4

RENDER_TRAINING = int(training_parameters['RENDER_TRAINING'])
RENDER_TESTING = int(training_parameters['RENDER_TESTING'])
RENDER_WAIT_TIME = int(training_parameters['RENDER_WAIT_TIME'])

HIDDEN_SIZE = int(network_parameters['HIDDEN_SIZE'])

N_ACTIONS = int(env_parameters['N_ACTIONS'])
GRIDSIZE = int(env_parameters['GRIDSIZE'])
VISION_RADIUS = int(env_parameters['VISION_RADIUS'])
INITIAL_LENGTH = int(env_parameters['INITIAL_LENGTH'])

def make_env(renderID):
    # returns a function which creates a single environment
    def _thunk():
        env = SnakeEnv(GRIDSIZE, VISION_RADIUS, INITIAL_LENGTH, renderID=renderID, renderWait=RENDER_WAIT_TIME, channel_first=True)
        return env
    return _thunk

def test_env(env, agent):
    state = env.reset()
    hidden = (torch.zeros(1, HIDDEN_SIZE).to(agent.device), torch.zeros(1, HIDDEN_SIZE).to(agent.device))
    done = False
    total_reward = 0
    steps = 0
    while not done and steps<300:
        state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        action, _, value, hidden = agent.choose_action(state, hidden)
        if RENDER_TESTING:
            env.render(wait=True)
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        state = next_state
        total_reward += reward
        steps+=1
    return total_reward

def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x

if __name__ == '__main__':

    agent = Agent(n_actions = N_ACTIONS, agent_name=AGENT_NAME, input_channels=3, ppo_parameters=ppo_parameters, network_parameters=network_parameters)
    if resume_training:
        print('Resuming Training\n')
        agent.load_model()
    else:
        print('Initializing Brain\n')

    envs = [make_env(i+1) for i in range(NUM_ENVS)]
    envs = SubprocVecEnv(envs)
    env = SnakeEnv(GRIDSIZE, VISION_RADIUS, INITIAL_LENGTH, renderWait=RENDER_WAIT_TIME, renderID='Test', channel_first=True)

    state = envs.reset()
    hidden = [torch.zeros(NUM_ENVS, HIDDEN_SIZE).to(agent.device), torch.zeros(NUM_ENVS, HIDDEN_SIZE).to(agent.device)]
    early_stop = False
    
    training_epochs = 0
    frame_idx = 0

    while not early_stop:
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        hiddens_0 = []
        hiddens_1 = []

        for _ in range(PPO_STEPS):
            hiddens_0.append(hidden[0])
            hiddens_1.append(hidden[1])
            state = torch.FloatTensor(state).to(agent.device)
            action, log_prob, value, hidden = agent.choose_action(state, hidden)
            if RENDER_TRAINING:
                envs.render()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(agent.device))
            mask = torch.FloatTensor(1-done).unsqueeze(1).to(agent.device)
            hidden = (hidden[0]*mask, hidden[1]*mask)
            masks.append(mask)
            
            states.append(state)
            actions.append(action)
            
            state = next_state
            frame_idx+=1

        next_state = torch.FloatTensor(next_state).to(agent.device)
        _, _, next_value, hidden = agent.choose_action(next_state, hidden)
        returns = compute_gae(next_value, rewards, masks, values, gamma=GAMMA, lam=GAE_LAMBDA)

        returns   = torch.cat(returns)
        log_probs = torch.cat(log_probs)
        values    = torch.cat(values)
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantages = returns - values
        advantages = normalize(advantages)
        hiddens_0 = torch.cat(hiddens_0)
        hiddens_1 = torch.cat(hiddens_1)

        agent.learn(frame_idx = frame_idx, states=states, actions=actions, log_probs=log_probs, advantages=advantages, returns=returns, hiddens=(hiddens_0, hiddens_1))
        training_epochs+=1

        if training_epochs%TEST_FREQ == 0:
            agent.save_model()
            test_rewards = []
            reward_sum = 0
            games_cleared = 0
            for _ in range(TEST_EPOCHS):
                total_reward = test_env(env, agent)
                reward_sum+=total_reward
                test_rewards.append(total_reward)
                if total_reward >= TARGET_REWARD:
                    games_cleared +=1
                    if games_cleared == MIN_TEST_CLEARED:
                        early_stop = True
                        print("Agent trained successfully!")
                        break
            print(f"The Agent cleared {games_cleared}/{MIN_TEST_CLEARED} games this update!")
            print(f"Average Reward: {reward_sum/TEST_EPOCHS}", "\n")

    if early_stop:
        RENDER_TESTING = True
        for _ in range(TEST_EPOCHS):
            print(test_env(env, agent))