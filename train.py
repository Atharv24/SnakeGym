from actor_critic import Agent
import torch
from utils.multiprocessing_env import SubprocVecEnv
#from multiprocessing_env import SubprocVecEnv
import numpy as np
from SnakeEnv import SnakeEnv
import configparser
import gym

config = configparser.ConfigParser()
config.read('config.ini')

ppo_parameters = config['DEFAULT']
network_parameters = config['NETWORK_PARAMETERS']

NUM_ENVS            = int(ppo_parameters['NUM_ENVS'])
PPO_STEPS           = int(ppo_parameters['PPO_STEPS'])
GAE_LAMBDA          = 0.95

TEST_EPOCHS         = 10
TEST_FREQ           = int(ppo_parameters['TEST_FREQ'])
MIN_TEST            = 8
TARGET_REWARD       = 20

RENDER_TESTING = int(ppo_parameters['RENDER_TESTING'])
RENDER_WAIT_TIME = int(ppo_parameters['RENDER_WAIT_TIME'])


N_ACTIONS = 3
GRIDSIZE = 16
VISION_RADIUS = 8
INITIAL_LENGTH = 5

def make_env():
    # returns a function which creates a single environment
    def _thunk():
        env = SnakeEnv(GRIDSIZE, VISION_RADIUS, INITIAL_LENGTH, channel_first=True)
        return env
    return _thunk

def test_env(env, agent):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        action, _, value = agent.choose_action(state)
        if RENDER_TESTING:
            env.render(value.detach().cpu().numpy()[0])
            env.renderVision()
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        state = next_state
        total_reward += reward
    return total_reward

def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


def compute_gae(next_value, rewards, masks, values, gamma=float(ppo_parameters['GAMMA']), lam=GAE_LAMBDA):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        # prepend to get correct order back
        returns.insert(0, gae + values[step])
    return returns

if __name__ == '__main__':

    agent = Agent(n_actions = N_ACTIONS, input_channels=3, ppo_parameters=ppo_parameters, network_parameters=network_parameters)
    agent.load_model('trained_models/FullGrid')

    envs = [make_env() for i in range(NUM_ENVS)]
    envs = SubprocVecEnv(envs)
    env = SnakeEnv(GRIDSIZE, VISION_RADIUS, INITIAL_LENGTH, renderWait = RENDER_WAIT_TIME, channel_first=True)

    state = envs.reset()
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

        for _ in range(PPO_STEPS):
            state = torch.FloatTensor(state).to(agent.device)
            action, log_prob, value = agent.choose_action(state)
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(agent.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(agent.device))
            
            states.append(state)
            actions.append(action)
            
            state = next_state
            frame_idx+=1

        next_state = torch.FloatTensor(next_state).to(agent.device)
        _, _, next_value = agent.choose_action(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantages = returns - values
        advantages = normalize(advantages)

        agent.learn(frame_idx = frame_idx, states=states, actions=actions, log_probs=log_probs, advantages=advantages, returns=returns)
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
                    if games_cleared == MIN_TEST:
                        early_stop = True
                        print("Agent trained successfully!")
                        break
            print(f"The Agent cleared {games_cleared}/{MIN_TEST} games this update!")
            print(f"Average Reward: {reward_sum/TEST_EPOCHS}", "\n")

    if early_stop:
        RENDER_TESTING = True
        for _ in range(TEST_EPOCHS):
            print(test_env(env, agent))