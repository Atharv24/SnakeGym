from actor_critic import Agent
from utils.plotLearning import plotLearning
from utils.multiprocessing_env import SubprocVecEnv
import numpy as np
from SnakeEnv import SnakeEnv

ppo_parameters = {"HIDDEN_SIZE1" : 512,
                    "HIDDEN_SIZE2" : 256,
                    "LEARNING_RATE" : 1e-5,
                    "GAMMA" : 0.99,
                    "PPO_EPSILON" : 0.2,
                    "ENTROPY_BETA" : 0.001,
                    "MINI_BATCH_SIZE" : 128,
                    "PPO_EPOCHS" : 8,
                    "CRITIC_DISCOUNT" : 0.5
                    }

NUM_ENVS            = 3
PPO_STEPS           = 256
GAE_LAMBDA          = 0.95
TEST_EPOCHS         = 10
NUM_TESTS           = 10
TARGET_REWARD       = 2500

N_ACTIONS = 3
GRIDSIZE = 32
INITIAL_LENGTH = 10

def make_env(renderID):
    # returns a function which creates a single environment
    def _thunk():
        env = SnakeEnv(GRIDSIZE, INITIAL_LENGTH, render=True, renderID=renderID, renderWait = 20)
        return env
    return _thunk

def test_env(env, agent):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    return total_reward

def normalize(x):
    x -= x.mean()
    x /= (x.std() + 1e-8)
    return x


def compute_gae(next_value, rewards, masks, values, gamma=ppo_parameters['GAMMA'], lam=GAE_LAMBDA):
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

    agent = Agent(n_actions = N_ACTIONS, input_dims=[GRIDSIZE, GRIDSIZE], parameters=ppo_parameters)

    envs = [make_env(i) for i in range(NUM_ENVS)]
    envs = SubprocVecEnv(envs)

    state = envs.reset()
    early_stop = False

    while not early_stop:
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []

        for _ in range(PPO_STEPS):
            action, log_prob = agent.choose_action(state)
            value = agent.get_state_value(state)
            next_state, reward, done, _ = envs.step(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            masks.append(1-done)
            
            states.append(state)
            actions.append(action)
            
            state = next_state

        next_value = agent.get_state_value(state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns = np.array(returns)
        log_probs = np.array(log_probs)
        values = np.array(values)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        advantages = returns - values
        advantages = normalize(advantages)


        states = np.reshape(states, (-1, GRIDSIZE, GRIDSIZE))
        log_probs = np.reshape(log_probs, (-1, N_ACTIONS))
        advantages = np.reshape(advantages, (-1, 1))
        rewards = np.reshape(rewards, (-1, 1))
        values = np.reshape(values, (-1, 1))
        actions = np.reshape(actions, (-1, 1))
        returns = np.reshape(returns, (-1, 1))

        agent.learn(states, log_probs, advantages, actions, returns)


