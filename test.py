import torch
import configparser
import argparse

from lib.algos.ppo import Agent
from lib.envs.SnakeEnvSP import SnakeEnv

parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', help='Name of experiment to load')
parser.add_argument('-num_games', type=int, default = 5, help='Number of games to test for')
args = parser.parse_args()

exp_name = args.exp_name
exp_path = 'experiments/' + exp_name + '/'

config = configparser.ConfigParser()
config.read(exp_path + 'config.ini')

network_parameters = config['NETWORK_PARAMETERS']
env_parameters = config['ENV_PARAMETERS']

HIDDEN_SIZE = int(network_parameters['HIDDEN_SIZE'])

N_ACTIONS = int(env_parameters['N_ACTIONS'])
GRIDSIZE = int(env_parameters['GRIDSIZE'])
VISION_RADIUS = int(env_parameters['VISION_RADIUS'])
INITIAL_LENGTH = int(env_parameters['INITIAL_LENGTH'])

NUM_GAMES = args.num_games

agent = Agent(n_actions=N_ACTIONS, exp_name=exp_name, input_channels=3, ppo_parameters=None, network_parameters=network_parameters)
env = SnakeEnv(gridSize=GRIDSIZE, visionRadius=VISION_RADIUS, initialSnakeLength=INITIAL_LENGTH, renderID='Test', renderWait=100, channel_first=True)

agent.load_model(testing=True)
if __name__ == '__main__':
    history = []
    for _ in range(NUM_GAMES):
        hidden = (torch.zeros(1, HIDDEN_SIZE).to(agent.device), torch.zeros(1, HIDDEN_SIZE).to(agent.device))
        state = env.reset()
        done = False
        score = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            action, _, value, hidden = agent.choose_action(state, hidden)
            env.render()
            env.renderVision()
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            state = next_state
            score+=reward*(1-done)
        history.append(score)
    
    print(f'Scores for {NUM_GAMES} games:', history)