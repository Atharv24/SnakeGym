import torch
import configparser


from actor_critic import Agent
from SnakeEnv import SnakeEnv

config = configparser.ConfigParser()
config.read('config.ini')

network_parameters = config['NETWORK_PARAMETERS']
ppo_parameters = config['DEFAULT']

agent = Agent(n_actions = 3, input_channels=3, ppo_parameters=ppo_parameters, network_parameters=network_parameters)
env = SnakeEnv(16, 6, 5, 1, 100, True)

agent.load_model('trained_models/SeventhRun')    

for _ in range(100000):
    hidden = (torch.zeros(1, 128).to(agent.device), torch.zeros(1, 128).to(agent.device))
    state = env.reset()
    done = False
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        action, _, value, hidden = agent.choose_action(state, hidden)
        env.render(value.detach().cpu().numpy()[0])
        env.renderVision()
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        state = next_state