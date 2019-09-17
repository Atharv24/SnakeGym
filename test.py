import torch
import configparser


from actor_critic import Agent
from SnakeEnv import SnakeEnv

config = configparser.ConfigParser()
config.read('config.ini')

network_parameters = config['NETWORK_PARAMETERS']

agent = Agent(n_actions = 3, input_channels=3, network_parameters=network_parameters)
env = SnakeEnv(11, 5, 5, 0, 100, True)

agent.load_model()    

for _ in range(100000):
    state = env.reset()
    done = False
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        action, _, value = agent.choose_action(state)
        env.render(value.detach().cpu().numpy()[0])
        env.renderVision()
        next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
        state = next_state