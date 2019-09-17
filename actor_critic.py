import numpy as np
import datetime
from conv_network import ActorCritic
import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

class Agent(object):
    def __init__(self, input_channels, network_parameters, ppo_parameters=None, n_actions=3):
        if ppo_parameters:
            self.gamma = float(ppo_parameters['GAMMA'])
            self.alpha = float(ppo_parameters['LEARNING_RATE'])
            self.ppo_epsilon = float(ppo_parameters['PPO_EPSILON'])
            self.entropy_beta = float(ppo_parameters['ENTROPY_BETA'])
            self.minibatch_size = int(ppo_parameters['MINI_BATCH_SIZE'])
            self.ppo_epochs = int(ppo_parameters['PPO_EPOCHS'])
            self.critic_discount = float(ppo_parameters['CRITIC_DISCOUNT'])
            self.save_path = ppo_parameters['SAVE_PATH']
        
        self.n_actions = n_actions
        self.input_channels = input_channels
        self.action_space = [i for i in range(n_actions)]
        
        self.writer = SummaryWriter()

        self.epochs_trained = 0
        
        # Autodetect CUDA
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")
        print('Device:', self.device)

        self.network = ActorCritic(3, self.n_actions, network_parameters).to(self.device)
        print(self.network)
        
        if ppo_parameters:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.alpha)

    def choose_action(self, observation):
        dist, value = self.network(observation)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value

    def ppo_iter(self, states, actions, log_probs, returns, advantages):
        batch_size = states.size(0)

        for _ in range(batch_size // self.minibatch_size):
            rand_ids = np.random.randint(0, batch_size, self.minibatch_size)
            yield states[rand_ids, :], actions[rand_ids], log_probs[rand_ids], returns[rand_ids, :], advantages[rand_ids, :]

    def learn(self, frame_idx, states, actions, log_probs, returns, advantages):
        count_steps = 0
        sum_returns = 0.0
        sum_advantage = 0.0
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0
        sum_entropy = 0.0
        sum_loss_total = 0.0
        
        for _ in range(self.ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(states, actions, log_probs, returns, advantages):
                dist, value = self.network(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = self.critic_discount * critic_loss + actor_loss - self.entropy_beta * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                sum_returns += return_.mean()
                sum_advantage += advantage.mean()
                sum_loss_actor += actor_loss
                sum_loss_critic += critic_loss
                sum_loss_total += loss
                sum_entropy += entropy

                count_steps+=1
        
        self.writer.add_scalar("returns", sum_returns / count_steps, frame_idx)
        self.writer.add_scalar("advantage", sum_advantage / count_steps, frame_idx)
        self.writer.add_scalar("loss_actor", sum_loss_actor / count_steps, frame_idx)
        self.writer.add_scalar("loss_critic", sum_loss_critic / count_steps, frame_idx)
        self.writer.add_scalar("entropy", sum_entropy / count_steps, frame_idx)
        self.writer.add_scalar("loss_total", sum_loss_total / count_steps, frame_idx)
        self.writer.flush()
    

    def save_model(self):
        torch.save(self.network.state_dict(), self.save_path+'.pt')
        torch.save(self.optimizer.state_dict(), self.save_path+'-opt.pt')
    def load_model(self, path):
        self.network.load_state_dict(torch.load(path+".pt"))
        self.optimizer.load_state_dict(torch.load(path+"-opt.pt"))