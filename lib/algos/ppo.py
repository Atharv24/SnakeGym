import numpy as np
import datetime
from ..nets.lstm_network import ActorCritic
import torch
import torch.optim as optim
from tqdm import trange

from tensorboardX import SummaryWriter

class Agent(object):
    def __init__(self, agent_name, input_channels, network_parameters, ppo_parameters=None, n_actions=3):
        self.name = agent_name
        self.save_path = 'agents/' + self.name + '/'
        self.n_actions = n_actions
        self.input_channels = input_channels
        self.action_space = [i for i in range(n_actions)]
        
        if ppo_parameters:
            self.gamma = float(ppo_parameters['GAMMA'])
            self.lam = float(ppo_parameters['GAE_LAMBDA'])
            self.alpha = float(ppo_parameters['LEARNING_RATE'])
            self.ppo_epsilon = float(ppo_parameters['PPO_EPSILON'])
            self.entropy_beta = float(ppo_parameters['ENTROPY_BETA'])
            self.minibatch_size = int(ppo_parameters['MINI_BATCH_SIZE'])
            self.ppo_epochs = int(ppo_parameters['PPO_EPOCHS'])
            self.critic_discount = float(ppo_parameters['CRITIC_DISCOUNT'])
            self.training_sequence_length = 4
            self.writer = SummaryWriter(logdir=self.save_path +'logs')

        self.epochs_trained = 0
        
        # Autodetect CUDA
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        print('Device:', self.device, '\n')

        self.network = ActorCritic(input_channels=self.input_channels, n_actions=self.n_actions, parameters=network_parameters).to(self.device)
        print(self.network, '\n')
        
        if ppo_parameters:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.alpha)

    def choose_action(self, observation, hidden):
        dist, value, hidden = self.network(observation, hidden)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, value, hidden

    def compute_gae(self, next_value, rewards, masks, values):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.lam * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def ppo_iter(self, states, actions, log_probs, returns, advantages, hiddens):
        batch_size = states.size(0)
        (hiddens_0, hiddens_1) = hiddens
        for _ in range(batch_size // self.minibatch_size):
            rand_ids = np.random.randint(0, batch_size-9, self.minibatch_size)
            yield [(states[rand_ids+i*3, :], actions[rand_ids+i*3], log_probs[rand_ids+i*3], returns[rand_ids+i*3, :], advantages[rand_ids+i*3, :]) for i in range(self.training_sequence_length)], (hiddens_0[rand_ids, :], hiddens_1[rand_ids])

    def learn(self, frame_idx, states, actions, log_probs, returns, advantages, hiddens):
        count_steps = 0
        sum_returns = 0.0
        sum_advantage = 0.0
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0
        sum_entropy = 0.0
        sum_loss_total = 0.0
        
        t = trange(self.ppo_epochs, desc=f'{self.name} is learning', unit='update', leave=False)
        for _ in t:
            for seq, hidden in self.ppo_iter(states, actions, log_probs, returns, advantages, hiddens):
                loss = torch.zeros([]).to(self.device)

                for (state, action, old_log_probs, return_, advantage) in seq:

                    dist, value, hidden = self.network(state, hidden, grads=True)
                    entropy = dist.entropy().mean()
                    new_log_probs = dist.log_prob(action)

                    ratio = (new_log_probs - old_log_probs).exp()
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * advantage

                    actor_loss  = - torch.min(surr1, surr2).mean()
                    critic_loss = (return_ - value).pow(2).mean()

                    loss += self.critic_discount * critic_loss + actor_loss - self.entropy_beta * entropy

                loss/=self.training_sequence_length

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
        torch.save(self.network.state_dict(), self.save_path + 'saved_model/network_weights.pt')
        torch.save(self.optimizer.state_dict(), self.save_path + 'saved_model/optimizer_weights.pt')

    def load_model(self, testing=False):
        self.network.load_state_dict(torch.load(self.save_path + 'saved_model/network_weights.pt'))
        if not testing:
            self.optimizer.load_state_dict(torch.load(self.save_path + 'saved_model/optimizer_weights.pt'))
        print('Brain succesfully loaded\n')