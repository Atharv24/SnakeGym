import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, input_channels, n_actions, parameters):
        super(ActorCritic, self).__init__()
        self.input_channels = input_channels
        self.conv1_filters = int(parameters['CONV1_FILTERS'])
        self.conv2_filters = int(parameters['CONV2_FILTERS'])
        self.conv_filter_size = int(parameters['CONV_FILTER_SIZE'])
        self.stride = int(parameters['CONV_STRIDE'])
        self.hidden_size = int(parameters['HIDDEN_SIZE'])


        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=self.conv1_filters, kernel_size=self.conv_filter_size, stride=self.stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.conv1_filters, out_channels=self.conv2_filters, kernel_size=self.conv_filter_size, stride=self.stride),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.lstm = nn.LSTMCell(128, self.hidden_size)
        self.critic_head = nn.Linear(self.hidden_size, 1)
        self.actor_head = nn.Linear(self.hidden_size, n_actions)

    def forward(self, x, hidden, grads=False):
        if grads:
            features = self.feature_extractor(x)
            hidden = self.lstm(features, hidden)

            value = self.critic_head(hidden[0])
            probs = F.softmax(self.actor_head(hidden[0]), dim=1)
            dist  = Categorical(probs)
        else:
            with torch.no_grad():
                features = self.feature_extractor(x)
                hidden = self.lstm(features, hidden)

                value = self.critic_head(hidden[0])
                probs = F.softmax(self.actor_head(hidden[0]), dim=1)
                dist  = Categorical(probs)

        return dist, value, hidden