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
        self.conv3_filters = int(parameters['CONV3_FILTERS'])
        self.conv_filter_size = int(parameters['CONV_FILTER_SIZE'])
        self.stride = int(parameters['CONV_STRIDE'])
        self.hidden_size = int(parameters['HIDDEN_SIZE'])


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=self.conv1_filters, kernel_size=self.conv_filter_size, stride=self.stride),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv1_filters, out_channels=self.conv2_filters, kernel_size=self.conv_filter_size, stride=self.stride),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv2_filters, out_channels=self.conv3_filters, kernel_size=self.conv_filter_size, stride=self.stride),
            nn.ReLU()
        )

        self.volume_to_linear = nn.Flatten()
        self.feature_extractor_head = nn.Linear(1376, self.hidden_size)

        self.critic_head = nn.Linear(self.hidden_size, 1)
        self.actor_head = nn.Linear(self.hidden_size, n_actions)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv1_flattened = self.volume_to_linear(conv1)
        conv2_flattened = self.volume_to_linear(conv2)
        conv3_flattened = self.volume_to_linear(conv3)


        features = self.feature_extractor_head(torch.cat((conv1_flattened, conv2_flattened, conv3_flattened), dim=1))
        
        value = self.critic_head(features)
        probs = F.softmax(self.actor_head(features), dim=1)
        dist  = Categorical(probs=probs)

        return dist, value