from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Activation, Input, Flatten, Conv2D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

class Agent(object):
    def __init__(self, input_dims, parameters, n_actions=3):
        self.gamma = parameters['GAMMA']
        self.alpha = parameters['LEARNING_RATE']
        self.input_dims = input_dims
        self.fc1_dims = parameters['HIDDEN_SIZE1']
        self.fc2_dims = parameters['HIDDEN_SIZE2']
        self.n_actions = n_actions
        self.ppo_epsilon = parameters['PPO_EPSILON']
        self.entropy_beta = parameters['ENTROPY_BETA']
        self.minibatch_size = parameters['MINI_BATCH_SIZE']
        self.ppo_epochs = parameters['PPO_EPOCHS']
        self.critic_discount = parameters['CRITIC_DISCOUNT']

        self.actor, self.critic, self.policy = self.build_actor_critic_network()
        self.action_space = [i for i in range(n_actions)]

    def build_actor_critic_network(self):
        input_frame = Input(shape=(self.input_dims))
        old_log_prob = Input(shape=(self.n_actions,))
        advantage = Input(shape=(1,))

        flattened_input = Flatten()(input_frame)
        dense1 = Dense(self.fc1_dims, activation='relu')(flattened_input)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)
        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            new_policy_probs = y_pred
            ratio = K.exp(K.log(new_policy_probs + 1e-8) - old_log_prob)
            
            p1 = ratio*advantage
            p2 = K.clip(ratio, min_value=1-self.ppo_epsilon, max_value=1+self.ppo_epsilon)*advantage

            actor_loss = -K.mean(K.minimum(p1, p2))
            total_loss = actor_loss - self.entropy_beta * K.mean(-new_policy_probs*K.log(new_policy_probs + 1e-8))
            return total_loss

        actor = Model(inputs=[input_frame, old_log_prob, advantage], outputs=[probs])

        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        critic = Model(inputs=[input_frame], outputs=[values])

        critic.compile(optimizer=Adam(lr=self.alpha), loss='mean_squared_error')

        policy = Model(inputs=[input_frame], outputs=[probs])

        return actor, critic, policy

    def choose_action(self, observation):
        probabilities = self.policy.predict(observation)
        _, n = probabilities.shape
        action = np.array([np.random.choice(n, p=row) for row in probabilities])
        log_probs = np.log(probabilities + 1e-8)

        return action, log_probs
    
    def get_state_value(self, observation):
        value = self.critic.predict(observation)
        return value[:, 0]

    def learn(self, states, log_probs, advantages, actions, returns):
        actions_one_hot = K.one_hot(actions, self.n_actions)
        self.actor.fit([states, log_probs, advantages], [actions_one_hot], epochs=self.ppo_epochs, verbose=True, shuffle=True, batch_size=self.minibatch_size)
        self.critic.fit([states], [returns], epochs=self.ppo_epochs, verbose=True, shuffle=True, batch_size=self.minibatch_size)