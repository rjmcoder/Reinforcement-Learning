import numpy as np
import random


class QLearner:

    def __init__(self,
                 states=10,
                 actions=10,
                 alpha=0.8,         # learning rate - how much of the error to carry forward
                 gamma=0.9,         # discount rate for future rewards
                 rar=1.0,           # random action --exploration (high) vs exploitation (low) rate
                 min_rar = 0.01,
                 max_rar = 1.0,
                 radr=0.001): # random action decay rate

        self.states = states
        self.actions = actions
        self.lr = alpha
        self.dr = gamma
        self.rar = rar
        self.min_rar = min_rar
        self.max_rar = max_rar
        self.radr = radr

        self.num_states = len(self.states)
        self.num_actions = len(self.actions)
        self.Q = np.zeros([self.num_states, self.num_actions])


    def query_initial_state(self, state):
        self.s = state

        random_number = np.random.random()

        # exploitation
        if random_number > self.rar:
            a = np.argmax(self.Q[state])

        # exploration
        else:
            a = np.random.randint(0, self.num_actions - 1)

        action = self.actions[a]
        self.a = action

        return action


    def get_next_action(self, state):

        random_number = np.random.random()

        # exploitation
        if random_number > self.rar:
            a = np.argmax(self.Q[state])

        # exploration
        else:
            a = np.random.randint(0, self.num_actions-1)

        action = self.actions[a]

        return action


    def decay_rar(self, epoch):
        return self.min_rar + (self.max_rar - self.min_rar) * np.exp(-radr * epoch)


    def update_Q(self, last_state, last_action, new_state, reward):

        old_q_value = self.Q[last_state, last_action]
        next_optimal_q_value = np.max(self.Q[new_state, :])

        return old_q_value + self.lr * (reward + self.dr * next_optimal_q_value - old_q_value)


    def query(self, new_state, last_action_reward):
        pass




