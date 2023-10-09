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

        # self.num_states = len(self.states)
        # self.num_actions = len(self.actions)
        self.Q = np.zeros([self.states, self.actions])


    def get_next_action_without_Q_table_update(self, state):
        self.s = state

        random_number = np.random.random()

        # exploitation
        if random_number > self.rar:
            action = np.argmax(self.Q[state])

        # exploration
        else:
            action = np.random.randint(0, self.actions - 1)

        self.a = action

        return action

    def get_next_action_with_Q_table_update(self, new_state, reward):

        last_state = self.s
        last_action = self.a

        self.update_Q(last_state, last_action, new_state, reward)

        self.s = new_state
        self.a = self.get_action(new_state)

        # if debug:
        #     print(f"last state: {last_state} --> last action: {last_action} --> new state: {new_state}, reward: {reward}")
        #
        # if self.verbose:
        #     print(f"s = {s_prime}, a = {self.a}, r={r}")
        return self.a

    def get_action(self, state):

        random_number = np.random.random()

        # exploitation
        if random_number > self.rar:
            action = np.argmax(self.Q[state])

        # exploration
        else:
            action = np.random.randint(0, self.actions-1)

        return action


    def decay_rar(self, epoch):
        # exponential decay of rar
        return self.min_rar + (self.max_rar - self.min_rar) * np.exp(-self.radr * epoch)


    def update_Q(self, last_state, last_action, new_state, reward):

        old_q_value = self.Q[last_state, last_action]
        next_optimal_q_value = np.max(self.Q[new_state, :])

        self.Q[last_state, last_action] = old_q_value + self.lr * (reward + self.dr * next_optimal_q_value - old_q_value)





