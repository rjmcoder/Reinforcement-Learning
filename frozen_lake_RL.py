import numpy as np
import matplotlib.pyplot as plt
import os
import gym
import time
from gym.envs.toy_text.frozen_lake import generate_random_map


def try_environment():
    for step in range(15):
        print(env.render())
        action = env.action_space.sample()
        observation, reward, done, trunc, info = env.step(action)
        time.sleep(0.2)
        os.system('cls')
        if done:
            env.reset()

    env.close()


# Hyper parameters

EPOCHS = 20000 #episodes, how many times the agents plays the game until it hits done
ALPHA = 0.8 # LEARNING RATE
GAMMA = 0.95 # DISCOUNT RATE
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001


def reset_q_table(env):
    action_size = env.action_space.n
    state_size = env.observation_space.n

    # rows: states, columns: actions
    q_table = np.zeros([state_size, action_size])

    return q_table


def epsilon_greedy_action_selection(epsilon, q_table, discrete_state):
    random_number = np.random.random()

    # EXPLOITATION (choose the action that maximizes Q)
    if random_number > epsilon:

        state_row = q_table[discrete_state, :]
        action = np.argmax(state_row)

    # EXPLORATION (choose a random action)
    else:
        action = env.action_space.sample()

    return action

def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
    return old_q_value + ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)

def train(env, epsilon):

    debug = False

    rewards = []
    log_interval = 1000

    # reset q_table
    q_table = reset_q_table(env)

    for episode in range(EPOCHS):

        if debug: print(f"============= running episode: {episode} of {EPOCHS} =================")

        state = env.reset()[0]
        done = False
        total_rewards = 0

        while not done:

            # action
            action = epsilon_greedy_action_selection(epsilon, q_table, state)

            # state, reward... env.stepp()
            new_state, reward, done, trunc, info = env.step(action)

            # OLD == CURRENT Q VALUE
            old_q_value = q_table[state, action]

            # get next optimal Q value
            next_optimal_q_value = np.max(q_table[new_state, :])

            # compute the next Q value
            next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)

            # update the table
            q_table[state, action] = next_q

            # track rewards
            total_rewards += reward

            # new state is now the state
            state = new_state

        if debug: print(q_table)


        # agent finsihed a round of the game
        episode += 1

        epsilon = reduce_epsilon(epsilon, episode)

        rewards.append(total_rewards)

        if episode % log_interval == 0:
            print(np.sum(rewards))

    env.close()
    return q_table


# exponential decay of epsilon
def reduce_epsilon(epsilon, epoch):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * epoch)


# lets see how the agent is performing after training
def check_performance_after_training(env):
    state = env.reset()[0]

    for steps in range(100):
        print(env.render())
        action = np.argmax(q_table[state, :])
        state, reward, done, trunc, info = env.step(action)
        time.sleep(0.5)
        os.system('cls')

        if done:
            break

    env.close()

if __name__ == "__main__":
    # try_environment()

    # create a random environment
    env = gym.make('FrozenLake-v1',
                   desc=generate_random_map(size=5),
                   is_slippery=False,
                   render_mode='ansi',
                   max_episode_steps=1000)
    # help(env)
    # print(env.spec.max_episode_steps)

    # create a default environment
    # env = gym.make('FrozenLake-v1', desc=None, map_name='8x8', is_slippery=False, render_mode='ansi')

    env.reset()

    # run the training
    q_table = train(env, epsilon)

    # check the performance
    check_performance_after_training(env)