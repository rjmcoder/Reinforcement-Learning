import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import pygame

env = gym.make("CartPole-v1", render_mode="human")
env.reset()

# for step in range(50):
#     env.render()
#     action = env.action_space.sample()
#     observation, reward, done, trunc, info = env.step(action)
#     # print(observation)  # (cart position, cart velocity, pole angle, pole angular velocity)
#     time.sleep(0.1)
#
# env.close()


def create_bins(num_bins_per_obs=10):
    bins_cart_position = np.linspace(-4.8, 4.8, num_bins_per_obs)
    bins_cart_velocity = np.linspace(-5, 5, num_bins_per_obs)
    bins_pole_angle = np.linspace(-0.418, 0.418, num_bins_per_obs)
    bins_pole_angular_velocity = np.linspace(-5, 5, num_bins_per_obs)

    bins = np.array([bins_cart_position, bins_cart_velocity, bins_pole_angle, bins_pole_angular_velocity])

    return bins


NUM_BINS = 10
BINS = create_bins(NUM_BINS)
EPOCHS = 20000
ALPHA = 0.8
GAMMA = 0.9
epsilon = 1
BURN_IN = 1
EPSILON_END = 10000
EPSILON_REDUCE = 0.0001


def reduce_epsilon(epsilon, epoch):
    if BURN_IN <= epoch <= EPSILON_END:
        epsilon -= EPSILON_REDUCE

    return epsilon


# this function is really flexible where you can adjust how agent is rewarded/punished
def fail(done, points, reward):
    if done and points < 150:
        reward = -200

    return reward


def discretize_observation(observations, bins):

    binned_observations = []

    for i, observation in enumerate(observations):
        discretized_observation = np.digitize(observation, bins[i])  # cool function to find which bin a value lies in given an array of bins

        binned_observations.append(discretized_observation)

    return tuple(binned_observations)


# # of bins for cart_position, # of bins of cart_velocity, # of bins for pole angle, # of bins for pole angular velocity
q_table_shape = (NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, env.action_space.n)
q_table = np.zeros(q_table_shape)


def epsilon_greedy_action_selection(epsilon, q_table, discrete_state):
    """
    Returns an action for the agent. It uses a random exploration vs exploitation trade-off
    """
    random_number = np.random.random()

    # EXPLOITATION (choose the action that maximizes Q)
    if random_number > epsilon:

        action = np.argmax(q_table[discrete_state])

    # EXPLORATION (choose a random action)
    else:
        action = np.random.randint(0, env.action_space.n)

    return action


def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
    return old_q_value + ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)


####### Visualization ################
log_interval = 500
render_interval = 30000

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
fig.canvas.draw()
#####################################


points_log = []
mean_points_log = []
epochs = []

for epoch in range(EPOCHS):

    print(f"\n======================= running epoch: {epoch} ======================")

    initial_state = env.reset()[0]
    discretized_state = discretize_observation(initial_state, BINS)
    done = False
    points = 0

    epochs.append(epoch)

    # play game
    while not done:
        action = epsilon_greedy_action_selection(epsilon, q_table, discretized_state)
        # print(f"action: {action}")
        next_state, reward, done, trunc, info = env.step(action)

        reward = fail(done, points, reward)

        next_state_discretized = discretize_observation(next_state, BINS)
        old_q_value = q_table[discretized_state + (action,)]
        next_optimal_q_value = np.max(q_table[next_state_discretized])

        next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)
        q_table[discretized_state + (action,)] = next_q

        discretized_state = next_state_discretized
        points += 1

    print(f"\t points: {points}")
    epsilon = reduce_epsilon(epsilon, epoch)
    points_log.append(points)
    running_mean = round(np.mean(points_log[-30:]), 2)
    mean_points_log.append(running_mean)

    #####################################################
    if epoch % log_interval == 0:
        ax.clear()
        ax.scatter(epochs, points_log)
        ax.plot(epochs, points_log)
        ax.plot(epochs, mean_points_log, label=f"Running Mean: {running_mean}")
        plt.legend()
        fig.canvas.draw()

env.close()

print("============== done! ====================")













