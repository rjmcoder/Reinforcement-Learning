import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import pygame
import QLearner as ql

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


# # # of bins for cart_position, # of bins of cart_velocity, # of bins for pole angle, # of bins for pole angular velocity
# q_table_shape = (NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, env.action_space.n)
# q_table = np.zeros(q_table_shape)


####### Visualization ################
log_interval = 50
render_interval = 30000

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
fig.canvas.draw()
#####################################

def train(env):

    points_log = []
    mean_points_log = []
    epochs = []

    debug = True

    rewards = []
    log_interval = 1000

    # # of bins for cart_position, # of bins of cart_velocity, # of bins for pole angle, # of bins for pole angular velocity
    learner = ql.QLearner(states=(NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS),
                          actions=env.action_space.n)


    for epoch in range(EPOCHS):

        if debug: print(f"============= running episode: {epoch} of {EPOCHS} =================")

        initial_state = env.reset()[0]
        discretized_state = discretize_observation(initial_state, BINS)

        done = False
        points = 0
        epochs.append(epoch)

        action = learner.get_next_action_without_Q_table_update(discretized_state)

        while not done:

            # state, reward... env.step()
            new_state, reward, done, trunc, info = env.step(action)

            #discretize the state
            discretized_state = discretize_observation(new_state, BINS)

            # custom reward
            reward = fail(done, points, reward)

            # get next action
            action = learner.get_next_action_with_Q_table_update(discretized_state, reward)
            # print(action)

            # track rewards
            points += 1

        # if debug: print(learner.Q)

        # decay the random action rate
        learner.decay_rar(epoch)

        points_log.append(points)
        running_mean = round(np.mean(points_log[-30:]), 2)
        mean_points_log.append(running_mean)

        #####################################################
        if epoch % log_interval == 0:
            print(f"current mean rewards: {running_mean}")
            ax.clear()
            ax.scatter(epochs, points_log)
            ax.plot(epochs, points_log)
            ax.plot(epochs, mean_points_log, label=f"Running Mean: {running_mean}")
            plt.legend()
            fig.canvas.draw()
            plt.show()

    env.close()
    return learner.Q

# lets see how the agent is performing after training
def check_performance_after_training(q_table, env):

    total_reward = 0

    state = env.reset()[0]

    for steps in range(1000):
        env.render()
        discrete_state = discretize_observation(state, BINS)  # get bins
        action = np.argmax(q_table[discrete_state])  # and chose action from the Q-Table
        state, reward, done, trunc, info = env.step(action)  # Finally perform the action
        total_reward += 1
        if done:
            print(f"You got {total_reward} points!")
            break

    env.close()

# points_log = []
# mean_points_log = []
# epochs = []
#
# for epoch in range(EPOCHS):
#
#     print(f"\n======================= running epoch: {epoch} ======================")
#
#     initial_state = env.reset()[0]
#     discretized_state = discretize_observation(initial_state, BINS)
#     done = False
#     points = 0
#
#     epochs.append(epoch)
#
#     # play game
#     while not done:
#         action = epsilon_greedy_action_selection(epsilon, q_table, discretized_state)
#         # print(f"action: {action}")
#         next_state, reward, done, trunc, info = env.step(action)
#
#         reward = fail(done, points, reward)
#
#         next_state_discretized = discretize_observation(next_state, BINS)
#         old_q_value = q_table[discretized_state + (action,)]
#         next_optimal_q_value = np.max(q_table[next_state_discretized])
#
#         next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)
#         q_table[discretized_state + (action,)] = next_q
#
#         discretized_state = next_state_discretized
#         points += 1
#
#     print(f"\t points: {points}")
#     epsilon = reduce_epsilon(epsilon, epoch)
#     points_log.append(points)
#     running_mean = round(np.mean(points_log[-30:]), 2)
#     mean_points_log.append(running_mean)
#
#     #####################################################
#     if epoch % log_interval == 0:
#         ax.clear()
#         ax.scatter(epochs, points_log)
#         ax.plot(epochs, points_log)
#         ax.plot(epochs, mean_points_log, label=f"Running Mean: {running_mean}")
#         plt.legend()
#         fig.canvas.draw()
#         plt.show()

# env.close()

if __name__ == "__main__":
    # try_environment()

    # create the environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")  # ["human", "rgb_array"]
    env.reset()
    # help(env)
    # print(env.spec.max_episode_steps)

    NUM_BINS = 10
    BINS = create_bins(NUM_BINS)

    training = True
    testing = True

    if training == True:
        ALPHA = 0.8
        GAMMA = 0.9
        epsilon = 1
        BURN_IN = 1
        EPSILON_END = 10000
        EPSILON_REDUCE = 0.0001

        # run the training
        EPOCHS = 30000  # episodes, how many times the agents plays the game until it hits done
        q_table = train(env)
        np.save('q_table.npy', q_table)

    if testing == True:
        q_table = np.load('q_table.npy')
        env = gym.make("CartPole-v1", render_mode="human")
        # check the performance
        check_performance_after_training(q_table, env)













