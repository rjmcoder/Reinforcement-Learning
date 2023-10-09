import numpy as np
import matplotlib.pyplot as plt
import os
import gym
import time
import QLearner as ql
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


def train(env):

    debug = False

    rewards = []
    log_interval = 1000

    learner = ql.QLearner(states=env.observation_space.n,
                          actions=env.action_space.n)


    for episode in range(EPOCHS):

        if debug: print(f"============= running episode: {episode} of {EPOCHS} =================")

        state = env.reset()[0]
        done = False
        total_rewards = 0
        action = learner.get_next_action_without_Q_table_update(state)

        while not done:

            # state, reward... env.step()
            new_state, reward, done, trunc, info = env.step(action)

            # get next action
            action = learner.get_next_action_with_Q_table_update(new_state, reward)

            # track rewards
            total_rewards += reward

        # if debug: print(learner.Q)

        # agent finsihed a round of the game
        episode += 1

        # decay the random action rate
        learner.decay_rar(episode)

        rewards.append(total_rewards)

        if episode % log_interval == 0:
            print(np.sum(rewards))

    env.close()
    return learner.Q


# lets see how the agent is performing after training
def check_performance_after_training(q_table, env):
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
                   max_episode_steps=1000)   # max actions to take before stopping the game
    # help(env)
    # print(env.spec.max_episode_steps)

    # create a default environment
    # env = gym.make('FrozenLake-v1', desc=None, map_name='8x8', is_slippery=False, render_mode='ansi')

    env.reset()

    # run the training
    EPOCHS = 10000  # episodes, how many times the agents plays the game until it hits done
    q_table = train(env)

    env.reset()
    # check the performance
    check_performance_after_training(q_table, env)