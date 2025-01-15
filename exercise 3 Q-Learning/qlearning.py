import gym
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
import random
import seaborn as sns
import pandas as pd

import sys

np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

# Whether to perform training or use the stored .npy file
MODE = "TEST" # TRAINING, TEST

episodes = 20000
test_episodes = 100
num_of_actions = 2  # 2 discrete actions for Cartpole

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Parameters
gamma = 0.98
alpha = 0.1
constant_eps = 0#.2
b = 2223  # TODO: choose b so that with GLIE we get an epsilon of 0.1 after 20'000 episodes
# epsilon_k = b / (b+k), k = numero episodio corrente (es: 100esimo episodio, k = 100)

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

# Initialize Q values
#q_grid = np.zeros((discr, discr, discr, discr, num_of_actions))  #x, xdot=v, th, thdot=av, action
q_grid = 50*np.ones((discr, discr, discr, discr, num_of_actions))  #x, xdot=v, th, thdot=av, action


if MODE == "TEST":
    q_grid = np.load('q_values.npy')

def find_nearest(array, value):
    return np.argmin(np.abs(array - value))

def get_cell_index(state):
    """Returns discrete state from continuous state"""
    x = find_nearest(x_grid, state[0])
    v = find_nearest(v_grid, state[1])
    th = find_nearest(th_grid, state[2])
    av = find_nearest(av_grid, state[3])
    return x, v, th, av


def get_action(state, q_values, greedy=False): #q_values = q_grid
    x, v, th, av = get_cell_index(state)

    if greedy: # TEST -> greedy policy
        best_action_estimated = np.argmax(q_values[x, v, th, av])  # TODO: greedy w.r.t. q_grid

        return best_action_estimated, 0, 1

    else: # TRAINING -> epsilon-greedy policy

        if np.random.rand() < epsilon:
            # Random action
            #print("random action")
            action_chosen = random.randint(0, 1)  # TODO: choose random action with equal probability among all actions

            return action_chosen, 1, 0
        else:
            # Greedy action
            #print("greedy action")
            best_action_estimated = np.argmax(q_values[x, v, th, av])  # TODO: greedy w.r.t. q_grid

            return best_action_estimated, 0, 1


        """
        #tasks 3.3 - assumed
        # greedy policy
        best_action_estimated = np.argmax(q_values[x, v, th, av])  # TODO: greedy w.r.t. q_grid

        return best_action_estimated, 0, 1
        """


def plot_heatmap():
    value_function = np.max(q_grid, axis=4)  # 16,16,16,16 - max on the number of actions
    averaged_values = np.mean(value_function, axis=(1, 3))  # Average over xdot and thetadot axes - 16, 16

    plt.figure()
    plt.imshow(averaged_values,
               extent=[x_min, x_max,th_min, th_max],  #transposed
               #extent=[th_min, th_max, x_min, x_max],  #not transposed
               #origin='lower',
               #cmap='viridis',
               aspect='auto')
    plt.colorbar()
    plt.title("averaged Value function heatmap")
    plt.xlabel("x - position")
    plt.ylabel("theta - angle")
    plt.grid()
    plt.show()

def update_q_value(old_state, action, new_state, reward, done, q_array):
    old_cell_index = get_cell_index(old_state)
    new_cell_index = get_cell_index(new_state)

    # Target value used for updating our current Q-function estimate at Q(old_state, action)
    if done is True:
        target_value = reward  # HINT: if the episode is finished, there is not next_state. Hence, the target value is simply the current reward.
    else:
        target_value = reward + gamma * np.max(q_array[new_cell_index[0], new_cell_index[1], new_cell_index[2], new_cell_index[3], :])  # TODO

    # Update Q value
    q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] = q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action] + alpha*(target_value - q_array[old_cell_index[0], old_cell_index[1], old_cell_index[2], old_cell_index[3], action])  # TODO

    return


# Training loop
ep_lengths, epl_avg = [], []
train_reward_history = []
average_reward_history = []
rnd_actions, grd_actions = [], []
random_a, greedy_a = 0, 0
for ep in range(episodes+test_episodes):
#for ep in range(episodes):

    #if ep == 0:
        #plot_heatmap()
    #if ep == 1:
        #plot_heatmap()
    if ep == episodes/2:
        plot_heatmap()

    test = ep > episodes

    train_reward = 0


    if MODE == "TEST":
        test = True

    state, done, steps = env.reset(), False, 0
    epsilon = constant_eps  # TODO: change to GLIE schedule (task 3.1) or 0 (task 3.3)
    #epsilon = b / (b + ep) #ep = k

    while not done:
        action, r_a, g_a = get_action(state, q_grid, greedy=test)

        new_state, reward, done, _ = env.step(action)
        if not test:
            update_q_value(state, action, new_state, reward, done, q_grid)
        else:
            env.render()
        state = new_state
        steps += 1
        train_reward += reward

        random_a += r_a
        greedy_a += g_a

    rnd_actions.append(random_a)
    grd_actions.append(greedy_a)
    random_a = 0
    greedy_a = 0

    train_reward_history.append(train_reward)
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))
        print('Epsilon:', epsilon)

    #insert a trend on the average every 100 episodes to visualise the progress of the datas
    if ep > 100:
        avg = np.mean(train_reward_history[-100:])
    else:
        avg = np.mean(train_reward_history)
    average_reward_history.append(avg)

print("Episode {}, average timesteps: {:.2f}".format(ep+1, np.mean(ep_lengths[max(0, ep-200):])))
print('Epsilon:', epsilon)


#PLOTS FOR TASK 3.1 
plt.figure(1)
plt.plot(np.arange(0, episodes), train_reward_history, label='Episodes Rewards')
plt.plot(np.arange(0, episodes), average_reward_history, label='Average Rewards every 100 episodes')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.plot()
plt.grid(True)
plt.legend()
"""
window_size = 100  # Number of episodes for averaging
avg_rnd_actions = np.convolve(rnd_actions, np.ones(window_size)/window_size, mode='same')
avg_grd_actions = np.convolve(grd_actions, np.ones(window_size)/window_size, mode='same')

print(np.shape(avg_grd_actions))

plt.figure(2)
plt.plot(np.arange(0, episodes), avg_rnd_actions, label='random actions taken')
plt.plot(np.arange(0, episodes), avg_grd_actions, label='greedy actions taken')
plt.xlabel('Episodes')
plt.ylabel('Action')
plt.grid(True)
plt.legend(["0 - random action", "1 - greedy action"])


plt.show()
"""

#TASK 3.2
#print(np.shape(q_grid)) #16,16,16,16,2
plot_heatmap()

if MODE == 'TEST':
    sys.exit()

# Save the Q-value array
#np.save("q_values.npy", q_grid)