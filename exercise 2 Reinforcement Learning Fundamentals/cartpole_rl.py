"""
    Robot Learning
    Exercise 2

    Reinforcement Learning 

    Polito A-Y 2023-2024
"""
import math

import torch
import gym
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import sys
from agent import Agent, Policy
from utils import get_space_dim

import sys


# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", "-t", type=str, default=None,
                        help="Model to be tested")
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument("--train_episodes", type=int, default=500,
                        help="Number of episodes to train for")
    parser.add_argument("--render_training", action='store_true',
                        help="Render each frame during training. Will be slower.")
    parser.add_argument("--render_test", action='store_true', help="Render test")
    parser.add_argument("--central_point", type=float, default=0.0,
                        help="Point x0 to fluctuate around")
    parser.add_argument("--random_policy", action='store_true', help="Applying a random policy training")
    return parser.parse_args(args)


# Policy training function
def train(agent, env, train_episodes, early_stop=True, render=False,
          silent=False, train_run_id=0, x0=0, random_policy=False):
    x0 = [-2.0, 2.0]
    # Arrays to keep track of rewards
    reward_history, timestep_history = [], []
    average_reward_history = []
    point = 0
    cp = -1 #basically unassigned to not impose a specific side to start with
    m = -6.9
    passed = 1

    # Run actual training
    for episode_number in range(train_episodes):
        reward_sum, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state (it's a random initial state with small values)
        observation = env.reset()

        # Loop until the episode is over
        while not done:
            # Get action from the agent
            action, action_probabilities = agent.get_action(observation)

            if random_policy:
                # Task 1.1

                #Sample a random action from the action space

                action = random.randint(0, 1)
                print('action', action)

            previous_observation = observation
            ppoint = point

            # Perform the action on the environment, get new state and reward
            # note that after env._max_episode_steps the episode is over, if we stay alive that long
            observation, reward, done, info = env.step(action)

            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            # TODO
            reward, point, cp, m, passed = new_reward(observation, x0, point, cp, m, passed)
            #print('point: ', point)


            # Store action's outcome (so that the agent can improve its policy)
            agent.store_outcome(previous_observation, action_probabilities, action, reward)

            # Draw the frame, if desired
            if render:
                env.render()

            # Store total episode reward
            reward_sum += reward
            timesteps += 1

        point = 0
        cp = -1
        m = -6.9
        passed = 1
        if not silent:
            print("Episode {} finished. Total reward: {:.3g} ({} timesteps)"
                  .format(episode_number, reward_sum, timesteps))

        # Bookkeeping (mainly for generating plots)
        reward_history.append(reward_sum)
        timestep_history.append(timesteps)
        if episode_number > 100:
            avg = np.mean(reward_history[-100:])
        else:
            avg = np.mean(reward_history)
        average_reward_history.append(avg)

        # If we managed to stay alive for 15 full episodes, assume it's learned
        # (in the default setting)
        if early_stop and np.mean(timestep_history[-15:]) == env._max_episode_steps:
            if not silent:
                print("Looks like it's learned. Finishing up early")
            break

        # Let the agent do its magic (update the policy)
        agent.episode_finished(episode_number)

    # Store the data in a Pandas dataframe for easy visualization
    data = pd.DataFrame({"episode": np.arange(len(reward_history)),
                         "train_run_id": [train_run_id]*len(reward_history),
                         "reward": reward_history,
                         "mean_reward": average_reward_history})
    return data


# Function to test a trained policy
def test(agent, env, episodes, render=False, x0=0):
    x0 = [-2.0, 2.0]
    trajectories = np.zeros((500, 50))
    trj = 0
    tri = 0
    test_reward, test_len, tot_reward = 0, 0, 0
    test_reward_history = []
    avg_test_reward = []
    final_position = np.array([])
    avg_positions = []
    point = 0
    cp = -1
    m = -6.9
    passed = 1

    vel = 0

    print("episodes", episodes)
    #episodes = 100

    print('Num testing episodes:', episodes)

    for ep in range(episodes):
        done = False
        observation = env.reset()
        while not done:
        # Task 1.2
            """
            Test on 500 timesteps
            """
            # TODO

            action, _ = agent.get_action(observation, evaluation=True)  # Similar to the training loop above -
                                                                        # get the action, act on the environment, save total reward
                                                                        # (evaluation=True makes the agent always return what it thinks to be
                                                                        # the best action - there is no exploration at this point)
            previous_observation = observation
            observation, reward, done, info = env.step(action)

            if observation[1] > vel:
                vel = observation[1]
            # Task 3.1
            """
                Use a different reward, overwriting the original one
            """
            # TODO
            #print(point, x0[point])
            reward, point, cp, m, passed= new_reward(observation, x0, point, cp, m, passed)
            #print('point: ', point)

            #insert by column
            trajectories[tri][trj] = observation[0]
            tri += 1
            if render:
                env.render()
            test_reward += reward
            tot_reward += reward
            test_len += 1


        tri = 0
        trj += 1
        point = 0
        cp = -1
        passed = 1
        m = -6.9

        test_reward_history.append(tot_reward)
        avg_test_reward.append(np.mean(test_reward_history))

        if trj == 50:
            final_position = np.mean(trajectories, axis=1, keepdims=True)
            plt.plot(np.arange(0, 500), final_position, label='trajectory %d' %(ep/50))
            trj = 0
            trajectories.fill(0)

        #trajectory for single point
        #final_position.append(observation[0])
        #avg_positions.append(np.mean(final_position))

        #print("avg test reward:", test_reward/episodes, "episode length:", test_len, "episode:", ep, "test reward:", tot_reward, "last position:", observation[0])

        print('max velocity reached during episode ', ep, ' is ', vel, ' with ', tot_reward, ' tot reward')
        vel = 0

        tot_reward = 0
        test_len = 0
    #print("average stopping point: ", np.mean(final_position))

    #for oscillation between points

    #for single point
    #plt.plot(np.arange(0, episodes), final_position, label='position')
    #plt.plot(np.arange(0,episodes), avg_positions, label='avg position')
    #plt.title("Final position recorded during every episode")
    plt.axhline(2.4)
    plt.axhline(-2.4)
    plt.xlabel('timesteps')
    plt.ylabel('x coordinate')

    plt.legend()
    plt.grid(True)
    plt.show()

    data = pd.DataFrame({"test_reward": test_reward_history, "avg_test_reward": avg_test_reward})
    return data, (test_reward/episodes)

def new_reward(state, x0, point, cp, m, passed):
    # Task 3.1
    """
        Use a different reward, overwriting the original one
    """
    # TODO

    #PROBLEMA: il sistema impara ad andare nel senso opposto rispetto a dove si trova x0

    reward = 0.0
    curr_x = state[0]
    vel = state[1]
    checkpoints = [(x0[0]/2), (x0[1]/2), 0, x0[0], x0[1]]
    # 0 -> metaS, 1 -> metaD, 2 -> zero, 3 -> S, 4 -> D


    if cp == -1:
        if curr_x > 0.20:
            #print("----------------------------------------VADO A DESTRA")
            cp = 1
            point = 1
        elif curr_x < -0.20:
            #print("----------------------------------------VADO A SINISTRA")
            cp = 0
            point = 0

    goal = checkpoints[cp]
    distance = abs(checkpoints[cp] - curr_x)

    q = 10 #max distance between x0 and -x0 on the 2x line

    l2n = np.absolute((x0[point]) - (curr_x))

    #case single point

    #reward = -2 * l2n + 4.8

    #if (l2n <= 0.05):
    #    reward += 20

    if cp >= 0:
        reward = (m * distance + q)#*passed
        if distance > q/m:
            reward = reward /5.0
    else:
        reward = 0.5

    reward += abs(vel)/2.15

    if ( (l2n - (x0[1])/2) <= 0.5) and ((np.sign(x0[point])*np.sign(checkpoints[cp])) > 0):
        if (cp == 0):
            reward += 950*passed
            passed += 0.25
            cp = 3
            m = -6.9
            #print('||----------META>SINISTRA----------||----------||----------|| {:+.2f} m: {} vel: {:.2f}'.format(curr_x, m, vel))

        elif (cp == 1):
            reward += 950*passed
            passed += 0.25
            cp = 4
            m = -6.9
            #print('||----------||----------||----------META>DESTRA----------|| {:+.2f} m: {} vel: {:.2f}'.format(curr_x,m, vel))
    elif (abs(checkpoints[cp] - curr_x) <= 0.5) and ((np.sign(x0[point]) * np.sign(checkpoints[cp])) < 0):
        if (cp == 0) or (cp == 3):
            reward += 950*passed
            passed += 0.25
            cp = 2
            m = -6.9
            #print('||----------SINISTRA>META----------||----------||----------|| {:+.2f} m: {} vel: {:.2f}'.format(curr_x, m, vel))
        elif (cp == 1) or (cp == 4):
            reward += 950*passed
            passed += 0.25
            cp = 2
            m = -6.9
            #print('||----------||----------||----------DESTRA>META----------||   {:+.2f} m: {} vel: {:.2f}'.format(curr_x, m, vel))

    #center after changing direction
    if (cp == 2) and (abs(curr_x) <= 0.5):
        passed += 0.25
        if np.sign(x0[point]) > 0:
            reward += 900*passed
            cp = 1
            m = -6.6
            #print('||----------||----------CENTRO>DESTRA----------||----------|| {:+.2f} m: {} vel: {:.2f}'.format(curr_x, m, vel))
        elif np.sign(x0[point]) < 0:
            reward += 900*passed
            cp = 0
            m = -6.6
            #print('||----------||----------CENTRO>SINISTRA----------||----------|| {:+.2f} m: {} vel: {:.2f}'.format(curr_x, m, vel))

    if (np.sign(curr_x)*np.sign(x0[point]) < 0) and (passed < 1.5): #when it suddenly changes direction
        if (cp == 0) and (abs(x0[1] - curr_x) <= 0.6): #-x
            #print('SINISTRA~~~~~~~~~~~~~~~~~~~~CAMBIO LATO~~~~~~~~~~~~~~~~~~~~ vel: {:.2f}'.format(vel))
            #reward += 500
            passed += 0.25
            cp = 4
            point = 1
        elif (cp == 1) and (abs(x0[0] - curr_x) <= 0.6): #+x
            #print('~~~~~~~~~~~~~~~~~~~~CAMBIO LATO~~~~~~~~~~~~~~~~~~~~DESTRA vel: {:.2f}'.format(vel))
            #reward += 500
            passed += 0.25
            cp = 3
            point = 0

    if (l2n <= 0.3):
        if point == 0:
            reward += 1050*passed
            passed += 0.25
            point = 1
            cp = 0
            m = -19
            #print('CAMBIO----------||----------||----------||----------||      {:+.2f} m: {} vel: {:.2f}'.format(curr_x,m, vel))

        elif point == 1:
            reward += 1050*passed
            passed += 0.25
            point = 0
            cp = 1
            m = -19
            #print('||----------||----------||----------||----------CAMBIO      {:+.2f} m: {} vel: {:.2f}'.format(curr_x,m, vel))

    #print('reward towards: ', x0)

    if (curr_x < (x0[0] - 0.20)) or (curr_x > (x0[1] + 0.20)):
        reward -= 175

    return reward, point, cp, m, passed

# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    env = gym.make(args.env)

    # Task 1.2
    """
    # For CartPole-v0 - change the maximum episode length
    """
    # TODO
    env._max_episode_steps = 500

    # Get dimensionalities of actions and observationsl
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Instantiate agent and its policy
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy)

    # Print some stuff
    print("Environment:", args.env)
    print("Training device:", agent.train_device)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)

    #point to point motion
    args.central_point = [-2.0, 2.0]

    # If no model was passed, train a policy from scratch.
    # Otherwise load the policy from the file and go directly to testing.
    if args.test is None:
        # Train
        training_history = train(agent, env, args.train_episodes, False, args.render_training, x0=args.central_point, random_policy=args.random_policy)

        # Save the model
        model_file = "%s_params.ai" % args.env
        torch.save(policy.state_dict(), model_file)
        print("Model saved to", model_file)

        # Plot rewards
        sns.lineplot(x="episode", y="reward", data=training_history, color='blue', label='Reward')
        sns.lineplot(x="episode", y="mean_reward", data=training_history, color='orange', label='100-episode average')
        plt.legend()
        plt.title("Reward history (%s)" % args.env)
        plt.grid(True, linestyle=':')
        plt.show()
        print("Training finished.")
    else:
        # Test
        print("Loading model from", args.test, "...")
        state_dict = torch.load(args.test)
        policy.load_state_dict(state_dict)
        print("Testing...")
        test_history, avg = test(agent, env, args.train_episodes, args.render_test, x0=0)#x0=args.central_point)

        print("average test reward: ", avg)

        sns.lineplot(x=np.arange(args.train_episodes), y="test_reward", data=test_history, label='test reward')
        sns.lineplot(x=np.arange(args.train_episodes), y="avg_test_reward", data=test_history, label='avg test reward')
        plt.legend()
        plt.title("Test results on 500 timesteps")
        plt.grid(True, linestyle=':')
        plt.show()

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

