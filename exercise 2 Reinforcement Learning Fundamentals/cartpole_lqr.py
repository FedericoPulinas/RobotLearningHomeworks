"""
    Robot Learning
    Exercise 2

    Linear Quadratic Regulator

    Polito A-Y 2023-2024
"""
import gym
import numpy as np
from scipy import linalg     # get riccati solver
import argparse
import matplotlib.pyplot as plt
import sys
from utils import get_space_dim, set_seed
import pdb 
import time

# Parse script arguments
def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0",
                        help="Environment to use")
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--time_sleep", action='store_true',
                        help="Add timer for visualizing rendering with a slower frame rate")
    parser.add_argument("--mode", type=str, default="multiple_R",
                        help="Type of test ['control', 'multiple_R']")
    return parser.parse_args(args)

def linearized_cartpole_system(mp, mk, lp, g=9.81):
    mt=mp+mk
    a = g/(lp*(4.0/3 - mp/(mp+mk)))
    # state matrix
    A = np.array([[0, 1, 0, 0],
                [0, 0, a, 0],
                [0, 0, 0, 1],
                [0, 0, a, 0]])

    # input matrix
    b = -1/(lp*(4.0/3 - mp/(mp+mk)))
    B = np.array([[0], [1/mt], [0], [b]])
    return A, B

def optimal_controller(A, B, R_value):
    R = R_value*np.eye(1, dtype=int)  # choose R (weight for input)
    Q = 5*np.eye(4, dtype=int)        # choose Q (weight for state)
   # solve ricatti equation
    P = linalg.solve_continuous_are(A, B, Q, R)

    # calculate optimal controller gain
    K = np.dot(np.linalg.inv(R),
            np.dot(B.T, P))
    return K

def apply_state_controller(K, x):
    # feedback controller
    u = -np.dot(K, x)   # u = -Kx
    if u > 0:
        return 1, u     # if force_dem > 0 -> move cart right
    else:
        return 0, u     # if force_dem <= 0 -> move cart left

def multiple_R(env, mp, mk, l, g, time_sleep=False, terminate=True):
    """
    Vary the value of R within the range [0.01, 0.1, 10, 100] and plot the forces 
    """
    forces = np.array([])
    #TODO:
    Rs = [0.01, 0.1, 10, 100]
    t = 0

    for R in Rs:
        force, t = control(env, mp, mk, l, g, R, False, time_sleep, terminate)
        if force[0] < 0:
            force *= -1
        plt.plot(np.arange(t), force, alpha = 0.5, linewidth=2.0, label="R: %.2f" %R)

    #print(np.shape(forces))
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.show()


def control(env, mp, mk, l, g, R, singleR, time_sleep=False, terminate=True):
    observations = np.array([])
    forces = np.array([])
    """
    Control using LQR
    """
    #TODO: plot the states of the system ...

    obs = env.reset()    # Reset the environment for a new episode
    
    A, B = linearized_cartpole_system(mp, mk, l, g)
    K = optimal_controller(A, B, R)    # Re-compute the optimal controller for the current R value

    for i in range(1000):

        env.render()
        if time_sleep:
            time.sleep(.1)
        
        # get force direction (action) and force value (force)
        action, force = apply_state_controller(K, obs)
        forces = np.append(forces, force)

        # absolute value, since 'action' determines the sign, F_min = -10N, F_max = 10N
        abs_force = abs(float(np.clip(force, -10, 10)))
        
        # change magnitute of the applied force in CartPole
        env.env.force_mag = abs_force

        # apply action
        obs, reward, done, _ = env.step(action)
        observations = np.append(observations, obs)

        if terminate and done:
            print(f'Terminated after {i+1} iterations.')
            observations = np.reshape(observations, (int(len(observations)/4), 4))
            #print(np.shape(observations))
            #print(np.shape(forces))
            if singleR:
                plt.figure(1)
                plt.plot(np.arange(0, i+1, 1), observations[:, 0], label='position')
                plt.plot(np.arange(0, i+1, 1), observations[:, 1], label='velocity')
                plt.plot(np.arange(0, i+1, 1), observations[:, 2], label='angle')
                plt.plot(np.arange(0, i+1, 1), observations[:, 3], label='angular velocity')
                plt.grid(True, linestyle=':')
                plt.legend(loc='upper right')
                plt.show()
            break
    return forces, i+1

# The main function
def main(args):
    # Create a Gym environment with the argument CartPole-v0 (already embedded in)
    env = gym.make(args.env)

    env._max_episode_steps = 400

    # Get dimensionalities of actions and observations
    action_space_dim = get_space_dim(env.action_space)
    observation_space_dim = get_space_dim(env.observation_space)

    # Print some stuff
    print("Environment:", args.env)
    print("Observation space dimensions:", observation_space_dim)
    print("Action space dimensions:", action_space_dim)
    print("Seed:", args.seed)

    set_seed(args.seed)    # seed for reproducibility
    env.env.seed(args.seed)
    
    mp, mk, l, g = env.masspole, env.masscart, env.length, env.gravity

    if args.mode == "control":
        control(env, mp, mk, l, g, 1, True, args.time_sleep, terminate=True)
    elif args.mode == "multiple_R":
        multiple_R(env, mp, mk, l, g, args.time_sleep, terminate=True)

    env.close()

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)

