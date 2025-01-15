"""
    Robot Learning
    Exercise 1

    Extended Kalman Filter

    Polito A-Y 2023-2024
"""
import pdb

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.stats import norm

from ExtendedKalmanFilter import ExtendedKalmanFilter

# Discretization time step (frequency of measurements)
deltaTime=0.01

# Initial true state
x0 = np.array([np.pi/3, 0.5])

# Simulation duration in timesteps
simulationSteps=400
totalSimulationTimeVector=np.arange(0, simulationSteps*deltaTime, deltaTime)

# System dynamics (continuous, non-linear) in state-space representation (https://en.wikipedia.org/wiki/State-space_representation)
def stateSpaceModel(x,t):
    """
        Dynamics may be described as a system of first-order
        differential equations:
        dx(t)/dt = f(t, x(t))
    """
    g=9.81
    l=1
    dxdt=np.array([x[1], -(g/l)*np.sin(x[0])])
    return dxdt

# True solution x(t)
x_t_true = odeint(stateSpaceModel, x0, totalSimulationTimeVector)



"""
    EKF initialization
"""
# Initial state belief distribution (EKF assumes Gaussian distributions)
x_0_mean = np.zeros(shape=(2,1))  # column-vector
x_0_mean[0] = x0[0] + 3*np.random.randn()
x_0_mean[1] = x0[1] + 3*np.random.randn()
x_0_cov = 10*np.eye(2,2)  # initial value of the covariance matrix

# Process noise covariance matrix (close to zero, we do not want to model noisy dynamics)
Q=0.00001*np.eye(2,2)

# Measurement noise covariance matrix for EKF

Rs = np.array([[0.00001], [0.05], [5], [100], [500], [50000]])
print(np.shape(Rs))
i = 1
# create the extended Kalman filter object
for r in Rs:
    EKF = ExtendedKalmanFilter(x_0_mean, x_0_cov, Q, r, deltaTime)



    """
    Simulate process
    """
    measurement_noise_var = 0.05  # Actual measurement noise variance (uknown to the user)

    for t in range(simulationSteps-1):
        # PREDICT step
        EKF.forwardDynamics()

        # Measurement model
        z_t = x_t_true[t,0] + np.sqrt(measurement_noise_var)*np.random.randn()

        # UPDATE step
        EKF.updateEstimate(z_t)



    """
    Plot the true vs. estimated state variables
    """

    ### Estimates
    # EKF.posteriorMeans
    # EKF.posteriorCovariances

    ekf_posteriorMeans = np.reshape(EKF.posteriorMeans, (400, 2))
    ekf_postCov = np.array(EKF.posteriorCovariances)

    """
    to understand the covariance matrix results i can  compare it with the estimation error between true and estimated and compare it in order to get the accuracy
    """

    plt.figure(i)

    plt.subplot(211)
    plt.title("True state x1 vs Estimated state x1, R = %f" %r)
    plt.plot(totalSimulationTimeVector, x_t_true[:, 0], linestyle='dashed',  label="x_1_true", color='orange')
    plt.plot(totalSimulationTimeVector, ekf_posteriorMeans[:, 0], label="x_1_est", color='#7bd34e', linewidth=1)
    plt.xlabel(r'Total Simulation time [$\it{s}$]')
    plt.ylabel(r'Values of $\theta$ [$x_1$]')
    plt.grid(True, linestyle=':')
    plt.legend(loc='upper right')

    plt.subplot(212)
    plt.title("True state x2 vs Estimated state x2, R = %f" %r)
    plt.plot(totalSimulationTimeVector, x_t_true[:, 1], linestyle='dashed', label="x_2_true", color='orange')
    plt.plot(totalSimulationTimeVector, ekf_posteriorMeans[:, 1], label="x_2_est", color='#7bd34e', linewidth = 1)
    plt.xlabel(r'Total Simulation time [$\it{s}$]')
    plt.ylabel(r'Values of $\dot{\theta}$ [$x_2$]')
    plt.grid(True, linestyle=':')
    plt.legend(loc='upper right')


    plt.figure(i+np.size(Rs))
    plt.title("Covariance Matrix wrt time, R= %f" %r)
    #plt.plot(totalSimulationTimeVector, ekf_postCov[:, 0, 0], label="covariance 1,1")
    #plt.plot(totalSimulationTimeVector, ekf_postCov[:, 0, 1], label="covariance 1,2")
    #plt.plot(totalSimulationTimeVector, ekf_postCov[:, 1, 1], label="covariance 2,2")
    plt.plot(np.arange(-100, 100, deltaTime), norm.pdf(np.arange(-100, 100, deltaTime), ekf_posteriorMeans[0, 0], ekf_postCov[0,0,0]), label=r'initial distribution $x_1$-$x_1$', color='#C2FF91')
    plt.plot(np.arange(-100, 100, deltaTime), norm.pdf(np.arange(-100, 100, deltaTime), ekf_posteriorMeans[ np.shape(ekf_posteriorMeans)[0] - 1, 0], ekf_postCov[ np.shape(ekf_postCov)[0] - 1, 0, 0]), label=r'final distribution $x_1$-$x_1$', color='#00A000')

    plt.plot(np.arange(-100, 100, deltaTime), norm.pdf(np.arange(-100, 100, deltaTime), ekf_posteriorMeans[0, 1], ekf_postCov[0,1,1]), label=r'initial distribution $x_2$-$x_2$', color='#D2BFFF')
    plt.plot(np.arange(-100, 100, deltaTime), norm.pdf(np.arange(-100, 100, deltaTime), ekf_posteriorMeans[ np.shape(ekf_posteriorMeans)[0] - 1, 1], ekf_postCov[ np.shape(ekf_postCov)[0] - 1, 1, 1]), label=r'final distribution $x_2$-$x_2$', color='#4000A2')

    plt.xlabel(r'Possible range of $\sigma$')
    plt.ylabel(r'Values of mean $\mu$')
    plt.grid(True, linestyle=':')
    plt.legend(loc='upper right')

    i += 1
plt.show()
