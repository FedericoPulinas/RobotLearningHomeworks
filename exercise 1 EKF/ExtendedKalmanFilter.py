"""
    Implementation of the Extended Kalman Filter
    for an unactuated pendulum system
"""
import math

import numpy as np

class ExtendedKalmanFilter(object):


    def __init__(self, x0, P0, Q, R, dT):
        """
           Initialize EKF

            Parameters
            x0 - mean of initial state prior
            P0 - covariance of initial state prior
            Q  - covariance matrix of the process noise
            R  - covariance matrix of the measurement noise
            dT - discretization step for forward dynamics
        """
        self.x0=x0
        self.P0=P0
        self.Q=Q
        self.R=R
        self.dT=dT


        self.g = 9.81  # Gravitational constant
        self.l = 1  # Length of the pendulum

        self.currentTimeStep = 0


        self.priorMeans = []
        self.priorMeans.append(None)  # no prediction step for timestep=0
        self.posteriorMeans = []
        self.posteriorMeans.append(x0)

        self.priorCovariances=[]
        self.priorCovariances.append(None)  # no prediction step for timestep=0
        self.posteriorCovariances=[]
        self.posteriorCovariances.append(P0)



    def stateSpaceModel(self, x, t):
        dxdt = np.array([[x[1,0]], [-(self.g/self.l)*np.sin(x[0,0])]])

        return dxdt

    def discreteTimeDynamics(self, x_t):
        """
            Forward Euler integration.

            returns next state as x_t+1 = x_t + dT * (dx/dt)|_{x_t}
        """
        x_tp1 = x_t + self.dT*self.stateSpaceModel(x_t, None)
        return x_tp1


    def jacobianStateEquation(self, x_t):
        """
            Jacobian of discrete dynamics w.r.t. the state variables,
            evaluated at x_t

            Parameters:
                x_t : state variables (column-vector)
        """
        A = np.zeros(shape=(2,2))  # TODO: shape?

        # TODO DONE
        # compute the Jacobian of the discrete dynamics
        A = np.array([ [1, self.dT],
                         [-1 * self.dT * (self.g/self.l) * np.cos(x_t[1,0]), 1]] )
        #print("jacobian: ", J_f)

        return A


    def jacobianMeasurementEquation(self, x_t):
        """
            Jacobian of measurement model.

            Measurement model is linear, hence its Jacobian
            does not actually depend on x_t
        """
        C = np.zeros(shape=(1,2))

        # TODO DONE
        C = np.array([[1, 0]])
        #print("C", C)

        return C


    def forwardDynamics(self):
        self.currentTimeStep = self.currentTimeStep+1  # t-1 ---> t


        """
            Predict the new prior mean for timestep t
        """
        #x_t_prior_mean = self.discreteTimeDynamics(self.posteriorMeans[self.currentTimeStep-1])
        x_t_prior_mean = self.discreteTimeDynamics(self.posteriorMeans[self.currentTimeStep - 1])

        """
            Predict the new prior covariance for timestep t
        """
        # Linearization: jacobian of the dynamics at the current a posteriori estimate
        #A_t_minus = self.jacobianStateEquation(self.posteriorMeans[self.currentTimeStep-1])
        A_t_minus = self.jacobianStateEquation(self.posteriorMeans[self.currentTimeStep - 1])

        # TODO: propagate the covariance matrix forward in time DONE
        #x_t_prior_cov = A_t_minus @ self.posteriorCovariances[self.currentTimeStep - 1] @ A_t_minus.transpose() + self.Q
        x_t_prior_cov = A_t_minus @ self.posteriorCovariances[self.currentTimeStep - 1] @ A_t_minus.T + self.Q

        # Save values
        self.priorMeans.append(x_t_prior_mean)
        self.priorCovariances.append(x_t_prior_cov)


    def updateEstimate(self, z_t):
        """
            Compute Posterior Gaussian distribution,
            given the new measurement z_t
        """

        # Jacobian of measurement model at x_t
        #Ct = self.jacobianMeasurementEquation(self.priorMeans[self.currentTimeStep])
        Ct = self.jacobianMeasurementEquation(self.priorMeans[self.currentTimeStep])

        # TODO: Compute the Kalman gain matrix
        #K_t = self.priorCovariances[self.currentTimeStep] @ Ct.transpose() @ np.linalg.inv(Ct @ self.priorCovariances[self.currentTimeStep] @ Ct.transpose() + self.R)
        K_t = self.priorCovariances[self.currentTimeStep] @ Ct.T @ np.linalg.inv(Ct @ self.priorCovariances[self.currentTimeStep] @ Ct.T + self.R)

        # TODO: Compute posterior mean
        #x_t_mean = self.priorMeans[self.currentTimeStep] + K_t @ (z_t - Ct @ self.priorMeans[self.currentTimeStep])
        x_t_mean = self.priorMeans[self.currentTimeStep] + K_t @ (z_t - Ct @ self.priorMeans[self.currentTimeStep])

        # TODO: Compute posterior covariance
        #x_t_cov = (np.eye(2,2) - K_t @ Ct) @ self.priorCovariances[self.currentTimeStep]
        x_t_cov = (np.eye(2, 2) - K_t @ Ct) @ self.priorCovariances[self.currentTimeStep]

        # Save values
        self.posteriorMeans.append(x_t_mean)
        self.posteriorCovariances.append(x_t_cov)
