import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from .physics_functions import pendolum_kinematics, angle_norm2, standardize_angle
from .parameters import ModelParams, LearningParams

class Environment:

    def __init__(self, theta_init=0, theta_dot_init=0, x_ref=np.zeros(2)):
        self.x = np.array([theta_init, theta_dot_init])
        self.mparams = ModelParams()
        self.lparams = LearningParams()
        self.steps_taken = 0
        self.x_ref = x_ref

    def encode_state(self, x):
        if len(x.shape) == 1:
            x_encoded = np.insert(x, 0, 0)
            x_encoded[0] = np.cos(x[0])
            x_encoded[1] = np.sin(x[1])
        else:
            x_encoded = np.insert(x, 0, 0, axis=1)
            x_encoded[:, 0] = np.cos(x[:, 0])
            x_encoded[:, 1] = np.sin(x[:, 1])
        return x_encoded
    
    def decode_state(self, x_encoded):
        if len(x_encoded.shape) == 1:
            x = x_encoded[1:]
            x[0] = np.arctan2(x_encoded[1], x_encoded[0])
        else:
            x = x_encoded[:, 1:]
            x[:, 0] = np.arctan2(x_encoded[:, 1], x_encoded[:, 0])
        return x

    def reset(self):
        self.x = np.array([np.random.random() * np.pi/2, 0])
        self.steps_taken = 0
        return self.x, self.get_encoded_state()
    
    def dynamics(self, x, t, u):
        f = np.zeros_like(x)
        f[0] = x[1]
        gravity = - self.mparams.g / self.mparams.L * np.sin(x[0])
        damping = 0. #- self.mparams.d * x[1]
        f[1] = gravity + damping + u
        return f
    
    def compute_next_state(self, x, u):
        return np.array(odeint(self.dynamics, 
                        x, 
                        np.linspace(0, self.mparams.dt, 2), 
                        args=(u, ))[-1, :])
    
    def compute_next_encoded_state(self, x_encoded, u):
        return self.encode_state(self.compute_next_state(self.decode_state(x_encoded), u))
    
    def reward(self, x, u):
        angle_error = angle_norm2(x[0], self.x_ref[0])
        velocity_error = (x[1] - self.x_ref[1])**2
        action_penalty = u**2
        return  - angle_error - 0.1*velocity_error - 0.001*action_penalty
    
    def step(self, u):
        reward = self.reward(self.x, u)
        self.x = self.compute_next_state(self.x, u)
        self.x[0] = standardize_angle(self.x[0])
        x_encoded_next = self.encode_state(self.x)
        self.steps_taken += 1
        done = self.steps_taken > self.lparams.max_steps
        return self.x, x_encoded_next, reward, done
    
    def set_reference(self, x_ref):
        self.x_ref = np.array(x_ref)
    
    def get_encoded_state(self):
        return self.encode_state(self.x)
    
    def get_encoded_reference(self):
        return self.encode_state(self.x_ref)
    
    def visualize(self, action=None):
        plt.clf()  # Clear the previous frame
        plt.axis('equal')  # Maintain equal scaling for x and y axe

        # plot reference pendolum
        x, y = pendolum_kinematics(self.mparams.L, self.x_ref[0])
        plt.plot([0, x], [0, y], marker='o', linestyle='--', markersize=4, color='grey')

        # plot actual pendolum
        x, y = pendolum_kinematics(self.mparams.L, self.x[0])
        plt.plot([0, x], [0, y], marker='o', markersize=8, color='b')
        plt.plot([0, 0], marker='o', markersize=12, color='black')

        # Set plot limits and show the plot
        plt.xlim(-2*self.mparams.L, 2*self.mparams.L)
        plt.ylim(-2*self.mparams.L, self.mparams.L)
        plt.title('Pendulum Simulation')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        if action is not None:
            plt.bar(x=0.3, height=0.25, width=0.051, y=-0.2-0.125, color='grey')
            plt.bar(x=0.3, height=0.005*action, width=0.05, y=-0.2, color='red')
            plt.bar(x=0.3, height=0.001, width=0.05, y=-0.2, color='black')
        plt.pause(1/60.)  # Pause to allow the plot to be displayed
