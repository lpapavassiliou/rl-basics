import numpy as np
from physics_functions import pendolum_kinematics, angle_norm2
from parameters import ModelParams, LearningParams
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Environment:

    def __init__(self, theta_init=0, theta_dot_init=0):
        self.theta = theta_init
        self.theta_dot = theta_dot_init
        self.mparams = ModelParams()
        self.lparams = LearningParams()
        self.done = False
        self.reference_state = np.zeros(2)
        self.steps_taken = 0

    def reset(self):
        self.theta = np.random.random() * np.pi/2
        self.theta_dot = 0
        self.steps_taken = 0
        return self.get_state()
    
    def dynamics(self, torque):
        # theta_dot_dot = (- self.mparams.m * self.mparams.g * np.sin(self.theta) - torque - self.mparams.d * self.theta_dot)/rod_inertia(self.mparams.m, self.mparams.L)
        gravity = - self.mparams.g / self.mparams.L * np.sin(self.theta)
        damping = - self.mparams.d * self.theta_dot
        theta_dot_dot = gravity + torque + damping
        return theta_dot_dot
    
    def step(self, torque):
        reward = self.reward(torque)
        k1 = self.dynamics(torque)
        k2 = self.dynamics(self.mparams.dt * k1/2 + torque)
        k3 = self.dynamics(self.mparams.dt * k2/2 + torque)
        k4 = self.dynamics(self.mparams.dt * k3 + torque)
        self.theta = self.theta + self.mparams.dt * self.theta_dot
        self.theta_dot = self.theta_dot + self.mparams.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        next_state = self.get_state()
        self.steps_taken += 1
        done = self.steps_taken > 2000
        return next_state, reward, done
    
    def get_state(self):
        return np.array([self.theta, self.theta_dot])
    
    def set_reference(self, reference_state):
        reference_state = np.array(reference_state)
        self.reference_state = reference_state
    
    def visualize(self, action=None):
        plt.clf()  # Clear the previous frame
        plt.axis('equal')  # Maintain equal scaling for x and y axe

        # plot reference pendolum
        x,y = pendolum_kinematics(self.mparams.L, self.reference_state[0])
        plt.plot([0, x], [0, y], marker='o', linestyle='--', markersize=4, color='grey')

        # plot actual pendolum
        x,y = pendolum_kinematics(self.mparams.L, self.theta)
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
        plt.pause(0.001)  # Pause to allow the plot to be displayed
    
    def reward(self, action_to_take):
        state = self.get_state()
        angle_error = angle_norm2(state[0], self.reference_state[0])
        velocity_error = (state[1] - self.reference_state[1])**2
        action_penalty =  (action_to_take)**2
        return  - angle_error - velocity_error - self.lparams.action_weight*action_penalty
