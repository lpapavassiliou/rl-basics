import numpy as np

def pendolum_kinematics(L, theta):
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    return x, y

def standardize_angle(angle):
    return np.mod(angle, 2*np.pi)

def angle_norm2(angle1, angle2):
    angle1 = standardize_angle(angle1)
    angle2 = standardize_angle(angle2)
    diff = np.abs(angle1 - angle2)
    # Wrap around to the shortest distance
    diff = np.minimum(diff, 2*np.pi - diff)
    return diff**2