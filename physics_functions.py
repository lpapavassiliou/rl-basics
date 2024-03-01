import numpy as np

def rod_inertia(m, L):
    return 1/12 * m * L**2

def pendolum_kinematics(L, theta):
    x = L * np.sin(theta)
    y = -L * np.cos(theta)
    return x, y

def standardize_angle(angle):
    while angle < -np.pi:
        angle += 2*np.pi
    while angle > np.pi:
        angle -= 2*np.pi
    return angle

def angle_norm2(angle1, angle2):
    angle1 = standardize_angle(angle1)
    angle2 = standardize_angle(angle2)
    return (angle1 - angle2)**2