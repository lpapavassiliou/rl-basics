class ModelParams:
    dt = 1/60. # sampling time
    m = 2 # mass
    L = 0.2 # pendulum lenght
    g = 9.81 # gravtity constant
    d = 0.5 # damping
    process_noise_std = 0.05
    u_min = -25
    u_max = 25

class LearningParams:
    max_steps = 500
    action_weight = 0.2