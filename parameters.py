class ModelParams:
    dt = 0.005 # sampling time
    m = 2 # mass
    L = 0.2 # pendulum lenght
    g = 9.81 # gravtity constant
    d = 0.5 # damping
    process_noise_std = 0.05

class LearningParams:
    action_weight = 0.2