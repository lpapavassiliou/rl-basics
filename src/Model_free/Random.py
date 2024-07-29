import numpy as np


class Random_Policy:
    """
    Random policy class for generating random actions within a specified range.

    Attributes:
        u_max (float): The maximum value of the action range.
        u_min (float): The minimum value of the action range.
    """

    def __init__(self, u_max: float, u_min: float) -> None:
        self.u_max = u_max
        self.u_min = u_min

    def get_action(self, x: np.ndarray) -> float:
        """
        Generate a random action within the specified range.

        Parameters:
            x (np.ndarray): The input state.

        Returns:
            float: The randomly generated action.
        """
        return self.u_min + (self.u_max - self.u_min) * np.random.random()
