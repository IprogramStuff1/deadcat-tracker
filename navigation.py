import numpy as np
k_p_forward = 0.0 #todo tune these constants
k_p_yaw = 0.0
def SimpleProportionalControl(error_vector):
    forward_command = np.clip(k_p_forward * error_vector[0], -0.5, 0.5)
    yaw_command = np.clip(k_p_yaw * error_vector[-1], -0.5, 0.5)
    return forward_command, yaw_command