import numpy as np
k_p_forward = 0.3 #todo tune these constants
k_p_yaw = 0.3
def SimpleProportionalControl(error_vector):
    forward_command = np.clip(k_p_forward * error_vector[0], -0.3, 0.3)
    yaw_command = np.clip(k_p_yaw * error_vector[-1], -0.3, 0.3)
    return forward_command, yaw_command