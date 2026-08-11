from vision import camera_to_drone
import numpy as np

error_vector = np.array(camera_to_drone(2.0))
k_p_forward = 0.0 #todo tune these constants
k_p_yaw = 0.0
def SimpleFeedForward():
    forward_command = k_p_forward * error_vector[0]
    yaw_command = k_p_yaw * error_vector[-1]
    return forward_command, yaw_command