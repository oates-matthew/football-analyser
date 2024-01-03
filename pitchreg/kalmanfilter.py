import numpy as np
import torch


class KF:

    def __init__(self):
        self.state = np.array([1, 0, 0, 0, 1, 0, 0, 0])
        self.P = np.eye(8)
        self.F = np.eye(8)  # Assuming a simple linear model for state transition
        self.F = np.eye(8)  # Assuming a simple linear model for state transition
        self.H = np.eye(8)  # Direct observation model
        self.Q = np.eye(8) * 0.01  # Process noise covariance, adjust as needed
        self.R = np.eye(8) * 0.5   # Measurement noise covariance, adjust as needed

    def process_homography(self, h_matrix):
        h_matrix = h_matrix.cpu().detach().numpy()
        # Extract parameters from the matrix
        measurement = np.array([h_matrix[0, 0], h_matrix[0, 1], h_matrix[0, 2],
                                h_matrix[1, 0], h_matrix[1, 1], h_matrix[1, 2],
                                h_matrix[2, 0], h_matrix[2, 1]])

        # Predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ (measurement - self.H @ self.state)
        self.P = self.P - K @ self.H @ self.P

        # Form updated homography matrix
        updated_h_matrix = np.array([[self.state[0], self.state[1], self.state[2]],
                                     [self.state[3], self.state[4], self.state[5]],
                                     [self.state[6], self.state[7], 1]])

        return torch.from_numpy(updated_h_matrix)