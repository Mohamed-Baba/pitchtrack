"""Constant-velocity Kalman filter on boxes.

Measurement z = (cx, cy, w, h) in pixels; state x = (cx, cy, w, h, vcx, vcy, vw, vh), the
velocities in pixels per SECOND (time steps come from real timestamps, never from frame
counts). Motion model: piecewise-constant random acceleration on each coordinate.

Every standard deviation is proportional to the current box height h: a player of
height h pixels is filmed at s = h / 1.8 pixels per metre, so his accelerations, his
velocities and the localisation errors of the detector all scale like h. Parameters are
therefore given in box heights: q_pos = 4.0 means an acceleration standard deviation of
4 h pixels / s^2 for the centre.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHI2_4_95 = 9.4877   # 95 % quantile of the chi-square distribution with 4 degrees of freedom

H_MAT = np.hstack([np.eye(4), np.zeros((4, 4))])   # measurement matrix: z = H x


@dataclass
class KalmanBoxFilter:
    q_pos: float = 4.0     # acceleration std of the centre, box heights / s^2
    q_size: float = 1.0    # acceleration std of w and h, box heights / s^2
    r_pos: float = 0.05    # measurement std of the centre, box heights (scripts/estimate_noise.py)
    r_size: float = 0.05   # measurement std of w and h, box heights (scripts/estimate_noise.py)
    v0: float = 2.0        # initial velocity std, box heights / s

    def _stds(self, h: float, pos: float, size: float) -> np.ndarray:
        h = max(float(h), 1.0)
        return np.array([pos * h, pos * h, size * h, size * h])

    def measurement_cov(self, h: float) -> np.ndarray:
        """R (4 x 4): diagonal, standard deviations proportional to h."""
        return np.diag(self._stds(h, self.r_pos, self.r_size) ** 2)

    def initiate(self, z) -> tuple[np.ndarray, np.ndarray]:
        """Mean and covariance of a new track from its first box z = (cx, cy, w, h)."""
        z = np.asarray(z, dtype=np.float64).reshape(4)
        mean = np.concatenate([z, np.zeros(4)])
        cov = np.diag(np.concatenate([self._stds(z[3], self.r_pos, self.r_size) ** 2,
                                      self._stds(z[3], self.v0, self.v0 / 2) ** 2]))
        return mean, cov

    # ----------------------------------------------------------------------- prediction
    def motion(self, dt: float, h: float) -> tuple[np.ndarray, np.ndarray]:
        """Transition F(dt) (8 x 8) and process noise Q(dt) (8 x 8) of the model.

        Per coordinate, with acceleration std sigma: F = [[1, dt], [0, 1]] and
        Q = sigma^2 [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]. The state is ordered
        (4 positions, 4 velocities); sigma = (q_pos h, q_pos h, q_size h, q_size h).
        """
        # Identity and zero matrices of size 4 x 4
        I4 = np.eye(4)
        Z4 = np.zeros((4, 4))

        # Transition matrix
        F = np.block([
            [I4,       dt * I4],
            [Z4,       I4]
        ])

        # Standard deviations of the random accelerations
        sigma = np.array([
            self.q_pos * h,
            self.q_pos * h,
            self.q_size * h,
            self.q_size * h
        ])

        # Squared standard deviations
        sigma2 = sigma ** 2

        # Three diagonal blocks of the process noise matrix
        Q_pp = np.diag(sigma2 * dt**4 / 4)
        Q_pv = np.diag(sigma2 * dt**3 / 2)
        Q_vv = np.diag(sigma2 * dt**2)

        # Complete process noise matrix
        Q = np.block([
            [Q_pp, Q_pv],
            [Q_pv, Q_vv]
        ])

        return F, Q

    def predict(self, mean, cov, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Prediction step: x <- F x, P <- F P F^T + Q (Q computed with the current height)."""
        # Get transition matrix and process noise

        # mean[3] corresponds to the current height h of the bounding box
        F, Q = self.motion(dt, mean[3])

        # Predict the state mean
        mean_pred = F @ mean

        # Predict the state covariance
        cov_pred = F @ cov @ F.T + Q

        return mean_pred, cov_pred

    # ----------------------------------------------------------------------- correction
    def update(self, mean, cov, z) -> tuple[np.ndarray, np.ndarray]:
        """Correction with the measured box z = (cx, cy, w, h).

        S = H P H^T + R, K = P H^T S^{-1}, x <- x + K (z - H x), and the Joseph form
        P <- (I - K H) P (I - K H)^T + K R K^T, which stays symmetric positive.
        """

        # Ensure the measurement has shape (4,)
        z = np.asarray(z, dtype=np.float64).reshape(4)

        # Measurement matrix: extracts the first four components of the state
        H = np.hstack([
            np.eye(4),
            np.zeros((4, 4))
        ])

        # Measurement noise covariance
        R = self.measurement_cov(mean[3])

        # Innovation covariance
        S = H @ cov @ H.T + R

        # Compute the Kalman gain using a linear system
        # K.T = solve(S, H @ P), because S is symmetric
        K_T = np.linalg.solve(S, H @ cov)
        K = K_T.T

        # Innovation: difference between the observed and predicted measurements
        innovation = z - H @ mean

        # Correct the state mean
        mean_new = mean + K @ innovation

        # Correct the covariance using the Joseph form
        I8 = np.eye(8)
        A = I8 - K @ H

        cov_new = A @ cov @ A.T + K @ R @ K.T

        return mean_new, cov_new

    # ----------------------------------------------------------------------- gating
    def gating_distance(self, mean, cov, zs) -> np.ndarray:
        """Squared Mahalanobis distances (N,) between the predicted measurement and N boxes zs (N, 4).

        d^2 = (z - H x)^T S^{-1} (z - H x) with S = H P H^T + R. Under the model, d^2 follows a
        chi-square law with 4 degrees of freedom: CHI2_4_95 is its 95 % quantile.
        """
        
        # Convert detections to a float64 array of shape (N, 4)
        zs = np.asarray(zs, dtype=np.float64).reshape(-1, 4)

        # Handle the case where there are no detections
        if len(zs) == 0:
            return np.empty(0, dtype=np.float64)

        # Measurement matrix: extracts (cx, cy, w, h)
        H = np.hstack([
            np.eye(4),
            np.zeros((4, 4))
        ])

        # Measurement noise covariance
        R = self.measurement_cov(mean[3])

        # Innovation covariance
        S = H @ cov @ H.T + R

        # Cholesky decomposition: S = L @ L.T
        L = np.linalg.cholesky(S)

        # Difference between each detection and the predicted measurement
        innovations = zs - H @ mean

        # Solve L @ Y = innovations.T
        Y = np.linalg.solve(L, innovations.T)

        # Squared Mahalanobis distances
        distances = np.sum(Y ** 2, axis=0)

        return distances


def box_from_mean(mean) -> np.ndarray:
    """xyxy box (4,) of a state mean."""
    cx, cy, w, h = mean[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
