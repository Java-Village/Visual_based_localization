from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np
from scipy.spatial.transform import Rotation as R


def _rvec_to_quat(rvec: np.ndarray) -> np.ndarray:
	"""Convert Rodrigues rotation vector to quaternion (x, y, z, w)."""
	rot = R.from_rotvec(rvec)
	return rot.as_quat()  # x, y, z, w


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
	"""Spherical linear interpolation between two quaternions.

	Handles the antipodal case and normalizes output.
	"""
	q0 = q0 / np.linalg.norm(q0)
	q1 = q1 / np.linalg.norm(q1)
	dot = float(np.dot(q0, q1))
	if dot < 0.0:
		q1 = -q1
		dot = -dot
	DOT_THRESHOLD = 0.9995
	if dot > DOT_THRESHOLD:
		# Linear interpolation for very close quaternions
		result = q0 + alpha * (q1 - q0)
		return result / np.linalg.norm(result)
		theta_0 = np.arccos(dot)
	theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
	sin_theta_0 = np.sin(theta_0)
	if sin_theta_0 < 1e-6:
		return q0
	theta = theta_0 * alpha
	sin_theta = np.sin(theta)
	s0 = np.sin(theta_0 - theta) / sin_theta_0
	s1 = sin_theta / sin_theta_0
	return (s0 * q0) + (s1 * q1)


class DriftAwareFusion:
	"""Fusion of OpenVINS odometry and ArUco vision with drift-aware weights.

	- Uses OpenVINS odometry for prediction (position, orientation)
	- Corrects position/orientation using vision, boosting weights when drift detected
	- Designed as a placeholder fusion that can be replaced by EKF/UKF later
	"""
	def __init__(
		self,
		odometry_weight: float = 0.7,
		vision_position_weight: float = 0.6,
		vision_orientation_weight: float = 0.4,
		drift_vision_boost: float = 0.25,
		vision_position_frame: str = "marker_in_camera",  # or "camera_in_marker"
	) -> None:
		self.odometry_weight = odometry_weight
		self.vision_position_weight = vision_position_weight
		self.vision_orientation_weight = vision_orientation_weight
		self.drift_vision_boost = drift_vision_boost
		self.vision_position_frame = vision_position_frame

		self._position = np.zeros(3)
		self._orientation = R.identity().as_quat()
		self._last_t: Optional[float] = None

	def update(
		self,
		odometry: Optional[Dict[str, Any]],
		vision: Optional[Dict[str, Any]],
		drift: Optional[Dict[str, Any]] = None,
	) -> Dict[str, Any]:
		"""Update fusion with latest OpenVINS odometry and optional vision correction.

		`odometry` expects keys: `position` (xyz), `orientation_quat` (xyzw), `t` (seconds).
		`vision` expects keys: `rvec`, `tvec` in the camera/world frame.
		"""
		# Predict with OpenVINS odometry
		if odometry is not None and odometry.get("t") is not None:
			odom_pos = odometry.get("position")
			odom_quat = odometry.get("orientation_quat")
			
			if odom_pos is not None:
				# Blend odometry position with current estimate
				w = self.odometry_weight
				self._position = (1 - w) * self._position + w * np.asarray(odom_pos)
			
			if odom_quat is not None:
				# Blend odometry orientation with current estimate
				w = self.odometry_weight
				self._orientation = _quat_slerp(self._orientation, np.asarray(odom_quat), w)

		# Correct with vision
		corr_pos = None
		corr_quat = None
		
		if vision is not None:
			vision_tvec = np.asarray(vision.get("tvec", [np.nan, np.nan, np.nan]))
			vision_rvec = np.asarray(vision.get("rvec", [np.nan, np.nan, np.nan]))

			# Choose position/orientation frame
			corr_pos = vision_tvec
			if not np.isnan(vision_rvec).any():
				corr_quat = _rvec_to_quat(vision_rvec)

			if self.vision_position_frame == "camera_in_marker":
				# Invert pose: camera pose in marker frame
				if not (np.isnan(vision_tvec).any() or np.isnan(vision_rvec).any()):
					R_mc = R.from_rotvec(vision_rvec)  # marker in camera
					R_cm = R_mc.as_matrix().T          # camera in marker
					t_mc = vision_tvec.reshape(3, 1)
					p_cam_in_marker = (-R_cm @ t_mc).reshape(3)
					corr_pos = p_cam_in_marker
					# Orientation of camera in marker frame is inverse rotation
					corr_quat = R_mc.inv().as_quat()

		# Apply position correction if available
		if corr_pos is not None and not np.isnan(corr_pos).any():
			w = self.vision_position_weight
			if drift and drift.get("drifting"):
				w = min(0.99, w + self.drift_vision_boost)
			self._position = (1 - w) * self._position + w * corr_pos

		# Apply orientation correction if available
		if corr_quat is not None:
			wq = self.vision_orientation_weight
			if drift and drift.get("drifting"):
				wq = min(0.99, wq + self.drift_vision_boost)
			self._orientation = _quat_slerp(self._orientation, corr_quat, wq)

		return {
			"position": self._position.copy(),
			"orientation_quat": self._orientation.copy(),
		}
