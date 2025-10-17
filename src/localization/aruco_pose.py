from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


_ARUCO_DICTS = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class ArucoPoseEstimator:
	"""Detect ArUco markers and estimate their pose.

	Requires camera intrinsics and marker side length. Returns list of detections
	with `id`, `rvec`, `tvec`, and `corners`.
	"""
	def __init__(
		self,
		dictionary: str,
		marker_length_m: float,
		camera_matrix: Optional[np.ndarray],
		dist_coeffs: Optional[np.ndarray],
	) -> None:
		if dictionary not in _ARUCO_DICTS:
			raise ValueError(f"Unknown aruco dictionary: {dictionary}")
		self.marker_length_m = float(marker_length_m)
		self.camera_matrix = camera_matrix
		self.dist_coeffs = dist_coeffs
		self._dict = cv2.aruco.getPredefinedDictionary(_ARUCO_DICTS[dictionary])
		self._params = cv2.aruco.DetectorParameters()
		self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)

	def detect_and_estimate(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
		"""Return list of detected marker poses in the camera frame using solvePnP.

		If intrinsics are missing, returns empty list.
		"""
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
		corners, ids, _ = self._detector.detectMarkers(gray)
		poses: List[Dict[str, Any]] = []
		if ids is None or len(ids) == 0:
			return poses
		if self.camera_matrix is None or self.dist_coeffs is None:
			return poses

		# Define 3D marker corner points in marker frame (Z=0), order matches ArUco corners
		L = float(self.marker_length_m)
		half = L / 2.0
		objp = np.array([
			[-half,  half, 0.0],  # top-left
			[ half,  half, 0.0],  # top-right
			[ half, -half, 0.0],  # bottom-right
			[-half, -half, 0.0],  # bottom-left
		], dtype=np.float32)

		for idx, marker_id in enumerate(ids.flatten()):
			imgp = corners[idx].reshape(-1, 2).astype(np.float32)  # 4x2
			ok, rvec, tvec = cv2.solvePnP(
				objp, imgp, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
			)
			if not ok:
				continue
			poses.append({
				"id": int(marker_id),
				"rvec": rvec.reshape(3),
				"tvec": tvec.reshape(3),
				"corners": corners[idx],
			})
		return poses

	def fuse_multiple_markers(self, poses: List[Dict[str, Any]], marker_world_positions: Dict[int, List[float]]) -> Optional[Dict[str, Any]]:
		"""Fuse multiple marker detections into a single camera pose estimate.
		
		Args:
			poses: List of detected marker poses
			marker_world_positions: Dict mapping marker_id to [x, y, z] world position
			
		Returns:
			Fused camera pose in world frame, or None if no valid markers
		"""
		if not poses or not marker_world_positions:
			return None
			
		valid_poses = []
		for pose in poses:
			marker_id = pose['id']
			if marker_id in marker_world_positions:
				# Get marker world position
				marker_world_pos = np.array(marker_world_positions[marker_id])
				
				# Convert marker pose to camera pose in world frame
				R_mc = R.from_rotvec(pose['rvec'])  # marker rotation in camera frame
				t_mc = pose['tvec']  # marker translation in camera frame
				
				# CORRECT TRANSFORMATION:
				# t_mc = marker position in camera frame (from ArUco)
				# Camera position in world frame = marker_world_pos - t_mc
				# This is the simplest and most direct transformation
				t_cam_in_world = marker_world_pos - t_mc
				
				# Camera orientation in world frame = inverse of marker rotation
				R_cam_in_world = R_mc.inv()
				
				# Weight by distance (closer = higher weight)
				distance = np.linalg.norm(t_mc)
				weight = 1.0 / (1.0 + distance)  # Inverse distance weighting
				
				valid_poses.append({
					'position': t_cam_in_world,
					'orientation': R_cam_in_world.as_quat(),
					'weight': weight,
					'marker_id': marker_id,
					'distance': distance
				})
		
		if not valid_poses:
			return None
			
		# Weighted average of positions and orientations
		total_weight = sum(p['weight'] for p in valid_poses)
		if total_weight == 0:
			return None
			
		# Weighted position average
		avg_position = sum(p['position'] * p['weight'] for p in valid_poses) / total_weight
		
		# Weighted quaternion average (simplified - could use proper quaternion averaging)
		avg_orientation = valid_poses[0]['orientation']  # Use first as base
		for pose in valid_poses[1:]:
			alpha = pose['weight'] / total_weight
			avg_orientation = self._quat_slerp(avg_orientation, pose['orientation'], alpha)
		
		return {
			'position': avg_position,
			'orientation_quat': avg_orientation,
			'num_markers': len(valid_poses),
			'marker_ids': [p['marker_id'] for p in valid_poses]
		}
	
	def _quat_slerp(self, q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
		"""Spherical linear interpolation between quaternions."""
		from scipy.spatial.transform import Slerp
		r0 = R.from_quat(q0)
		r1 = R.from_quat(q1)
		slerp = Slerp([0, 1], R.concatenate([r0, r1]))
		return slerp(alpha).as_quat()
