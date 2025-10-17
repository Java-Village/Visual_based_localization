from __future__ import annotations

from typing import Optional, Dict, Any

import numpy as np


class ImuDriftDetector:
	"""Detect gyro bias drift using a sliding window average of angular velocity.

	If the mean angular rate magnitude exceeds a threshold, we flag drift. This is
	simple and robust for stationary-or-slow motions; parameters can be tuned.
	"""
	def __init__(self, gyro_window_s: float = 2.0, gyro_bias_threshold: float = 0.02) -> None:
		self.gyro_window_s = float(gyro_window_s)
		self.gyro_bias_threshold = float(gyro_bias_threshold)
		self._times: list[float] = []
		self._gyros: list[np.ndarray] = []

	def update(self, gyro_xyz: Optional[tuple[float, float, float]], t: Optional[float]) -> Dict[str, Any]:
		"""Update detector with latest gyro sample (rad/s) at time t (s).

		Returns dict with `drifting` and `bias_norm`.
		"""
		if gyro_xyz is None or t is None:
			return {"drifting": False, "bias_norm": 0.0}
		g = np.asarray(gyro_xyz, dtype=float)
		self._times.append(t)
		self._gyros.append(g)
		# Drop old samples beyond the window
		while self._times and (t - self._times[0]) > self.gyro_window_s:
			self._times.pop(0)
			self._gyros.pop(0)
		if len(self._gyros) < 5:
			return {"drifting": False, "bias_norm": 0.0}
		bias = np.mean(self._gyros, axis=0)
		bias_norm = float(np.linalg.norm(bias))
		return {"drifting": bias_norm > self.gyro_bias_threshold, "bias_norm": bias_norm}
