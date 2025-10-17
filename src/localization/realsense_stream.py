from __future__ import annotations

import threading
import time
from typing import Optional, Tuple, Dict, Any

try:
	import pyrealsense2 as rs
except Exception:
	rs = None  # type: ignore


class RealSenseStream:
	"""Thin wrapper over RealSense D435i pipeline to fetch color frames and accept OpenVINS odometry.

	- Color stream: BGR frames
	- OpenVINS odometry: position, orientation, timestamp
	- Threaded internal loop updates latest samples for non-blocking reads
	"""
	def __init__(
		self,
		serial: Optional[str] = None,
		width: int = 640,
		height: int = 480,
		fps: int = 30,
		enable_color: bool = True,
		enable_imu: bool = False,  # disabled by default, use OpenVINS odometry instead
	) -> None:
		self.serial = serial
		self.width = width
		height = int(height)
		self.height = height
		self.fps = fps
		self.enable_color = enable_color
		self.enable_imu = enable_imu

		self._pipeline: Optional["rs.pipeline"] = None
		self._cfg: Optional["rs.config"] = None
		self._active_profile: Optional["rs.pipeline_profile"] = None
		self._color_frame = None
		self._odometry: Optional[Dict[str, Any]] = None
		self._lock = threading.Lock()
		self._cv = threading.Condition(self._lock)
		self._running = False
		self._t0: Optional[float] = None  # normalize timestamps to start at 0
		self._frame_seq: int = 0  # monotonically increasing on new color frame

	def start(self) -> None:
		"""Start streaming from device with configured streams."""
		if rs is None:
			raise RuntimeError("pyrealsense2 not available")
		self._pipeline = rs.pipeline()
		self._cfg = rs.config()
		if self.serial:
			self._cfg.enable_device(self.serial)
		if self.enable_color:
			self._cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
		# IMU disabled - use OpenVINS odometry instead

		# Attempt to start with the requested configuration; fall back if device rejects rates
		try:
			self._active_profile = self._pipeline.start(self._cfg)
		except Exception as e:
			print(f"[RealSense] Primary start failed: {e}. Falling back to default IMU rates...")
			# Rebuild a minimal config with default IMU stream settings
			self._cfg = rs.config()
			if self.serial:
				self._cfg.enable_device(self.serial)
			if self.enable_color:
				self._cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
			if self.enable_imu:
				self._cfg.enable_stream(rs.stream.gyro)
				self._cfg.enable_stream(rs.stream.accel)
			try:
				self._active_profile = self._pipeline.start(self._cfg)
			except Exception as e2:
				print(f"[RealSense] Fallback start failed: {e2}. Starting color-only (no IMU)...")
				self._cfg = rs.config()
				if self.serial:
					self._cfg.enable_device(self.serial)
				if self.enable_color:
					self._cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
				self.enable_imu = False
				self._active_profile = self._pipeline.start(self._cfg)
		self._running = True

		def _loop() -> None:
			"""Background thread copying latest frames into member fields."""
			assert self._pipeline is not None
			while self._running:
				frames = self._pipeline.wait_for_frames(timeout_ms=2000)
				with self._lock:
					for f in frames:
						if f.is_frameset():
							continue
						# Color frames
						if self.enable_color and f.profile.stream_type() == rs.stream.color:
							self._color_frame = f.get_data()
							self._frame_seq += 1
							self._cv.notify_all()
						# IMU streams removed - OpenVINS provides odometry

		self._thread = threading.Thread(target=_loop, daemon=True)
		self._thread.start()

	def update_odometry(self, position: np.ndarray, orientation_quat: np.ndarray, timestamp: float) -> None:
		"""Update odometry from OpenVINS (position, orientation as quat, timestamp)."""
		with self._lock:
			if self._t0 is None:
				self._t0 = timestamp
			self._odometry = {
				"position": position.copy(),
				"orientation_quat": orientation_quat.copy(),
				"t": timestamp - self._t0,
			}

	def read(self) -> Tuple[Optional["np.ndarray"], Optional[Dict[str, Any]]]:
		"""Return latest color frame (BGR) and odometry dict."""
		with self._lock:
			color = None
			if self._color_frame is not None:
				# Lazy import to avoid hard dependency in environments without cv
				import numpy as np  # local import
				color = np.asanyarray(self._color_frame)
			odometry = self._odometry.copy() if self._odometry is not None else None
		return color, odometry

	def read_next(self, timeout: Optional[float] = None) -> Tuple[Optional["np.ndarray"], Optional[Dict[str, Any]]]:
		"""Block until a new color frame is available or timeout.

		Returns the latest (color_bgr, odometry). If timeout elapses, returns current latest values.
		"""
		end_time = None if timeout is None else (time.time() + timeout)
		with self._lock:
			start_seq = self._frame_seq
			# If we already have a frame, wait only if no new frame arrives
			while self._running and self._frame_seq == start_seq:
				remaining = None if end_time is None else max(0.0, end_time - time.time())
				if remaining is not None and remaining == 0.0:
					break
				self._cv.wait(timeout=remaining)
			# Prepare return values
			color = None
			if self._color_frame is not None:
				import numpy as np  # local import
				color = np.asanyarray(self._color_frame)
			odometry = self._odometry.copy() if self._odometry is not None else None
			return color, odometry

	def get_color_intrinsics(self) -> Tuple[Optional["np.ndarray"], Optional["np.ndarray"]]:
		"""Return (camera_matrix, dist_coeffs) from active color stream, if available."""
		if rs is None or self._active_profile is None:
			return None, None
		color_stream = self._active_profile.get_stream(rs.stream.color)
		try:
			vsp = color_stream.as_video_stream_profile()
			intr = vsp.get_intrinsics()
			# Lazy import to avoid top-level dependency
			import numpy as np  # local import
			K = np.array(
				[[intr.fx, 0.0, intr.ppx],
				 [0.0, intr.fy, intr.ppy],
				 [0.0, 0.0, 1.0]],
				dtype=float,
			)
			dist = np.array(list(intr.coeffs[:5]), dtype=float)  # k1,k2,p1,p2,k3
			return K, dist
		except Exception:
			return None, None

	def stop(self) -> None:
		"""Stop background thread and RealSense pipeline."""
		self._running = False
		if hasattr(self, "_thread"):
			self._thread.join(timeout=2.0)
		if self._pipeline is not None:
			self._pipeline.stop()
			self._pipeline = None
