# Localization Module (RealSense D435i + OpenCV ArUco + IMU Drift-Aware Fusion)

This project provides a localization pipeline using Intel RealSense D435i IMU + RGB, OpenCV ArUco for pose correction, and a drift-aware complementary fusion layer. Designed for a fixed camera orientation facing an ArUco board 20–80 cm away, with cross-shaped motion.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Calibrations Required
- Camera intrinsics (`camera_matrix`, `dist_coeffs`) from RealSense calibration or `calib*.json`.
- ArUco marker side length in meters (`marker_length_m`).
- IMU-to-camera extrinsics if using IMU orientation in camera frame.

## Run

```bash
python main.py
```

## Notes
- ArUco must be facing the camera; lighting should avoid glare.
- The fusion increases reliance on vision when drift is detected.
- For AprilTag support, swap the detector implementation or use apriltag library.
