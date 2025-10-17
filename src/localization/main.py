#!/usr/bin/env python3
"""
Test script that generates OpenVINS-style odometry data and feeds it to the localization system.
Creates a cross pattern motion: x: -1 to 1, y: -1 to 1, z: -1 to 1
"""

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from realsense_stream import RealSenseStream
from aruco_pose import ArucoPoseEstimator
from drift_detector import ImuDriftDetector
from fusion import DriftAwareFusion
import yaml
import cv2


def generate_cross_motion_odometry(duration_sec=20.0, fps=30):
    """Generate odometry data for cross pattern motion."""
    dt = 1.0 / fps
    t = 0.0
    positions = []
    orientations = []
    timestamps = []
    
    # Cross pattern: x: -1 to 1, y: -1 to 1, z: -1 to 1
    # Each axis takes duration_sec/3 seconds
    axis_duration = duration_sec / 3.0
    
    while t < duration_sec:
        # X axis motion: -1 to 1
        if t < axis_duration:
            progress = t / axis_duration
            x = -1.0 + 2.0 * progress
            y = 0.0
            z = 0.0
        # Y axis motion: -1 to 1  
        elif t < 2 * axis_duration:
            progress = (t - axis_duration) / axis_duration
            x = 1.0
            y = -1.0 + 2.0 * progress
            z = 0.0
        # Z axis motion: -1 to 1
        else:
            progress = (t - 2 * axis_duration) / axis_duration
            x = 1.0
            y = 1.0
            z = -1.0 + 2.0 * progress
        
        # Add some noise to make it realistic
        noise_scale = 0.01
        x += np.random.normal(0, noise_scale)
        y += np.random.normal(0, noise_scale)
        z += np.random.normal(0, noise_scale)
        
        # Simple orientation: slight rotation around Z axis
        yaw = 0.1 * np.sin(2 * np.pi * t / 10.0)  # slow rotation
        orientation = R.from_euler('z', yaw).as_quat()  # x, y, z, w
        
        positions.append([x, y, z])
        orientations.append(orientation)
        timestamps.append(t)
        
        t += dt
    
    return positions, orientations, timestamps


def main():
    """Test the localization system with synthetic OpenVINS odometry."""
    print("Generating cross pattern odometry data...")
    positions, orientations, timestamps = generate_cross_motion_odometry(duration_sec=20.0, fps=30)
    
    print(f"Generated {len(positions)} odometry samples")
    print("Motion pattern: x: -1→1, y: -1→1, z: -1→1")
    print("Starting localization test...")
    
    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    # Initialize components
    stream = RealSenseStream(
        serial=cfg.get("camera", {}).get("serial"),
        width=cfg.get("camera", {}).get("width", 640),
        height=cfg.get("camera", {}).get("height", 480),
        fps=cfg.get("camera", {}).get("fps", 30),
        enable_color=cfg.get("camera", {}).get("enable_color", True),
        enable_imu=False,  # We're providing odometry instead
    )
    stream.start()
    
    # Vision components
    vision_cfg = cfg.get("vision", {})
    camera_matrix = None
    dist_coeffs = None
    if vision_cfg.get("camera_matrix") is not None:
        camera_matrix = np.array(vision_cfg["camera_matrix"], dtype=float)
    if vision_cfg.get("dist_coeffs") is not None:
        dist_coeffs = np.array(vision_cfg["dist_coeffs"], dtype=float)
    
    estimator = ArucoPoseEstimator(
        dictionary=vision_cfg.get("aruco_dict", "DICT_4X4_1000"),
        marker_length_m=float(vision_cfg.get("marker_length_m", 0.08)),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )
    
    # Load marker world positions for multi-marker fusion
    marker_positions = vision_cfg.get("marker_positions", {})
    print(f"[DEBUG] Marker positions: {marker_positions}")
    
    # Fusion
    fusion_cfg = cfg.get("fusion", {})
    fusion = DriftAwareFusion(**fusion_cfg)
    
    # Runtime settings
    runtime_cfg = cfg.get("runtime", {})
    visualize = bool(runtime_cfg.get("visualize", True))
    log_poses = bool(runtime_cfg.get("log_poses", True))
    use_vision = bool(runtime_cfg.get("use_vision", True))
    use_synthetic_odometry = False  # Disable synthetic odometry for vision-only test
    
    print(f"[DEBUG] Vision enabled: {use_vision}")
    print(f"[DEBUG] ArUco dict: {vision_cfg.get('aruco_dict')}")
    print(f"[DEBUG] Marker size: {vision_cfg.get('marker_length_m')}m")
    print(f"[DEBUG] Camera matrix available: {camera_matrix is not None}")
    print(f"[DEBUG] Dist coeffs available: {dist_coeffs is not None}")
    if camera_matrix is not None:
        print(f"[DEBUG] Camera matrix:\n{camera_matrix}")
    if dist_coeffs is not None:
        print(f"[DEBUG] Dist coeffs: {dist_coeffs}")
    
    try:
        odom_idx = 0
        start_time = time.time()
        last_debug_time = 0.0
        last_position_time = 0.0
        
        while True:  # Run indefinitely for vision-only test
            # Get current odometry (only if synthetic odometry is enabled)
            if use_synthetic_odometry and odom_idx < len(positions):
                pos = np.array(positions[odom_idx])
                ori = np.array(orientations[odom_idx])
                ts = timestamps[odom_idx]
                
                # Feed odometry to stream
                stream.update_odometry(pos, ori, ts)
                odom_idx += 1
            
            # Read from stream (block until new frame to avoid busy waiting)
            frame, odometry = stream.read_next(timeout=1.0)
            
            # Vision processing with multi-marker fusion
            vision_pose = None
            current_time = time.time()
            if frame is not None and use_vision:
                poses = estimator.detect_and_estimate(frame)
                if len(poses) > 0:
                    # Use multi-marker fusion if marker positions are available
                    if marker_positions:
                        vision_pose = estimator.fuse_multiple_markers(poses, marker_positions)
                        if vision_pose:
                            # Debug print only every 1 second
                            if current_time - last_debug_time >= 1.0:
                                print(f"🔍 DETECTION: Found {vision_pose['num_markers']} ArUco markers (IDs: {vision_pose['marker_ids']})")
                                last_debug_time = current_time
                    else:
                        # Fallback to single marker (first detected)
                        vision_pose = poses[0]
                        # Debug print only every 1 second
                        if current_time - last_debug_time >= 1.0:
                            print(f"🔍 DETECTION: Found single ArUco marker (ID: {vision_pose['id']})")
                            last_debug_time = current_time
                else:
                    # Debug print only every 1 second
                    if current_time - last_debug_time >= 1.0:
                        print(f"🔍 DETECTION: No ArUco markers visible")
                        last_debug_time = current_time
            
            # Drift detection (simplified)
            drift = None
            if odometry is not None:
                drift = {"drifting": False, "bias_norm": 0.0}
            
            # Convert vision_pose to fusion format if needed
            fusion_vision = None
            if vision_pose is not None:
                if 'num_markers' in vision_pose:
                    # Multi-marker fusion result - already in world frame
                    from scipy.spatial.transform import Rotation as R
                    fusion_vision = {
                        'tvec': vision_pose['position'],
                        'rvec': R.from_quat(vision_pose['orientation_quat']).as_rotvec()
                    }
                else:
                    # Single marker result - use as-is
                    fusion_vision = vision_pose
            
            # Fusion
            fused = fusion.update(odometry, fusion_vision, drift)
            
            # Logging
            if log_poses:
                p = fused["position"]
                odom_str = "none"
                if odometry is not None:
                    odom_pos = odometry.get('position')
                    odom_t = odometry.get('t')
                    if odom_pos is not None and odom_t is not None:
                        odom_str = f"t={odom_t:.6f} pos={odom_pos}"
                
                vision_str = "disabled" if not use_vision else "none"
                if use_vision and vision_pose is not None:
                    if 'num_markers' in vision_pose:
                        # Multi-marker fusion result
                        vision_str = f"fused({vision_pose['num_markers']} markers) pos={vision_pose['position']}"
                    else:
                        # Single marker result
                        vision_str = f"id={vision_pose['id']} tvec={vision_pose['tvec']}"
                
                # Clear, understandable logging (only every 1 second)
                if current_time - last_position_time >= 1.0:
                    #print(f"📍 CAMERA POSITION: X={p[0]:+.3f}m Y={p[2]:+.3f}m Y={p[1]:+.3f}m")
                    if odometry is not None:
                        print(f"   📊 ODOMETRY: {odom_str}")
                    if use_vision and vision_pose is not None:
                        if 'num_markers' in vision_pose:
                            print(f"   👁️  VISION: Using {vision_pose['num_markers']} markers (IDs: {vision_pose['marker_ids']})")
                            print(f"📍 CAMERA POSITION: X={p[0]:+.3f}m Y={p[2]:+.3f}m Z={p[1]:+.3f}m")
                            #print(f"   📍 CAMERA POS FROM MARKERS: {vision_pose['position']}")
                            # Show which marker is being used for reference
                            #for marker_id in vision_pose['marker_ids']:
                                #if marker_id in marker_positions:
                                    #print(f"   🎯 REFERENCE MARKER {marker_id}: {marker_positions[marker_id]}")
                        else:
                            print(f"   👁️  VISION: Single marker ID {vision_pose['id']}")
                            print(f"   📏 CAMERA RELATIVE TO MARKER: {vision_pose['tvec']}")
                            #if vision_pose['id'] in marker_positions:
                                #print(f"   🎯 REFERENCE MARKER {vision_pose['id']}: {marker_positions[vision_pose['id']]}")
                    else:
                        print(f"   👁️  VISION: No markers detected")
                    print()  # Empty line for readability
                    last_position_time = current_time
            
            # Visualization
            if visualize and frame is not None:
                vis = frame.copy()
                if vision_pose is not None:
                    # Handle both single marker and multi-marker fusion formats
                    if 'num_markers' in vision_pose:
                        # Multi-marker fusion result - use the original poses for visualization
                        # We need to get the original marker poses, not the fused result
                        poses = estimator.detect_and_estimate(frame)
                        if poses:
                            # Draw axes for ALL detected markers
                            for pose in poses:
                                cv2.drawFrameAxes(
                                    vis,
                                    estimator.camera_matrix,
                                    estimator.dist_coeffs,
                                    pose["rvec"],
                                    pose["tvec"],
                                    0.05,
                                )
                    else:
                        # Single marker result
                        cv2.drawFrameAxes(
                            vis,
                            estimator.camera_matrix,
                            estimator.dist_coeffs,
                            vision_pose["rvec"],
                            vision_pose["tvec"],
                            0.05,
                        )
                cv2.imshow("localization", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
            
            # No busy wait or sleeps; loop is paced by new frames
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        cv2.destroyAllWindows()
        stream.stop()
        print("Test completed")


if __name__ == "__main__":
    main()
