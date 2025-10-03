#!/usr/bin/env python3

import os
import sys
import json
import time
import cv2
import numpy as np
import quaternion
import argparse
import threading
import base64
import select
import glob
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

try:
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector
except ImportError as e:
    print(f"Error: Cannot import habitat_sim - {e}")
    sys.exit(1)


def scan_available_scenes(base_path="./data/scene_datasets/mp3d"):
    """Scan for available MP3D scenes and let user choose"""
    print(f"Scanning for scenes in: {base_path}")

    if not os.path.exists(base_path):
        print(f"Error: Scene directory not found: {base_path}")
        return None

    # Find all .glb files in subdirectories
    scene_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.glb'):
                scene_path = os.path.join(root, file)
                scene_name = os.path.basename(os.path.dirname(scene_path))
                scene_files.append((scene_name, scene_path))

    if not scene_files:
        print(f"No .glb scene files found in {base_path}")
        return None

    # Sort by scene name
    scene_files.sort(key=lambda x: x[0])

    print("\n" + "=" * 60)
    print("AVAILABLE SCENES")
    print("=" * 60)
    for i, (scene_name, scene_path) in enumerate(scene_files):
        print(f"{i + 1:2d}. {scene_name}")
    print("=" * 60)

    while True:
        try:
            choice = input(f"Select scene (1-{len(scene_files)}): ").strip()
            scene_index = int(choice) - 1

            if 0 <= scene_index < len(scene_files):
                selected_scene = scene_files[scene_index]
                print(f"Selected: {selected_scene[0]}")
                print(f"Path: {selected_scene[1]}")
                return selected_scene[1]
            else:
                print(f"Please enter a number between 1 and {len(scene_files)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nSelection cancelled")
            return None


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html_page().encode())
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()

            # Keep connection alive and send updates
            while True:
                if hasattr(self.server, 'latest_frame') and self.server.latest_frame is not None:
                    try:
                        _, buffer = cv2.imencode('.jpg', self.server.latest_frame)
                        frame_data = base64.b64encode(buffer).decode()
                        self.wfile.write(f"data: {frame_data}\n\n".encode())
                        self.wfile.flush()
                    except:
                        break
                time.sleep(0.1)  # 10 FPS

    def get_html_page(self):
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Drone Navigation Data Collection</title>
    <style>
        body { margin: 0; padding: 20px; background: #000; color: #fff; font-family: Arial; }
        #video-container { text-align: center; }
        #controls { margin-top: 20px; padding: 20px; background: #333; border-radius: 10px; }
        .control-section { margin: 10px 0; }
        .key { background: #555; padding: 5px 10px; border-radius: 5px; margin: 0 5px; }
        #status { margin-top: 20px; padding: 15px; background: #444; border-radius: 5px; }
        .collision-status { font-weight: bold; }
        .physics-enabled { color: #0f0; }
        .physics-disabled { color: #f00; }
        .recording-status { font-weight: bold; }
        .recording-active { color: #0f0; }
        .recording-inactive { color: #f80; }
    </style>
</head>
<body>
    <h1>🚁 Drone Indoor Navigation - Data Collection</h1>

    <div id="video-container">
        <img id="video-feed" style="max-width: 100%; height: auto; border: 2px solid #555;">
    </div>

    <div id="controls">
        <h3>Control Instructions:</h3>
        <div class="control-section">
            <strong>Movement:</strong>
            <span class="key">W</span> Forward 0.25m
            <span class="key">A</span> Turn left 15°
            <span class="key">D</span> Turn right 15°
        </div>
        <div class="control-section">
            <strong>Vertical:</strong>
            <span class="key">Q</span> Move up 0.25m
            <span class="key">E</span> Move down 0.25m
        </div>
        <div class="control-section">
            <strong>Recording:</strong>
            <span class="key">B</span> Begin recording
            <span class="key">S</span> Stop recording & save
            <span class="key">R</span> Reset current trajectory
        </div>
        <div class="control-section">
            <strong>System:</strong>
            <span class="key">N</span> Reset position
            <span class="key">ESC</span> Exit
        </div>
    </div>

    <div id="status">
        <div id="recording-status" class="recording-status recording-inactive">Recording: Inactive</div>
        <div id="waypoint-count">Waypoints: 0</div>
        <div id="position">Position: [0.00, 0.00, 0.00]</div>
        <div id="height-info">Height: 0.00m</div>
        <div id="physics-status" class="collision-status physics-enabled">Physics: Enabled</div>
        <div id="collision-info">Collision: OK</div>
    </div>

    <script>
        const videoFeed = document.getElementById('video-feed');
        const eventSource = new EventSource('/stream');

        eventSource.onmessage = function(event) {
            videoFeed.src = 'data:image/jpeg;base64,' + event.data;
        };

        // Focus handling for keyboard input
        document.addEventListener('keydown', function(event) {
            // This page is just for viewing - actual control is still in the terminal
            console.log('Key pressed:', event.key);
        });

        setInterval(function() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    if (data.recording !== undefined) {
                        const recordingEl = document.getElementById('recording-status');
                        recordingEl.textContent = `Recording: ${data.recording ? 'Active' : 'Inactive'}`;
                        recordingEl.className = data.recording ? 'recording-status recording-active' : 'recording-status recording-inactive';
                    }
                    if (data.waypoints !== undefined) {
                        document.getElementById('waypoint-count').textContent = `Waypoints: ${data.waypoints}`;
                    }
                    if (data.position) {
                        document.getElementById('position').textContent = `Position: ${data.position}`;
                    }
                    if (data.height !== undefined) {
                        document.getElementById('height-info').textContent = `Height: ${data.height}m`;
                    }
                    if (data.physics_enabled !== undefined) {
                        const physicsEl = document.getElementById('physics-status');
                        physicsEl.textContent = `Physics: ${data.physics_enabled ? 'Enabled' : 'Disabled'}`;
                        physicsEl.className = data.physics_enabled ? 'collision-status physics-enabled' : 'collision-status physics-disabled';
                    }
                    if (data.collision_info) {
                        document.getElementById('collision-info').textContent = `Collision: ${data.collision_info}`;
                    }
                })
                .catch(() => {}); // Ignore errors
        }, 1000);
    </script>
</body>
</html>
        """


class DroneDataCollectorStreaming:
    def __init__(self, scene_path, output_dir="./drone_navigation_data",
                 sensor_height=1.5, image_width=640, image_height=480,
                 stream_port=8080):
        """
        Initialize drone data collector with streaming capability and proper collision detection
        """
        self.scene_path = scene_path
        self.output_dir = Path(output_dir)
        self.sensor_height = sensor_height
        self.image_width = image_width
        self.image_height = image_height
        self.stream_port = stream_port

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Movement parameters
        self.move_amount = 0.25
        self.turn_amount = 15.0
        self.vertical_amount = 0.25

        # Flight constraints for indoor drone navigation
        self.min_height = 0.5
        self.max_height = 3.0
        self.boundary_limit = 15.0

        # Collision detection parameters
        self.collision_buffer = 0.15  # Safety buffer around drone (30cm)
        self.max_ray_distance = 1.5  # Maximum distance for ray casting

        # Physics state
        self.physics_enabled = False

        # Recording state - Track whether we're currently recording
        self.is_recording = False
        self.recording_session = 0  # Track session number for unique naming
        self.current_path_id = None  # Track current trajectory path ID

        # Action tracking - NEW: Track the last action taken
        self.last_action = None
        
        # Trajectory data - Modified for R2R format
        self.current_trajectory = None
        self.all_trajectories = []  # Store all collected trajectories

        # Initialize simulator
        self.sim = None
        self.agent = None
        self.waypoint_count = 0

        # Streaming server
        self.server = None
        self.server_thread = None
        self.latest_frame = None

        # Collision status for display
        self.last_collision_check = "OK"

        print(f"Initializing drone data collector with physics-based collision detection")
        print(f"Stream will be available at: http://localhost:{stream_port}")

    def setup_simulator(self):
        """Setup Habitat simulator with physics enabled for collision detection"""
        print("Setting up simulator with physics enabled...")

        try:
            # Simulator configuration with physics enabled
            backend_cfg = habitat_sim.SimulatorConfiguration()
            backend_cfg.scene_id = self.scene_path
            backend_cfg.enable_physics = True  # ✅ Enable physics for collision detection
            backend_cfg.allow_sliding = False  # Prevent sliding through objects
            backend_cfg.create_renderer = True
            backend_cfg.leave_context_with_background_renderer = False

            # Agent configuration
            agent_cfg = habitat_sim.agent.AgentConfiguration()

            # RGB sensor configuration
            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = "color_sensor"
            rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_sensor_spec.resolution = [self.image_height, self.image_width]
            rgb_sensor_spec.position = [0.0, 0.0, 0.0]
            rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

            # Depth sensor configuration
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [self.image_height, self.image_width]
            depth_sensor_spec.position = [0.0, 0.0, 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

            agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

            # Create simulator
            cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
            self.sim = habitat_sim.Simulator(cfg)

            # Initialize agent
            self.agent = self.sim.initialize_agent(0)
            self.reset_agent_to_navigable_position()

            # Check if physics is actually enabled
            try:
                physics_lib = self.sim.get_physics_simulation_library()
                self.physics_enabled = physics_lib != habitat_sim.physics.PhysicsSimulationLibrary.NoPhysics
                print(f"Physics library: {physics_lib}")
                print(f"Physics enabled: {self.physics_enabled}")
            except Exception as e:
                print(f"Warning: Could not check physics status: {e}")
                self.physics_enabled = False

            print("Simulator setup completed!")

            if self.physics_enabled:
                print("✅ Physics-based collision detection is active")
            else:
                print("⚠️ Physics not available, using fallback collision detection")

        except Exception as e:
            print(f"Simulator setup failed: {e}")
            print("This might be due to:")
            print("1. MP3D scene compatibility issues with Bullet physics")
            print("2. Missing physics dependencies")
            print("3. Corrupted scene files")
            print("\nTrying fallback configuration...")

            # Fallback: try without physics but with simpler collision
            self._setup_fallback_simulator()

    def _setup_fallback_simulator(self):
        """Fallback simulator setup without physics"""
        try:
            backend_cfg = habitat_sim.SimulatorConfiguration()
            backend_cfg.scene_id = self.scene_path
            backend_cfg.enable_physics = False

            agent_cfg = habitat_sim.agent.AgentConfiguration()

            # RGB sensor configuration
            rgb_sensor_spec = habitat_sim.CameraSensorSpec()
            rgb_sensor_spec.uuid = "color_sensor"
            rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            rgb_sensor_spec.resolution = [self.image_height, self.image_width]
            rgb_sensor_spec.position = [0.0, 0.0, 0.0]
            rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

            # Depth sensor configuration
            depth_sensor_spec = habitat_sim.CameraSensorSpec()
            depth_sensor_spec.uuid = "depth_sensor"
            depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
            depth_sensor_spec.resolution = [self.image_height, self.image_width]
            depth_sensor_spec.position = [0.0, 0.0, 0.0]
            depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

            agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

            cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
            self.sim = habitat_sim.Simulator(cfg)
            self.agent = self.sim.initialize_agent(0)
            self.reset_agent_to_navigable_position()

            self.physics_enabled = False
            print("⚠️ Using fallback mode without physics")

        except Exception as e:
            print(f"Fallback simulator setup also failed: {e}")
            raise

    def setup_streaming_server(self):
        """Setup HTTP streaming server"""
        try:
            # Create server
            self.server = HTTPServer(('localhost', self.stream_port), StreamingHandler)
            self.server.latest_frame = None

            # Start server in separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()

            print(f"Streaming server started at http://localhost:{self.stream_port}")
            print("Open this URL in your browser to view the drone feed")

        except Exception as e:
            print(f"Failed to start streaming server: {e}")
            self.server = None

    def update_display_frame(self):
        """Update frame for streaming with collision information"""
        obs = self.get_observations()
        rgb_image = obs['color_sensor']

        # Create display frame with overlays
        display_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        # Add recording status overlay
        recording_text = f"Recording: {'ACTIVE' if self.is_recording else 'INACTIVE'}"
        recording_color = (0, 255, 0) if self.is_recording else (0, 165, 255)  # Green if recording, orange if not
        cv2.putText(display_image, recording_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, recording_color, 2)

        # Add text overlays
        cv2.putText(display_image, f"Waypoints: {self.waypoint_count}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        agent_state = self.agent.get_state()
        pos_text = f"Pos: [{agent_state.position[0]:.2f}, {agent_state.position[1]:.2f}, {agent_state.position[2]:.2f}]"
        cv2.putText(display_image, pos_text,
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        height_text = f"Height: {agent_state.position[1]:.2f}m ({self.min_height}-{self.max_height}m)"
        cv2.putText(display_image, height_text,
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Physics and collision status
        physics_text = f"Physics: {'Enabled' if self.physics_enabled else 'Disabled'}"
        physics_color = (0, 255, 0) if self.physics_enabled else (0, 0, 255)
        cv2.putText(display_image, physics_text,
                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, physics_color, 2)

        collision_text = f"Collision: {self.last_collision_check}"
        cv2.putText(display_image, collision_text,
                    (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Add control instructions
        instructions = [
            "W:Forward A:Left D:Right Q:Up E:Down",
            "B:Begin S:Stop&Save R:Reset N:ResetPos ESC:Exit"
        ]

        for i, instruction in enumerate(instructions):
            cv2.putText(display_image, instruction,
                        (10, display_image.shape[0] - 40 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Update server frame
        if self.server:
            self.server.latest_frame = display_image

        return display_image

    def reset_agent_to_navigable_position(self):
        """Reset agent to a navigable position"""
        try:
            if hasattr(self.sim, 'pathfinder') and self.sim.pathfinder.is_loaded:
                nav_point = self.sim.pathfinder.get_random_navigable_point()
                agent_state = habitat_sim.AgentState()
                agent_state.position = nav_point + np.array([0, self.sensor_height, 0])
                agent_state.rotation = np.quaternion(1, 0, 0, 0)
                self.agent.set_state(agent_state)
                print(f"Agent position reset to: {nav_point}")
            else:
                agent_state = habitat_sim.AgentState()
                agent_state.position = np.array([0.0, self.sensor_height, 0.0])
                agent_state.rotation = np.quaternion(1, 0, 0, 0)
                self.agent.set_state(agent_state)
        except Exception as e:
            print(f"Position reset error: {e}")
            agent_state = habitat_sim.AgentState()
            agent_state.position = np.array([0.0, self.sensor_height, 0.0])
            agent_state.rotation = np.quaternion(1, 0, 0, 0)
            self.agent.set_state(agent_state)

    def get_observations(self):
        """Get current observation data"""
        return self.sim.get_sensor_observations()

    def check_collision_with_physics(self, target_position):
        """
        Physics-based collision detection using ray casting
        Returns (is_safe, collision_info)
        """
        if not self.physics_enabled:
            return self._fallback_collision_check(target_position)

        try:
            current_pos = self.agent.get_state().position

            # Check basic boundaries first
            if (abs(target_position[0]) > self.boundary_limit or
                    abs(target_position[2]) > self.boundary_limit or
                    target_position[1] < self.min_height or
                    target_position[1] > self.max_height):
                self.last_collision_check = "Boundary violation"
                return False, "Out of bounds"

            # Multiple ray directions to check around the drone
            check_directions = [
                np.array([0, 0, 0]),  # Center
                np.array([0.2, 0, 0]),  # Right
                np.array([-0.2, 0, 0]),  # Left
                np.array([0, 0, 0.2]),  # Forward
                np.array([0, 0, -0.2]),  # Backward
                np.array([0, 0.2, 0]),  # Up
                np.array([0, -0.2, 0]),  # Down
                # Diagonal checks
                np.array([0.15, 0, 0.15]),  # Forward-right
                np.array([-0.15, 0, 0.15]),  # Forward-left
                np.array([0.15, 0, -0.15]),  # Backward-right
                np.array([-0.15, 0, -0.15]),  # Backward-left
            ]

            # Cast rays from current position to target position with offsets
            for direction in check_directions:
                test_target = target_position + direction
                ray_direction = test_target - current_pos
                ray_length = np.linalg.norm(ray_direction)

                if ray_length > 0:
                    ray_direction_normalized = ray_direction / ray_length
                    ray = habitat_sim.geo.Ray(current_pos, ray_direction_normalized)

                    # Cast ray with appropriate max distance
                    max_distance = min(ray_length + self.collision_buffer, self.max_ray_distance)
                    raycast_results = self.sim.cast_ray(ray, max_distance)

                    if raycast_results.has_hits():
                        hit_distance = raycast_results.hits[0].ray_distance
                        # If hit is closer than our movement + buffer, it's a collision
                        if hit_distance < ray_length + self.collision_buffer:
                            self.last_collision_check = f"Obstacle at {hit_distance:.2f}m"
                            return False, f"Collision risk: obstacle at {hit_distance:.2f}m"

            # Additional check: ray from target back to current (reverse check)
            reverse_direction = current_pos - target_position
            reverse_length = np.linalg.norm(reverse_direction)
            if reverse_length > 0:
                reverse_direction_normalized = reverse_direction / reverse_length
                reverse_ray = habitat_sim.geo.Ray(target_position, reverse_direction_normalized)
                reverse_results = self.sim.cast_ray(reverse_ray, reverse_length + self.collision_buffer)

                if reverse_results.has_hits():
                    hit_distance = reverse_results.hits[0].ray_distance
                    if hit_distance < reverse_length + self.collision_buffer:
                        self.last_collision_check = f"Reverse check failed at {hit_distance:.2f}m"
                        return False, f"Reverse collision check failed"

            self.last_collision_check = "Clear"
            return True, "Clear path"

        except Exception as e:
            print(f"Physics collision check failed: {e}")
            # Fallback to simple boundary check
            return self._fallback_collision_check(target_position)

    def _fallback_collision_check(self, position):
        """Fallback collision detection method when physics is not available"""
        is_safe = (abs(position[0]) < self.boundary_limit and
                   abs(position[2]) < self.boundary_limit and
                   self.min_height <= position[1] <= self.max_height)

        if is_safe:
            self.last_collision_check = "Clear (no physics)"
            return True, "Boundary check OK"
        else:
            self.last_collision_check = "Boundary violation"
            return False, "Out of bounds"

    # Movement methods with improved collision detection and action tracking
    def move_forward(self):
        agent_state = self.agent.get_state()
        forward_direction = quat_rotate_vector(agent_state.rotation, np.array([0, 0, -1])).astype(np.float32)
        new_position = agent_state.position + forward_direction * self.move_amount

        is_safe, collision_info = self.check_collision_with_physics(new_position)
        if is_safe:
            agent_state.position = new_position
            self.agent.set_state(agent_state)
            self.last_action = "move_forward"  # NEW: Record action
            return True
        else:
            print(f"❌ Forward movement blocked: {collision_info}")
            return False

    def turn_left(self):
        agent_state = self.agent.get_state()
        turn_quat = quat_from_angle_axis(np.radians(self.turn_amount), np.array([0, 1, 0]))
        agent_state.rotation = turn_quat * agent_state.rotation
        self.agent.set_state(agent_state)
        self.last_action = "turn_left"  # NEW: Record action
        return True

    def turn_right(self):
        agent_state = self.agent.get_state()
        turn_quat = quat_from_angle_axis(np.radians(-self.turn_amount), np.array([0, 1, 0]))
        agent_state.rotation = turn_quat * agent_state.rotation
        self.agent.set_state(agent_state)
        self.last_action = "turn_right"  # NEW: Record action
        return True

    def move_up(self):
        agent_state = self.agent.get_state()
        new_position = agent_state.position + np.array([0, self.vertical_amount, 0])

        is_safe, collision_info = self.check_collision_with_physics(new_position)
        if is_safe:
            agent_state.position = new_position
            self.agent.set_state(agent_state)
            self.last_action = "move_up"  # NEW: Record action
            return True
        else:
            print(f"❌ Upward movement blocked: {collision_info}")
            return False

    def move_down(self):
        agent_state = self.agent.get_state()
        new_position = agent_state.position - np.array([0, self.vertical_amount, 0])

        is_safe, collision_info = self.check_collision_with_physics(new_position)
        if is_safe:
            agent_state.position = new_position
            self.agent.set_state(agent_state)
            self.last_action = "move_down"  # NEW: Record action
            return True
        else:
            print(f"❌ Downward movement blocked: {collision_info}")
            return False

    def begin_recording(self):
        """Begin a new recording session with R2R format"""
        if self.is_recording:
            print("❌ Already recording! Stop current recording first.")
            return False

        self.is_recording = True
        self.recording_session += 1
        self.waypoint_count = 0

        # Generate unique path ID for R2R format
        scene_name = os.path.basename(self.scene_path).replace('.glb', '')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.current_path_id = f"{scene_name}_{self.recording_session}_{timestamp}"

        # Create directory for this trajectory's images
        self.current_trajectory_dir = self.output_dir / self.current_path_id
        self.current_trajectory_dir.mkdir(parents=True, exist_ok=True)

        # Initialize trajectory data in R2R format
        self.current_trajectory = {
            "path_id": self.current_path_id,
            "scene_id": scene_name,
            "path": [],  # Will contain waypoint data
            "heading": self._get_agent_heading(),
            "start_position": self.agent.get_state().position.tolist(),
            "start_rotation": [
                self.agent.get_state().rotation.w,
                self.agent.get_state().rotation.x,
                self.agent.get_state().rotation.y,
                self.agent.get_state().rotation.z
            ],
            "collection_metadata": {
                "collection_time": datetime.now().isoformat(),
                "sensor_height": self.sensor_height,
                "image_size": [self.image_width, self.image_height],
                "movement_params": {
                    "move_amount": self.move_amount,
                    "turn_amount": self.turn_amount,
                    "vertical_amount": self.vertical_amount
                },
                "collision_params": {
                    "collision_buffer": self.collision_buffer,
                    "max_ray_distance": self.max_ray_distance,
                    "min_height": self.min_height,
                    "max_height": self.max_height
                },
                "physics_enabled": self.physics_enabled
            }
        }

        # Save initial waypoint (action will be None for the first waypoint)
        self.last_action = None  # No action to get to start position
        self.save_waypoint()

        print(f"🎬 Recording session {self.recording_session} started!")
        print(f"Path ID: {self.current_path_id}")
        print(f"Initial waypoint saved. Total waypoints: {self.waypoint_count}")
        return True

    def stop_recording_and_save(self):
        """Stop current recording and save trajectory in R2R format"""
        if not self.is_recording:
            print("❌ Not currently recording!")
            return False

        self.is_recording = False

        # Update metadata
        self.current_trajectory["collection_metadata"]["session_end_time"] = datetime.now().isoformat()
        self.current_trajectory["collection_metadata"]["total_waypoints"] = len(self.current_trajectory["path"])
        
        # Add end position info
        agent_state = self.agent.get_state()
        self.current_trajectory["end_position"] = agent_state.position.tolist()
        self.current_trajectory["end_rotation"] = [
            agent_state.rotation.w, agent_state.rotation.x,
            agent_state.rotation.y, agent_state.rotation.z
        ]

        # Add to all trajectories list
        self.all_trajectories.append(self.current_trajectory)

        # Save/update the annotations.json file
        self.save_annotations()

        print(f"🛑 Recording session {self.recording_session} stopped and saved!")
        print(f"Trajectory saved with {len(self.current_trajectory['path'])} waypoints")
        return True

    def _get_agent_heading(self):
        """Get agent heading in degrees (0-360)"""
        agent_state = self.agent.get_state()
        # Convert quaternion to euler angles and extract yaw
        rotation = agent_state.rotation
        # Simple heading calculation from quaternion
        heading = np.degrees(np.arctan2(
            2.0 * (rotation.w * rotation.y + rotation.x * rotation.z),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z)
        ))
        # Normalize to 0-360
        if heading < 0:
            heading += 360
        return heading

    def save_waypoint(self):
        """Save current position as waypoint with action ground truth - R2R format"""
        if not self.is_recording:
            return  # Don't save waypoints when not recording

        agent_state = self.agent.get_state()
        obs = self.get_observations()

        # Create waypoint data in R2R-inspired format
        waypoint_data = {
            "viewpoint_id": f"{self.current_path_id}_{self.waypoint_count:04d}",
            "position": agent_state.position.tolist(),
            "rotation": [
                agent_state.rotation.w, agent_state.rotation.x,
                agent_state.rotation.y, agent_state.rotation.z
            ],
            "heading": self._get_agent_heading(),
            "action": self.last_action,  # NEW: Action ground truth
            "timestamp": time.time(),
            "physics_enabled": self.physics_enabled,
            "collision_status": self.last_collision_check
        }

        # Save RGB image with R2R-style naming
        rgb_image = obs['color_sensor']
        rgb_filename = f"{self.waypoint_count:04d}_rgb.png"
        rgb_path = self.current_trajectory_dir / rgb_filename
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        waypoint_data['rgb_image'] = rgb_filename

        # Save depth image
        depth_image = obs['depth_sensor']
        depth_filename = f"{self.waypoint_count:04d}_depth.png"
        depth_path = self.current_trajectory_dir / depth_filename
        depth_mm = (depth_image * 1000).astype(np.uint16)
        cv2.imwrite(str(depth_path), depth_mm)
        waypoint_data['depth_image'] = depth_filename

        # Add to current trajectory
        self.current_trajectory["path"].append(waypoint_data)
        self.waypoint_count += 1

        print(f"📍 Waypoint {self.waypoint_count - 1} saved (Path: {self.current_path_id})")
        print(f"Position: [{agent_state.position[0]:.2f}, {agent_state.position[1]:.2f}, {agent_state.position[2]:.2f}]")
        print(f"Action: {self.last_action}")
        print(f"Collision status: {self.last_collision_check}")

    def save_annotations(self):
        """Save all trajectories to annotations.json in R2R format"""
        annotations_file = self.output_dir / "annotations.json"
        
        # Create annotations data in R2R format
        annotations_data = {
            "version": "1.0",
            "split": "custom_collection",
            "dataset": "drone_navigation_mp3d", 
            "collection_info": {
                "collection_date": datetime.now().isoformat(),
                "scene_path": self.scene_path,
                "total_trajectories": len(self.all_trajectories),
                "physics_enabled": self.physics_enabled,
                "sensor_config": {
                    "height": self.sensor_height,
                    "image_width": self.image_width,
                    "image_height": self.image_height
                }
            },
            "trajectories": self.all_trajectories
        }

        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(annotations_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Annotations saved to: {annotations_file}")
        print(f"Total trajectories in dataset: {len(self.all_trajectories)}")

    def reset_trajectory(self):
        """Reset current trajectory data"""
        if self.is_recording:
            print("⚠️ Resetting trajectory while recording is active!")
            # Reset current trajectory
            if self.current_trajectory:
                self.current_trajectory["path"] = []
            self.waypoint_count = 0
            self.last_action = None
        print("🔄 Current trajectory data reset")

    def get_trajectory_name_input(self, fd, old_settings):
        """Safely get trajectory name input by temporarily restoring terminal settings"""
        try:
            # Restore normal terminal settings
            import termios
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            # Get input normally
            print("\n" + "=" * 50)
            trajectory_name = input("Enter trajectory name (or press Enter for default): ").strip()
            print("=" * 50)

            # Set raw mode again
            import tty
            tty.setraw(fd)

            return trajectory_name if trajectory_name else None

        except Exception as e:
            print(f"Input error: {e}")
            # Make sure we restore raw mode even if there's an error
            try:
                import tty
                tty.setraw(fd)
            except:
                pass
            return None

    def run_terminal_collection(self):
        """Run data collection from terminal with web streaming"""
        self.setup_simulator()
        self.setup_streaming_server()

        print("\n" + "=" * 60)
        print("🚁 DRONE DATA COLLECTION - R2R FORMAT MODE")
        print("=" * 60)
        print(f"📺 View live feed: http://localhost:{self.stream_port}")
        print(f"⚡ Physics: {'Enabled' if self.physics_enabled else 'Disabled (Fallback mode)'}")
        print("⌨️ Control from this terminal:")
        print("   W - Forward    A - Left      D - Right")
        print("   Q - Up         E - Down")
        print("   B - Begin recording       S - Stop recording & save")
        print("   R - Reset trajectory      N - Reset position")
        print("   ESC/Ctrl+C - Exit")
        print("=" * 60)

        # Start without recording - user must press B to begin
        self.update_display_frame()

        # Terminal settings variables
        fd = None
        old_settings = None
        use_raw_mode = True

        # Try to set up raw mode terminal input
        try:
            import termios, tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setraw(sys.stdin.fileno())
            print(f"\nReady! Press B to begin recording.")
            print("Open the web interface to see the drone view.")
            print("Using raw mode input (single key presses)")
        except ImportError:
            print("termios not available, using line input mode")
            use_raw_mode = False
        except Exception as e:
            print(f"Cannot set raw mode: {e}, using line input mode")
            use_raw_mode = False

        try:
            if use_raw_mode:
                # Raw mode input loop
                while True:
                    # Update display frame
                    self.update_display_frame()

                    # Check for keyboard input (non-blocking)
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        char = sys.stdin.read(1)

                        action_taken = False

                        if char.lower() == 'w':
                            if self.move_forward():
                                print("✅ Moved forward")
                                action_taken = True
                            else:
                                print("❌ Forward movement blocked")

                        elif char.lower() == 'a':
                            self.turn_left()
                            print("↰ Turned left")
                            action_taken = True

                        elif char.lower() == 'd':
                            self.turn_right()
                            print("↱ Turned right")
                            action_taken = True

                        elif char.lower() == 'q':
                            if self.move_up():
                                print("⬆️ Moved up")
                                action_taken = True
                            else:
                                print("❌ Cannot move up")

                        elif char.lower() == 'e':
                            if self.move_down():
                                print("⬇️ Moved down")
                                action_taken = True
                            else:
                                print("❌ Cannot move down")
                        elif ord(char) == 3:  # Ctrl+C
                            print("\n🛑 Exiting (Ctrl+C)...")
                            break
                        elif char.lower() == 'p':
                            print("\n🛑 Exiting (P key)...")
                            break
                        elif ord(char) == 27:  # ESC
                            print("\n🛑 Exiting (ESC)...")
                            break

                        elif char.lower() == 'b':
                            self.begin_recording()

                        elif char.lower() == 's':
                            self.stop_recording_and_save()

                        elif char.lower() == 'r':
                            self.reset_trajectory()

                        elif char.lower() == 'n':
                            self.reset_agent_to_navigable_position()
                            print("🔄 Position reset")

                        elif ord(char) == 27:  # ESC
                            break

                        # Only save waypoint after movement if we're recording
                        if action_taken and self.is_recording:
                            self.save_waypoint()

                    time.sleep(0.01)  # Small delay

            else:
                # Fallback line input mode
                print("Using line input mode (type commands and press Enter)")
                print("Commands: w, a, d, q, e, b, s, r, n, exit")

                while True:
                    self.update_display_frame()
                    try:
                        command = input("Command: ").strip().lower()

                        action_taken = False

                        if command == 'w':
                            if self.move_forward():
                                print("✅ Moved forward")
                                action_taken = True
                            else:
                                print("❌ Forward movement blocked")
                        elif command == 'a':
                            self.turn_left()
                            print("↰ Turned left")
                            action_taken = True
                        elif command == 'd':
                            self.turn_right()
                            print("↱ Turned right")
                            action_taken = True
                        elif command == 'q':
                            if self.move_up():
                                print("⬆️ Moved up")
                                action_taken = True
                            else:
                                print("❌ Cannot move up")
                        elif command == 'e':
                            if self.move_down():
                                print("⬇️ Moved down")
                                action_taken = True
                            else:
                                print("❌ Cannot move down")
                        elif command == 'b':
                            self.begin_recording()
                        elif command == 's':
                            self.stop_recording_and_save()
                        elif command == 'r':
                            self.reset_trajectory()
                        elif command == 'n':
                            self.reset_agent_to_navigable_position()
                            print("🔄 Position reset")
                        elif command == 'p':
                            print("🛑 Exiting...")
                            break
                        elif command in ['exit', 'quit']:
                            break

                        # Only save waypoint after movement if we're recording
                        if action_taken and self.is_recording:
                            self.save_waypoint()

                    except EOFError:
                        break
                    except KeyboardInterrupt:
                        break

        except KeyboardInterrupt:
            print("\n\n🛑 Collection stopped by user")

        finally:
            # Restore terminal settings if they were changed
            if old_settings is not None and fd is not None:
                try:
                    import termios
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                except:
                    pass

            # Cleanup
            if self.server:
                try:
                    self.server.shutdown()
                except:
                    pass

            if self.sim:
                try:
                    self.sim.close()
                except:
                    pass

            # Auto-save trajectory if we're still recording
            if self.is_recording and self.current_trajectory and len(self.current_trajectory["path"]) > 0:
                print("Auto-saving current recording...")
                self.stop_recording_and_save()

            print("\n✅ Data collection completed!")
            if self.all_trajectories:
                print(f"📊 Total trajectories collected: {len(self.all_trajectories)}")
                print("📁 Data saved in R2R format:")
                print(f"   - annotations.json contains all trajectory metadata")
                print(f"   - Each trajectory's images are in separate subdirectories")


def main():
    parser = argparse.ArgumentParser(description='Drone data collector with R2R format output')
    parser.add_argument('--scene', type=str,
                        help='MP3D scene file path (optional, will prompt for selection if not provided)')
    parser.add_argument('--output', type=str, default='./drone_navigation_data', help='Output directory')
    parser.add_argument('--height', type=float, default=0.5, help='Sensor height')
    parser.add_argument('--port', type=int, default=8080, help='Streaming port')

    args = parser.parse_args()

    # Scene selection logic
    scene_path = args.scene
    if not scene_path:
        # Scan and let user choose scene
        scene_path = scan_available_scenes()
        if not scene_path:
            print("No scene selected or found. Exiting.")
            sys.exit(1)
    else:
        # Validate provided scene path
        if not os.path.exists(scene_path):
            print(f"Error: Scene file not found: {scene_path}")
            print("Scanning for available scenes instead...")
            scene_path = scan_available_scenes()
            if not scene_path:
                print("No scene selected or found. Exiting.")
                sys.exit(1)

    # Check if Habitat-sim was built with Bullet physics
    try:
        # This will help us understand what physics libraries are available
        print("Checking available physics libraries...")
    except Exception as e:
        print(f"Note: {e}")

    collector = DroneDataCollectorStreaming(
        scene_path=scene_path,
        output_dir=args.output,
        sensor_height=args.height,
        stream_port=args.port
    )

    collector.run_terminal_collection()


if __name__ == "__main__":
    main()