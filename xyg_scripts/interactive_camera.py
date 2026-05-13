"""
Interactive camera control for LIBERO environments.
This script creates a web interface to control the camera view in real-time.
Works with VSCode remote connection via port forwarding.
"""
import argparse
import os
import numpy as np
from PIL import Image
import base64
import io
from flask import Flask, render_template, request, jsonify

import transforms3d.euler as euler

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import ControlEnv

# Flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching

# Global variables
env = None
camera_id = None
robot_base_name = None
camera_name = None
image_resolution = None
mujoco_wxyz = True
temp_dir = "temp_camera_views"

# Camera position and orientation
camera_pos = None
camera_quat = None

# Movement parameters
move_speed = None
rotate_speed = None


class Pose:
    # must be wxyz quaternion format
    def __init__(self, position=None, orientation=None):
        if position is None:
            position = np.zeros(3)
        if orientation is None:
            orientation = np.array([1.0, 0, 0, 0])

        self.position = np.array(position)
        self.orientation = np.array(orientation)

    def get_position(self):
        return self.position

    def get_orientation(self):
        return self.orientation

    def set_position(self, position):
        self.position = np.array(position)

    def set_orientation(self, orientation):
        self.orientation = np.array(orientation)

    def transform(self, pose):
        """Combine two poses"""
        # Using simplified transformation for demonstration
        new_position = self.position + pose.position
        # For simplicity, just using basic quaternion multiplication
        # This is simplified and might not be physically accurate
        w1, x1, y1, z1 = self.orientation
        w2, x2, y2, z2 = pose.orientation
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        new_orientation = np.array([w, x, y, z])
        return Pose(new_position, new_orientation)


def get_libero_env(task, resolution=512):
    """Initialize and return LIBERO environment and task description"""
    global camera_name
    
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file, 
        "camera_heights": resolution, 
        "camera_widths": resolution, 
        "hard_reset": False,
        "has_renderer": False,  # Set to False for headless rendering
        "has_offscreen_renderer": True,  # Need this for offscreen rendering
        "use_camera_obs": False, 
        "render_camera": camera_name
    }
    env = ControlEnv(**env_args)
    env.seed(0)
    return env, task_description


def update_camera():
    """Update camera position and orientation in the environment"""
    global env, camera_id, camera_pos, camera_quat
    
    env.sim.model.cam_pos[camera_id] = camera_pos
    env.sim.model.cam_quat[camera_id] = camera_quat
    env.sim.forward()


def move_camera(direction):
    """Move camera in the specified direction"""
    global camera_pos, camera_quat, move_speed
    
    # Extract rotation matrix from quaternion to get camera axes
    # For simplicity, we'll use basic transformations
    # Forward/backward: move along camera's z-axis
    # Left/right: move along camera's x-axis
    # Up/down: move along world's y-axis
    
    if direction == "forward":
        camera_pos[2] -= move_speed
    elif direction == "backward":
        camera_pos[2] += move_speed
    elif direction == "left":
        camera_pos[0] -= move_speed
    elif direction == "right":
        camera_pos[0] += move_speed
    elif direction == "up":
        camera_pos[1] += move_speed
    elif direction == "down":
        camera_pos[1] -= move_speed
    
    update_camera()


def rotate_camera(axis, angle):
    """Rotate camera around the specified axis by the given angle (in degrees)"""
    global camera_quat, mujoco_wxyz
    
    angle_rad = angle * np.pi / 180.0
    
    if axis == "x":
        rot_quat = euler.euler2quat(angle_rad, 0, 0, "sxyz")
    elif axis == "y":
        rot_quat = euler.euler2quat(0, angle_rad, 0, "sxyz")
    elif axis == "z":
        rot_quat = euler.euler2quat(0, 0, angle_rad, "sxyz")
    
    # Convert to wxyz format if needed
    if not mujoco_wxyz:
        rot_quat = np.array([rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2]])
    
    # Apply rotation
    w1, x1, y1, z1 = camera_quat
    w2, x2, y2, z2 = rot_quat
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    camera_quat = np.array([w, x, y, z])
    
    update_camera()


def get_camera_image():
    """Get the current camera image as a base64 string"""
    global env, camera_name, image_resolution
    
    # Render the camera image
    camera_img = env.sim.render(
        camera_name=camera_name,
        width=image_resolution,
        height=image_resolution,
        depth=False,
        mode='offscreen'
    )
    
    # Convert the image to base64
    pil_img = Image.fromarray(camera_img[::-1])
    img_io = io.BytesIO()
    pil_img.save(img_io, 'PNG')
    img_io.seek(0)
    encoded_img = base64.b64encode(img_io.getvalue()).decode('utf-8')
    
    return encoded_img


# Flask routes
@app.route('/')
def index():
    """Serve the main page"""
    return render_template('camera_control.html')


@app.route('/camera_image')
def camera_image():
    """Return the current camera image"""
    return jsonify({'image': get_camera_image()})


@app.route('/control', methods=['POST'])
def control():
    """Handle control commands"""
    command = request.json.get('command', '')
    
    if command == 'move_forward':
        move_camera('forward')
    elif command == 'move_backward':
        move_camera('backward')
    elif command == 'move_left':
        move_camera('left')
    elif command == 'move_right':
        move_camera('right')
    elif command == 'move_up':
        move_camera('up')
    elif command == 'move_down':
        move_camera('down')
    elif command == 'rotate_left':
        rotate_camera('y', -rotate_speed)
    elif command == 'rotate_right':
        rotate_camera('y', rotate_speed)
    elif command == 'rotate_up':
        rotate_camera('x', -rotate_speed)
    elif command == 'rotate_down':
        rotate_camera('x', rotate_speed)
    elif command == 'rotate_clockwise':
        rotate_camera('z', rotate_speed)
    elif command == 'rotate_counterclockwise':
        rotate_camera('z', -rotate_speed)
    elif command == 'reset':
        initialize_camera()
    
    return jsonify({'status': 'success', 'image': get_camera_image()})


def initialize_camera():
    """Initialize camera position and orientation"""
    global env, camera_id, camera_pos, camera_quat
    
    # Get current camera position and orientation
    camera_pos = env.sim.model.cam_pos[camera_id].copy()
    camera_quat = env.sim.model.cam_quat[camera_id].copy()


def create_html_template():
    """Create HTML template for the web interface"""
    global image_resolution
    
    os.makedirs('templates', exist_ok=True)
    with open('templates/camera_control.html', 'w') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Camera Control</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .camera-view {{
            border: 2px solid #333;
            margin-bottom: 20px;
        }}
        .controls {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }}
        button {{
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            border-radius: 5px;
        }}
        button:hover {{
            background-color: #45a049;
        }}
        .instructions {{
            background-color: #fff;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 20px;
            max-width: 600px;
        }}
        .status {{
            margin-top: 10px;
            padding: 8px;
            background-color: #f0f0f0;
            border-radius: 4px;
            font-family: monospace;
        }}
    </style>
</head>
<body>
    <h1>Interactive Camera Control</h1>
    
    <div class="camera-view">
        <img id="camera-image" src="" alt="Camera View" width="{image_resolution}" height="{image_resolution}">
    </div>
    
    <div class="status">
        <p>Camera Position: <span id="camera-position">Loading...</span></p>
        <p>Camera Orientation: <span id="camera-orientation">Loading...</span></p>
    </div>
    
    <div class="controls">
        <button onclick="sendCommand('rotate_left')">Rotate Left</button>
        <button onclick="sendCommand('move_forward')">Forward</button>
        <button onclick="sendCommand('rotate_right')">Rotate Right</button>
        
        <button onclick="sendCommand('move_left')">Left</button>
        <button onclick="sendCommand('move_backward')">Backward</button>
        <button onclick="sendCommand('move_right')">Right</button>
        
        <button onclick="sendCommand('rotate_up')">Look Up</button>
        <button onclick="sendCommand('move_up')">Up</button>
        <button onclick="sendCommand('rotate_down')">Look Down</button>
        
        <button onclick="sendCommand('rotate_counterclockwise')">Roll CCW</button>
        <button onclick="sendCommand('move_down')">Down</button>
        <button onclick="sendCommand('rotate_clockwise')">Roll CW</button>
        
        <button onclick="sendCommand('reset')" style="grid-column: span 3;">Reset Camera</button>
    </div>
    
    <div class="instructions">
        <h3>Keyboard Controls:</h3>
        <ul>
            <li><strong>W</strong> - Move Forward</li>
            <li><strong>S</strong> - Move Backward</li>
            <li><strong>A</strong> - Move Left</li>
            <li><strong>D</strong> - Move Right</li>
            <li><strong>Q</strong> - Move Up</li>
            <li><strong>E</strong> - Move Down</li>
            <li><strong>Arrow Up</strong> - Look Up</li>
            <li><strong>Arrow Down</strong> - Look Down</li>
            <li><strong>Arrow Left</strong> - Rotate Left</li>
            <li><strong>Arrow Right</strong> - Rotate Right</li>
            <li><strong>Z</strong> - Roll Counterclockwise</li>
            <li><strong>X</strong> - Roll Clockwise</li>
            <li><strong>R</strong> - Reset Camera</li>
        </ul>
    </div>
    
    <script>
        // Get initial image
        updateImage();
        
        // Set up keyboard controls
        document.addEventListener('keydown', function(event) {{
            switch(event.key) {{
                case 'w': case 'W': sendCommand('move_forward'); break;
                case 's': case 'S': sendCommand('move_backward'); break;
                case 'a': case 'A': sendCommand('move_left'); break;
                case 'd': case 'D': sendCommand('move_right'); break;
                case 'q': case 'Q': sendCommand('move_up'); break;
                case 'e': case 'E': sendCommand('move_down'); break;
                case 'ArrowUp': sendCommand('rotate_up'); break;
                case 'ArrowDown': sendCommand('rotate_down'); break;
                case 'ArrowLeft': sendCommand('rotate_left'); break;
                case 'ArrowRight': sendCommand('rotate_right'); break;
                case 'z': case 'Z': sendCommand('rotate_counterclockwise'); break;
                case 'x': case 'X': sendCommand('rotate_clockwise'); break;
                case 'r': case 'R': sendCommand('reset'); break;
            }}
        }});
        
        function sendCommand(command) {{
            fetch('/control', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify({{ command: command }}),
            }})
            .then(response => response.json())
            .then(data => {{
                document.getElementById('camera-image').src = 'data:image/png;base64,' + data.image;
            }});
        }}
        
        function updateImage() {{
            fetch('/camera_image')
            .then(response => response.json())
            .then(data => {{
                document.getElementById('camera-image').src = 'data:image/png;base64,' + data.image;
            }});
        }}
    </script>
</body>
</html>
        """)


def main(args):
    """Main function"""
    global env, camera_id, robot_base_name, camera_name, image_resolution, temp_dir
    global move_speed, rotate_speed, mujoco_wxyz
    
    # Set global variables from command line arguments
    robot_base_name = args.robot_base_name
    camera_name = args.camera_name
    image_resolution = args.resolution
    move_speed = args.move_speed
    rotate_speed = args.rotate_speed
    mujoco_wxyz = args.mujoco_wxyz
    temp_dir = args.temp_dir
    
    # Create temp directory
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    task = task_suite.get_task(args.task_id)
    
    env, _ = get_libero_env(task, resolution=image_resolution)
    env.reset()
    
    # Get camera ID
    camera_id = env.sim.model.camera_name2id(camera_name)
    
    # Initialize camera
    initialize_camera()
    
    # Create HTML template
    create_html_template()
    
    # Start Flask server
    print(f"\nStarting camera control interface on http://{args.host}:{args.port}")
    print(f"To access from VSCode, you may need to setup port forwarding:")
    print(f"1. Click on 'Ports' tab in the terminal panel")
    print(f"2. Click 'Forward a Port' and enter {args.port}")
    print(f"3. Then access http://localhost:{args.port} in your local browser\n")
    
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Interactive camera control for LIBERO environments")
    
    # LIBERO environment arguments
    parser.add_argument(
        "--libero_task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="LIBERO task suite. Example: libero_spatial",
        default="libero_spatial",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        help="ID of the task to load (default: 0)",
        default=0,
    )
    
    # Camera settings
    parser.add_argument(
        "--camera_name",
        type=str,
        help="Name of the camera to control (default: agentview)",
        default="agentview",
    )
    parser.add_argument(
        "--robot_base_name",
        type=str,
        help="Name of the robot base link (default: robot0_link0)",
        default="robot0_link0",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="Image resolution (width and height) (default: 512)",
        default=512,
    )
    parser.add_argument(
        "--mujoco_wxyz",
        type=bool,
        help="Whether MuJoCo uses wxyz quaternion format (default: True)",
        default=True,
    )
    
    # Control settings
    parser.add_argument(
        "--move_speed",
        type=float,
        help="Movement speed for camera position (default: 0.05)",
        default=0.05,
    )
    parser.add_argument(
        "--rotate_speed",
        type=float,
        help="Rotation speed in degrees (default: 5.0)",
        default=5.0,
    )
    
    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        help="Host address for the web server (default: 0.0.0.0, accessible from any network interface)",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port for the web server (default: 5000)",
        default=5000,
    )
    
    # Other settings
    parser.add_argument(
        "--temp_dir",
        type=str,
        help="Directory to store temporary files (default: temp_camera_views)",
        default="temp_camera_views",
    )
    
    args = parser.parse_args()

    # Start the program
    main(args) 