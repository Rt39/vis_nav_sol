# 作者：Rt39
# 日期：2024-12-04
# 描述：绘制迷宫地图
# 邮箱：sunrainshr@qq.com
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

# ============ PATH CONFIGURATIONS ============
# IMAGES_FOLDER = "D:/rob/MID/simexploration_data/images"
ACTIONS_FILE = os.path.join('.', 'image_actions.json')
OUTPUT_FILE = os.path.join('.', 'maze_map.png')
TARGET_IMAGE = 1470         # 改动之后重新运行此文件

# ============ MOVEMENT AND ANGLE PARAMETERS ============
DEGREES_PER_ROTATION = 2.5    
DISTANCE_PER_MOVE = 1/7       
ANGLE_SNAP = 5              
WALL_DISTANCE = 0.5          
ARROW_SPACING = 2            

# ============ ALIGNMENT PARAMETERS ============
MIN_STRAIGHT_MOVES = 12       
MAX_ROTATION_ACTIONS = 5     
DENSE_AREA_THRESHOLD = 5     

class MazeMapper:
    def __init__(self):
        # Initialize basic attributes
        self.current_pos = [0.0, 0.0]  # Current position as list for mutability
        self.start_pos = (0.0, 0.0)    # Start position as tuple
        self.direction = 0             # Current angle in degrees
        
        # Path and walls tracking
        self.path = [(0.0, 0.0)]      # List of all positions
        self.path_directions = [0]     # List of directions at each point
        self.walls = set()            # Set of wall positions
        
        # Bounds tracking
        self.min_x = 0.0
        self.min_y = 0.0
        self.max_x = 0.0
        self.max_y = 0.0
        
        # Movement tracking
        self.movement_count = 0
        self.rotation_count = 0
        self.accumulated_angle = 0
        self.movement_start_pos = [0.0, 0.0]
        self.movement_start_direction = 0
        self.positions_in_sequence = [[0.0, 0.0]]
        
        # Turn detection
        self.turns_90 = set()
        self.turns_180 = set()
        self.detected_turns_90 = 0
        self.detected_turns_180 = 0
        
        # Image and action tracking
        self.image_positions = {}
        self.action_counts = defaultdict(int)
        
        # Density tracking
        self.path_density = defaultdict(int)
        self.grid_size = 2.0
        
        # Store the first position
        self.update_path_density((0.0, 0.0))

    def get_grid_cell(self, pos):
        """Convert position to grid cell coordinates"""
        x = int(pos[0] / self.grid_size)
        y = int(pos[1] / self.grid_size)
        return (x, y)

    def update_path_density(self, pos):
        """Update path density map"""
        cell = self.get_grid_cell(pos)
        self.path_density[cell] += 1
        
        # Update neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor = (cell[0] + dx, cell[1] + dy)
                self.path_density[neighbor] += 0.5

    def is_dense_area(self, pos):
        """Check if current position is in a dense area"""
        cell = self.get_grid_cell(pos)
        nearby_density = sum(
            self.path_density[neighbor]
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for neighbor in [(cell[0] + dx, cell[1] + dy)]
        )
        return nearby_density >= DENSE_AREA_THRESHOLD

    def check_and_align_sequence(self):
        """Check if current sequence should be aligned and apply alignment"""
        if ((self.movement_count >= MIN_STRAIGHT_MOVES and 
             self.rotation_count < MAX_ROTATION_ACTIONS) or 
            (self.is_dense_area(self.current_pos) and self.movement_count >= MIN_STRAIGHT_MOVES/2)):
            
            total_dx = self.current_pos[0] - self.movement_start_pos[0]
            total_dy = self.current_pos[1] - self.movement_start_pos[1]
            
            # More aggressive alignment in dense areas
            if self.is_dense_area(self.current_pos):
                if abs(abs(total_dx) - abs(total_dy)) < 0.5:
                    cell = self.get_grid_cell(self.current_pos)
                    horizontal_density = (
                        self.path_density[(cell[0]-1, cell[1])] +
                        self.path_density[(cell[0]+1, cell[1])]
                    )
                    vertical_density = (
                        self.path_density[(cell[0], cell[1]-1)] +
                        self.path_density[(cell[0], cell[1]+1)]
                    )
                    if horizontal_density > vertical_density:
                        total_dy = 0
                    else:
                        total_dx = 0
            
            # Apply alignment
            if abs(total_dx) > abs(total_dy):
                avg_y = sum(pos[1] for pos in self.positions_in_sequence) / len(self.positions_in_sequence)
                self.direction = 90 if total_dx > 0 else 270
                for pos in self.positions_in_sequence:
                    pos[1] = avg_y
            else:
                avg_x = sum(pos[0] for pos in self.positions_in_sequence) / len(self.positions_in_sequence)
                self.direction = 0 if total_dy > 0 else 180
                for pos in self.positions_in_sequence:
                    pos[0] = avg_x
            
            # Update path
            for i, pos in enumerate(self.positions_in_sequence):
                idx = len(self.path) - len(self.positions_in_sequence) + i
                if idx < len(self.path):
                    self.path[idx] = (pos[0], pos[1])

    def reset_movement_tracking(self):
        """Reset movement tracking variables"""
        self.movement_count = 0
        self.rotation_count = 0
        self.movement_start_pos = self.current_pos.copy()
        self.movement_start_direction = self.direction
        self.positions_in_sequence = [self.current_pos.copy()]

    def check_and_align_sequence(self):
        """Check if current sequence should be aligned and apply alignment if needed"""
        if (self.movement_count >= MIN_STRAIGHT_MOVES and 
            self.rotation_count < MAX_ROTATION_ACTIONS):
            # Calculate overall movement direction
            total_dx = self.current_pos[0] - self.movement_start_pos[0]
            total_dy = self.current_pos[1] - self.movement_start_pos[1]
            
            # Determine primary direction
            if abs(total_dx) > abs(total_dy):
                # Horizontal alignment
                avg_y = sum(pos[1] for pos in self.positions_in_sequence) / len(self.positions_in_sequence)
                self.direction = 90 if total_dx > 0 else 270
                # Align all positions in sequence horizontally
                for pos in self.positions_in_sequence:
                    pos[1] = avg_y
            else:
                # Vertical alignment
                avg_x = sum(pos[0] for pos in self.positions_in_sequence) / len(self.positions_in_sequence)
                self.direction = 0 if total_dy > 0 else 180
                # Align all positions in sequence vertically
                for pos in self.positions_in_sequence:
                    pos[0] = avg_x
            
            # Update path with aligned positions
            for i, pos in enumerate(self.positions_in_sequence):
                idx = len(self.path) - len(self.positions_in_sequence) + i
                if idx < len(self.path):
                    self.path[idx] = (pos[0], pos[1])

    def add_walls(self, pos, direction):
        angle_rad = np.radians(direction)
        wall_dx = WALL_DISTANCE * np.cos(angle_rad)
        wall_dy = -WALL_DISTANCE * np.sin(angle_rad)
        
        pos = (round(pos[0], 3), round(pos[1], 3))
        wall1 = (round(pos[0] - wall_dx, 3), round(pos[1] - wall_dy, 3))
        wall2 = (round(pos[0] + wall_dx, 3), round(pos[1] + wall_dy, 3))
        
        self.walls.add(wall1)
        self.walls.add(wall2)

    def update_bounds(self):
        self.min_x = min(self.min_x, self.current_pos[0] - WALL_DISTANCE)
        self.max_x = max(self.max_x, self.current_pos[0] + WALL_DISTANCE)
        self.min_y = min(self.min_y, self.current_pos[1] - WALL_DISTANCE)
        self.max_y = max(self.max_y, self.current_pos[1] + WALL_DISTANCE)

    def move(self, is_forward):
        """Handle forward/backward movement"""
        angle_rad = np.radians(self.direction)
        dx = DISTANCE_PER_MOVE * np.sin(angle_rad)
        dy = DISTANCE_PER_MOVE * np.cos(angle_rad)
        
        if not is_forward:
            dx, dy = -dx, -dy
        
        # Update position
        self.current_pos[0] += dx
        self.current_pos[1] += dy
        self.current_pos[0] = round(self.current_pos[0], 3)
        self.current_pos[1] = round(self.current_pos[1], 3)
        
        # Update movement tracking
        self.movement_count += 1
        self.positions_in_sequence.append(self.current_pos.copy())
        
        # Check and apply alignment if needed
        self.check_and_align_sequence()
        
        # Add to path and walls
        self.add_walls((self.current_pos[0], self.current_pos[1]), self.direction)
        self.path.append((self.current_pos[0], self.current_pos[1]))
        self.path_directions.append(self.direction if is_forward else (self.direction + 180) % 360)
        self.update_bounds()

    def process_turn(self, turn_direction):
        """Handle turn actions"""
        angle_change = -DEGREES_PER_ROTATION if turn_direction == "LEFT" else DEGREES_PER_ROTATION
        self.direction = (self.direction + angle_change) % 360
        self.rotation_count += 1
        
        # Check for actual turns
        self.accumulated_angle += angle_change
        if abs(self.accumulated_angle) >= 180 - ANGLE_SNAP:
            pos = (round(self.current_pos[0]), round(self.current_pos[1]))
            self.turns_180.add(pos)
            self.detected_turns_180 += 1
            self.accumulated_angle = 0
            self.reset_movement_tracking()
        elif abs(self.accumulated_angle) >= 90 - ANGLE_SNAP:
            pos = (round(self.current_pos[0]), round(self.current_pos[1]))
            self.turns_90.add(pos)
            self.detected_turns_90 += 1
            self.accumulated_angle = 0
            self.reset_movement_tracking()

    def process_single_action(self, action):
        action = action.replace("Action.", "").upper()
        self.action_counts[action] += 1
        actions = action.split("|")
        
        try:
            for action in actions:
                if action == "FORWARD":
                    self.move(is_forward=True)
                elif action == "BACKWARD":
                    self.move(is_forward=False)
                elif action == "LEFT":
                    self.process_turn("LEFT")
                elif action == "RIGHT":
                    self.process_turn("RIGHT")
                elif action in ["IDLE", "QUIT", "CHECKIN"]:
                    self.reset_movement_tracking()
                else:
                    print(f"Warning: Unknown action type '{action}' - ignoring")
        except Exception as e:
            print(f"Warning: Error processing action '{action}': {e}")

    def process_action(self, action_string, image_number):
        self.image_positions[image_number] = (self.current_pos[0], self.current_pos[1])
        actions = action_string.split("|")
        for action in actions:
            self.process_single_action(action.strip())

    def create_visualization(self, target_image):
        fig, ax = plt.subplots(figsize=(20, 20), facecolor='white')
        ax.set_facecolor('white')
        
        # Plot walls
        wall_x = [w[0] for w in self.walls]
        wall_y = [w[1] for w in self.walls]
        ax.plot(wall_x, wall_y, '.', color='black', markersize=1, alpha=0.3)
        
        # Plot path with arrows
        path_x = [p[0] for p in self.path]
        path_y = [p[1] for p in self.path]
        
        # Draw main path
        ax.plot(path_x, path_y, '-', color='blue', linewidth=2, label='Path')
        
        # Add direction arrows
        for i in range(0, len(self.path)-1, ARROW_SPACING):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i+1]
            
            dx = x2 - x1
            dy = y2 - y1
            
            if abs(dx) > 0.001 or abs(dy) > 0.001:
                arrow = FancyArrowPatch(
                    (x1, y1),
                    (x2, y2),
                    color='blue',
                    arrowstyle='-|>',
                    mutation_scale=15,
                    linewidth=2,
                    alpha=0.7
                )
                ax.add_patch(arrow)
        
        # Plot turns
        for turn in self.turns_90:
            ax.plot(turn[0], turn[1], 'ro', markersize=8)
        for turn in self.turns_180:
            ax.plot(turn[0], turn[1], 'mo', markersize=8)
        
        # Plot start and current positions
        ax.plot(self.start_pos[0], self.start_pos[1], 'go', markersize=10, label='Start Position')
        if target_image in self.image_positions:
            pos = self.image_positions[target_image]
            ax.plot(pos[0], pos[1], 'yo', markersize=10, 
                   label=f'Current Position (Image {target_image})')
        
        ax.set_aspect('equal')
        padding = 2
        ax.set_xlim(self.min_x - padding, self.max_x + padding)
        ax.set_ylim(self.min_y - padding, self.max_y + padding)
        
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.axis('off')
        
        return fig

def process_maze_data():
    if not os.path.exists(ACTIONS_FILE):
        raise FileNotFoundError(f"Actions file not found at: {ACTIONS_FILE}")
    
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading actions from: {ACTIONS_FILE}")
    with open(ACTIONS_FILE, 'r') as f:
        action_data = list(json.load(f).values())
    
    print(f"Loaded {len(action_data)} actions")
    print(f"Sample first entry: {action_data[0]}")  # Debug print
    
    mapper = MazeMapper()
    
    print("Processing actions...")
    total_actions = len(action_data)
    
    try:
        for i, entry in enumerate(action_data):
            print(f"Processing entry {i}: {entry}")  # Debug print
            image_num = int(entry["image"].split('_')[1])
            action_str = entry["action"] if isinstance(entry["action"], str) else '|'.join(entry["action"])
            print(f"Processed to: image_num={image_num}, action_str={action_str}")  # Debug print
            mapper.process_action(action_str, image_num)
            
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{total_actions} actions")
    
        print("\nGenerating maze map...")
        fig = mapper.create_visualization(TARGET_IMAGE)
        
        print(f"Saving figure to {OUTPUT_FILE}")
        plt.savefig(OUTPUT_FILE, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        
        print("\nMaze Exploration Statistics:")
        print(f"Maze map saved as: {OUTPUT_FILE}")
        print(f"90° turns detected: {mapper.detected_turns_90}")
        print(f"180° turns detected: {mapper.detected_turns_180}")
        print(f"Maze dimensions: {mapper.max_x - mapper.min_x:.1f} x {mapper.max_y - mapper.min_y:.1f}")
        print("\nAction Summary:")
        for action, count in sorted(mapper.action_counts.items()):
            print(f"{action}: {count} times ({count/total_actions*100:.1f}%)")
    
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Q 954339022_.json Q
    try:
        process_maze_data()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
    # T 1_5312855651_.json EL Picture
        traceback.print_exc()
