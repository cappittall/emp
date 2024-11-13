import base64
import json
import logging
import os
import statistics
import threading
import time
import cv2
import sys  
import numpy as np

DEBUG = os.path.exists('data/debug')
#sys.path.append('/home/yordam/balor') 
sys.path.append('../balor')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'balor')))
sys.path.append('/home/yordam/balor')
try:
    from balor.sender import Sender # type: ignore 
    from balor.command_list import CommandList # type: ignore
    print('Balor object loaded...')
except Exception as e:
    print(f'Error loading Balor object: {e}')
    print(f'Current sys.path: {sys.path}')
    print(f'Current working directory: {os.getcwd()}')
    sys.exit(1)
    

GALVO_SETTINGS_DEFAULT = {
                    'travel_speed': 5000,
                    'frequency': 100,
                    'power': 50,
                    'cut_speed': 5000,
                    'laser_on_delay': 1,
                    'laser_off_delay': 1,
                    'polygon_delay': 50
                }

# import torch
class PatchCutter:
    
    def __init__(self, galvo_settings=GALVO_SETTINGS_DEFAULT):
        self.calibration_file = 'data/configs/calibration.json'
        self.is_cutting = False
        self.template_mask = None
        self.settings = {}  # Add settings dictionary
        self.bg_lower = np.array([0, 0, 0])  # Default values
        self.bg_upper = np.array([180, 255, 255])  # Default values
        self.galvo_settings = galvo_settings
        self.calibration_mode = False
        self.load_calibration()
        
        self.total_hex_distance = 51391 
        self.total_cm_distance = 16.3
        self.cm_per_hex_step = self.total_cm_distance / self.total_hex_distance
        self.hex_steps_per_cm = self.total_hex_distance / self.total_cm_distance
        
        self.galvo_connection = False
        self.galvo_control_thread = threading.Thread(target=self.connect_galvo_control)
        self.galvo_control_thread.daemon = True
        self.galvo_control_thread.start()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        self.boundary_walking_event = threading.Event()
        self.boundary_walking_thread = None

        # Initialize calibration variables (replacement ArUco calibration)
        self.calibration_visualization = None  
        self.calibration_contour = None        
        # Step size for adjusting pixel_cm_ratio
        self.pixel_cm_ratio_step = 1.0
                         
    def connect_galvo_control(self):
        max_attempts = 5
        attempt = 0
        while attempt < max_attempts and not self.galvo_connection:
            try:                
                self.sender = Sender()
                cor_table_data = open("data/configs/jetsonCalibrationdeneme1.cor", 'rb').read()
                if hasattr(self.sender, 'set_cor_table'):
                    self.sender.set_cor_table(cor_table_data)
                elif hasattr(self.sender, 'cor_table'):
                    self.sender.cor_table = cor_table_data
                self.sender.open(mock=DEBUG)
                self.galvo_connection = True
                self.send_to_top_left()
                logging.info("Galvo connected successfully")                    
            except Exception as e:
                attempt += 1
                logging.warning(f"Failed to connect to galvo (attempt {attempt}/{max_attempts}): {e}")
                self.galvo_connection = False
                self.sender = None
                time.sleep(2)
                
                
    def send_to_top_left(self):
        galvo_offset_x_hex, galvo_offset_y_hex = self.pixel_to_galvo_coordinates(self.galvo_offset_x, self.galvo_offset_y)
        self.sender.set_xy(galvo_offset_x_hex, galvo_offset_y_hex)
        
    def set_background_range(self, bg_analysis):
        """Set background color range from analysis"""
        self.bg_lower = np.array(bg_analysis['range']['lower'])
        self.bg_upper = np.array(bg_analysis['range']['upper'])
            
    def start_walk_galvo_boundary(self):
        if self.sender and (self.boundary_walking_thread is None or not self.boundary_walking_thread.is_alive()):
            self.boundary_walking_event.set()
            self.boundary_walking_thread = threading.Thread(target=self._walk_galvo_boundary)
            self.boundary_walking_thread.start()

    def stop_walk_galvo_boundary(self):
        self.boundary_walking_event.clear()
        if self.boundary_walking_thread and self.boundary_walking_thread.is_alive():
            self.boundary_walking_thread.join()
            self.boundary_walking_thread = None

    def _walk_galvo_boundary(self):
        while self.boundary_walking_event.is_set():
            self.sender.set_xy(0, 0)
            time.sleep(0.01)
            self.sender.set_xy(65535, 0)
            time.sleep(0.01)
            self.sender.set_xy(65535, 65535)
            time.sleep(0.01)
            self.sender.set_xy(0, 65535)
            time.sleep(0.01)            

    def analyze_background(self, bg_sample):
        """Enhanced background analysis for various colors"""
        hsv_bg = cv2.cvtColor(bg_sample, cv2.COLOR_RGB2HSV)
        
        # Calculate mean and std with more tolerance
        mean_color = np.mean(hsv_bg, axis=(0,1))
        std_color = np.std(hsv_bg, axis=(0,1))
        
        # Increased tolerance for varied backgrounds
        tolerance = 2.5
        
        color_range = {
            'lower': np.maximum(0, mean_color - tolerance * std_color),
            'upper': np.minimum([180, 255, 255], mean_color + tolerance * std_color)
        }
        
        return {
            'mean': mean_color.tolist(),
            'std': std_color.tolist(),
            'range': {
                'lower': color_range['lower'].tolist(),
                'upper': color_range['upper'].tolist()
            }
        }

    def extract_contour(self, patch_region, bg_analysis):
        """Dense point contour extraction with high precision"""
        hsv = cv2.cvtColor(patch_region, cv2.COLOR_RGB2HSV)
        
        # Enhanced edge detection using Canny
        gray = cv2.cvtColor(patch_region, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Combine color-based mask with edge detection
        bg_mean = np.array(bg_analysis['mean'])
        bg_std = np.array(bg_analysis['std'])
        lower_bound = np.clip(bg_mean - bg_std * 1.5, 0, 255)
        upper_bound = np.clip(bg_mean + bg_std * 1.5, 0, 255)
        
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        color_mask = cv2.bitwise_not(color_mask)
        
        # Combine masks
        combined_mask = cv2.bitwise_and(color_mask, edges)
        
        # Dense contour detection
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            # Generate dense points along the contour
            dense_contour = self.generate_dense_points(main_contour)
            
            return {
                'points': dense_contour.tolist(),
                'area': cv2.contourArea(main_contour),
                'perimeter': cv2.arcLength(main_contour, True)
            }
        return None

    def generate_dense_points(self, contour, spacing=2):
        """Generate high-density points along the contour"""
        perimeter = cv2.arcLength(contour, True)
        num_points = int(perimeter / spacing)
        
        # Interpolate points along the contour
        t = np.linspace(0, 1, num_points)
        contour = contour.squeeze()
        
        # Ensure closed contour
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        
        # Generate points using cubic interpolation
        x = np.interp(t * (len(contour)-1), range(len(contour)), contour[:, 0])
        y = np.interp(t * (len(contour)-1), range(len(contour)), contour[:, 1])
        
        dense_points = np.column_stack((x, y))
        return dense_points.reshape((-1, 1, 2)).astype(np.int32)

    def save_pattern(self, pattern_data):
        """Save pattern with complete metadata and encoded RGB image"""
        pattern_dir = os.path.join('data', 'patterns')
        os.makedirs(pattern_dir, exist_ok=True)
        
        pattern_id = pattern_data.get('pattern_id', f"P{int(time.time())}")
        
        # Convert BGR to RGB before encoding
        if 'patch_image' in pattern_data:
            rgb_image = cv2.cvtColor(pattern_data['patch_image'], cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.png', rgb_image)
            pattern_data['patch_image_encoded'] = base64.b64encode(buffer).decode('utf-8')
        
        pattern = {
            'pattern_id': pattern_id,
            'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'background': pattern_data['background'],
            'patch': pattern_data['patch'],
            'patch_image_encoded': pattern_data.get('patch_image_encoded'),
            'cutting_params': self.settings
        }
        
        pattern_file = os.path.join(pattern_dir, f'{pattern_id}.json')
        with open(pattern_file, 'w') as f:
            json.dump(pattern, f, indent=4)
        
        return pattern_id    
                      
    def adjust_galvo_offset(self, dx, dy):
        with self.lock:
            self.galvo_offset_x += dx
            self.galvo_offset_y += dy
            hex_x, hex_y = self.pixel_to_galvo_coordinates(self.galvo_offset_x, self.galvo_offset_y)
            self.sender.set_xy(hex_x, hex_y)
            print(f"Galvo offset: X={self.galvo_offset_x}, Y={self.galvo_offset_y}") 
            
    def toggle_calibration_mode(self):
        self.calibration_mode = not self.calibration_mode
        if self.calibration_mode:
            print("Calibration mode started.")
            # Load the first contour for visualization
            self.load_calibration_contour()
        else:
            print("Calibration mode stopped.")
            # Save the final pixel_cm_ratio to disk
            self.save_calibration()
     
    def save_calibration(self):
        calibration_data = {
            "pixel_cm_ratio": self.pixel_cm_ratio,
            "galvo_offset_x": self.galvo_offset_x,
            "galvo_offset_y": self.galvo_offset_y
        }
        os.makedirs(os.path.dirname(self.calibration_file), exist_ok=True)
        with open(self.calibration_file, "w") as file:
            json.dump(calibration_data, file)
        print(f"Calibration saved: galvo_offset_x = {self.galvo_offset_x}, galvo_offset_y = {self.galvo_offset_y}")
        
    def calibrate_cm_pixel_ratio(self, frame):
        """Calibrate using ArUco markers with error handling"""
        try:
            # Create detector parameters and dictionary
            parameters = cv2.aruco.DetectorParameters_create()
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)
        except AttributeError:
            # Handle newer OpenCV versions
            parameters = cv2.aruco.DetectorParameters()
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

        try:
            # Create detector and detect markers
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, _, _ = detector.detectMarkers(frame)
            
            if corners:
                aruco_perimeter = cv2.arcLength(corners[0], True)
                self.pixel_cm_ratio = aruco_perimeter / 20  # Assuming 20cm marker size
                
                # Save calibration data
                calibration_data = {
                    "pixel_cm_ratio": self.pixel_cm_ratio,
                    "galvo_offset_x": self.galvo_offset_x, 
                    "galvo_offset_y": self.galvo_offset_y                   
                }
                
                with open("data/configs/calibration.json", "w") as f:
                    json.dump(calibration_data, f, indent=4)
                    
                return True
            else:
                raise ValueError("No ArUco markers detected in frame")
                
        except Exception as e:
            raise Exception(f"Calibration failed: {str(e)}")
    
    def load_calibration(self):
        try:
            with open(self.calibration_file, 'r') as file:
                calibration_data = json.load(file)
                self.pixel_cm_ratio = calibration_data.get('pixel_cm_ratio', 39.633)
                self.galvo_offset_x = calibration_data.get('galvo_offset_x', -35)
                self.galvo_offset_y = calibration_data.get('galvo_offset_y', 532)
                
            print(f'Loaded calibration: pixel_cm_ratio = {self.pixel_cm_ratio},  '
                  f'galvo_offset_x = {self.galvo_offset_x}, galvo_offset_y = {self.galvo_offset_y}')
        except FileNotFoundError:
            print(f"Calibration file {self.calibration_file} not found. Using default values.")
            self.galvo_offset_x = 40
            self.galvo_offset_y = 380
            self.pixel_cm_ratio = 39.633
            
    def clamp_galvo_coordinates(self, x, y):
        # Assuming the valid range is 0-65535 (16-bit)
        x = max(0, min(x, 65535))
        y = max(0, min(y, 65535))
        return x, y  
    
    def pixel_to_galvo_coordinates(self, x, y):
        # Convert pixel coordinates to cm using the fixed ratio
        cm_x = x / self.pixel_cm_ratio
        cm_y = y / self.pixel_cm_ratio

        # Convert cm to hex steps
        hex_x = round(cm_x * self.hex_steps_per_cm)
        hex_y = round(cm_y * self.hex_steps_per_cm)

        # Clamp the coordinates to valid range
        hex_x, hex_y = self.clamp_galvo_coordinates(hex_x, hex_y)

        # Convert to hexadecimal and ensure 4-digit representation
        hex_x_str = f"{hex_x:04X}"
        hex_y_str = f"{hex_y:04X}"

        return hex_x, hex_y
                    
    def calculate_column_width(self, pattern_data):
        """Calculate optimal column width based on pattern distribution"""
        if len(pattern_data) < 2:
            return 100  # default value for single pattern
            
        # Sort by x coordinate
        sorted_x = sorted(pattern_data, key=lambda p: p[0])
        
        # Calculate distances between adjacent patterns
        distances = []
        for i in range(len(sorted_x) - 1):
            dist = sorted_x[i + 1][0] - sorted_x[i][0]
            if dist > 10:  # Minimum threshold to avoid noise
                distances.append(dist)
        
        if not distances:
            return 100  # fallback value
            
        # Use median distance as column width
        column_width = statistics.median(distances)
        return column_width * 0.8  # Use 80% of median distance for reliable grouping

    def cut_detected_patterns(self, contours, pattern_id):
        """Execute cutting sequence for detected patterns in optimized order"""
        if not self.sender:
            raise RuntimeError("Laser sender not initialized")

        # Get centers and organize into columns
        pattern_data = []
        for idx, contour in enumerate(contours):
            # Calculate the centroid of the contour for better accuracy
            M = cv2.moments(contour)
            if M['m00'] != 0:
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
                pattern_data.append((x, y, idx))
            else:
                # Fallback to first point if contour is degenerate
                x = contour[0][0][0]
                y = contour[0][0][1]
                pattern_data.append((x, y, idx))

        # Calculate column width based on pattern spacing
        column_width = self.calculate_column_width(pattern_data)

        # Group patterns into columns
        columns = {}
        for x, y, idx in pattern_data:
            col_idx = int(x // column_width)
            if col_idx not in columns:
                columns[col_idx] = []
            columns[col_idx].append((x, y, idx))

        # Create an optimized list of indices to cut
        optimized_indices = []
        column_indices = sorted(columns.keys())
        for i, col_idx in enumerate(column_indices):
            column_patterns = columns[col_idx]
            # Alternate the direction of processing for each column
            if i % 2 == 0:
                # Top to bottom
                column_patterns.sort(key=lambda p: p[1])
            else:
                # Bottom to top
                column_patterns.sort(key=lambda p: p[1], reverse=True)
            optimized_indices.extend([p[2] for p in column_patterns])

        # Debug: Verify that all indices are included
        print(f"Total contours detected: {len(contours)}")
        print(f"Optimized indices: {optimized_indices}")

        # Send the laser to the initial offset position before starting
        self.send_to_top_left()
        time.sleep(0.1)

        # Cutting parameters
        params = {
            'travel_speed': self.galvo_settings['travel_speed'],
            'frequency': self.galvo_settings['frequency'],
            'power': self.galvo_settings['power'],
            'cut_speed': self.galvo_settings['cut_speed'],
            'laser_on_delay': self.galvo_settings['laser_on_delay'],
            'laser_off_delay': self.galvo_settings['laser_off_delay'],
            'polygon_delay': self.galvo_settings['polygon_delay'],
        }

        # Cut patterns in optimized order
        for idx in optimized_indices:
            contour = contours[idx]

            # Move to starting position of the contour
            start_point = contour[0][0]
            x, y = start_point

            # Apply galvo offsets
            x_off = x + self.galvo_offset_x
            y_off = y + self.galvo_offset_y

            # Convert to galvo coordinates
            x_hex, y_hex = self.pixel_to_galvo_coordinates(x_off, y_off)

            # Move to the starting position
            self.sender.set_xy(x_hex, y_hex)
            time.sleep(0.05)  # Small delay to allow movement

            # Prepare the cutting command sequence
            def tick(cmds, loop_index):
                cmds.clear()
                cmds.set_mark_settings(**params)

                # Begin cutting the contour
                for point in contour:
                    px, py = point[0]

                    # Apply galvo offsets
                    px_off = px + self.galvo_offset_x
                    py_off = py + self.galvo_offset_y

                    # Convert to galvo coordinates
                    px_hex, py_hex = self.pixel_to_galvo_coordinates(px_off, py_off)

                    # Light command for cutting
                    cmds.light(px_hex, py_hex, light=True)

            # Create and execute the cutting job
            job = self.sender.job(tick=tick)
            job.execute(1)
            time.sleep(0.05)  # Small delay between patterns

        # After all patterns are cut, move to a safe position
        self.send_to_top_left()
        time.sleep(0.1)
  
                
    def cleanup(self):
        """Cleanup resources safely"""
        self.is_running = False
        
        if hasattr(self, 'sender') and self.sender:
            try:
                if self.galvo_connection:
                    self.sender.close()
            except Exception as e:
                logging.warning(f"Error during sender cleanup: {e}")
            finally:
                self.sender = None
                self.galvo_connection = False
        
        cv2.destroyAllWindows()
        
        
        
""" 
 
    # END galvo setting 
    def increase_pixel_cm_ratio(self):
        if self.calibration_mode:
            self.pixel_cm_ratio += self.pixel_cm_ratio_step
            print(f"pixel_cm_ratio increased to {self.pixel_cm_ratio}")
            self.update_calibration_visualization()

    def decrease_pixel_cm_ratio(self):
        if self.calibration_mode and self.pixel_cm_ratio > self.pixel_cm_ratio_step:
            self.pixel_cm_ratio -= self.pixel_cm_ratio_step
            print(f"pixel_cm_ratio decreased to {self.pixel_cm_ratio}")
            self.update_calibration_visualization()
    
    # TODO send contours to this function
    def load_calibration_contour(self):
        # Assume you have a method to capture a frame and detect contours
        frame = self.capture_frame()
        contours = self.detect_contours(frame)
        if contours:
            self.calibration_contour = contours[0]
            self.update_calibration_visualization()
        else:
            print("No contours found for calibration visualization.")
                  
    def update_calibration_visualization(self):
        if self.calibration_contour is None:
            return

        # Capture a new frame for visualization
        frame = self.capture_frame()

        # Create a copy to draw on
        visualization = frame.copy()

        # Apply galvo offsets to the contour points
        adjusted_contour = []
        for point in self.calibration_contour:
            x, y = point[0]
            x_off = x + self.galvo_offset_x
            y_off = y + self.galvo_offset_y
            adjusted_contour.append([[x_off, y_off]])

        adjusted_contour = np.array(adjusted_contour, dtype=np.int32)

        # Draw the adjusted contour on the visualization image
        cv2.drawContours(visualization, [adjusted_contour], -1, (0, 255, 0), 2)

        # Save the visualization image for display
        self.calibration_visualization = visualization

        # Optionally, display the image here if you're not using a GUI framework
        # cv2.imshow("Calibration Visualization", self.calibration_visualization)
        # cv2.waitKey(1)
        
"""