import base64
import json
import logging
import os
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
    
    def __init__(self, golvo_settings=GALVO_SETTINGS_DEFAULT):
        self.calibration_file = 'data/configs/calibration.json'
        self.is_cutting = False
        self.template_mask = None
        self.settings = {}  # Add settings dictionary
        self.bg_lower = np.array([0, 0, 0])  # Default values
        self.bg_upper = np.array([180, 255, 255])  # Default values
        self.golvo_settings = golvo_settings
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
                logging.info("Galvo connected successfully")                    
            except Exception as e:
                attempt += 1
                logging.warning(f"Failed to connect to galvo (attempt {attempt}/{max_attempts}): {e}")
                self.galvo_connection = False
                self.sender = None
                time.sleep(2)
                
    def set_background_range(self, bg_analysis):
        """Set background color range from analysis"""
        self.bg_lower = np.array(bg_analysis['range']['lower'])
        self.bg_upper = np.array(bg_analysis['range']['upper'])
            
        
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
    
    # END galvo setting       
    def adjust_galvo_offset(self, dx, dy):
        with self.lock:
            self.galvo_offset_x += dx
            self.galvo_offset_y += dy
            hex_x, hex_y = self.pixel_to_galvo_coordinates(self.galvo_offset_x, self.galvo_offset_y)
            self.sender.set_xy(hex_x, hex_y)
            print(f"Galvo offset: X={self.galvo_offset_x}, Y={self.galvo_offset_y}") 
            
    def toggle_calibration_mode(self):
        self.calibration_mode = not self.calibration_mode
        print(f"Calibration mode: {'ON' if self.calibration_mode else 'OFF'}")
        if not self.calibration_mode:
            # Exiting calibration mode, save the current offsets
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
                    
    def cut_detected_patterns(self, contours, pattern_id):
        """Execute cutting sequence for detected patterns"""
        if not self.sender:
            raise RuntimeError("Laser sender not initialized")
        
        # Sort contours by position (left to right, top to bottom)
        sorted_contours = sorted(contours, 
                            key=lambda c: (c[0][0][1], c[0][0][0]))
        
        for contour_idx, contour in enumerate(sorted_contours):
            try:
                # Move to starting position without cutting
                start_point = contour[0][0]
                self.sender.set_xy(start_point[0], start_point[1])
                time.sleep(0.1)  # Allow time for movement
                
                
                # Prepare cutting parameters
                params = {
                    'travel_speed': self.golvo_settings['travel_speed'],
                    'frequency': self.golvo_settings['frequency'],
                    'power': self.golvo_settings['power'],
                    'cut_speed': self.golvo_settings['cut_speed'],
                    'laser_on_delay': self.golvo_settings['laser_on_delay'],
                    'laser_off_delay': self.golvo_settings['laser_off_delay'],
                    'polygon_delay': self.golvo_settings['polygon_delay'],
                }
                
                # Execute cutting sequence
                def tick(cmds, loop_index):
                    cmds.clear()
                    cmds.set_mark_settings(**params)
                    
                    # Convert contour points to cutting coordinates
                    for point in contour:
                        x_hex, y_hex = self.pixel_to_galvo_coordinates(point[0][0], point[0][1])
                        
                        cmds.light(x_hex, y_hex, light=True, jump_delay=100)
                
                # Create and execute cutting job
                job = self.sender.job(tick=tick)
                job.execute(1)
                
                # Move to safe position after cutting
                self.sender.set_xy(0x8000, 0x8000)
                
            except Exception as e:
                raise Exception(f"Error cutting pattern {contour_idx}: {str(e)}")
                
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