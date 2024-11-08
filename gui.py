import base64
import json
import os
import re
import shutil
import signal
import sys 
import threading
import time
import tkinter as tk

from tkinter import ttk, filedialog, messagebox, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import cv2
from matplotlib import pyplot as plt
import numpy as np
from patch_cutter import PatchCutter

# 
PATCH_SETTINGS_DEFAULT = {
    'sam_confidence': 0.7,        # SAM2 confidence threshold
    'stability_score': 0.92,      # SAM2 stability score
    'min_patch_area': 25.0,       # Minimum patch area
    'travel_speed': 5000,
    'cut_speed': 5000,
    'laser_power': 50,
    'frequency': 100,
    'laser_on_delay': 1,
    'laser_off_delay': 1
}

class PatchCutterGUI:  
    def __init__(self, master, video_source=0):
        self.settings_file = 'data/configs/patch_settings.json'
        self.screen_size_file = 'data/configs/screen-size.txt'
        self.load_settings(self.settings_file)
        self.master = master
        self.master.title("Embroidery Patch Cutter")
        try:
            with open(self.screen_size_file, 'r') as f:
                screen_size = f.read().strip()
            # Validate screen size
            if not re.match(r'^\d+x\d+$', screen_size):
                raise ValueError("Invalid screen size format")
            width, height = map(int, screen_size.split('x'))
            if width <= 100 or height <= 100:
                raise ValueError("Window size too small")
        except (FileNotFoundError, ValueError):
            screen_size = "800x600"

        self.right_panel_width = 250   # Fixed width for right panel   
        self.master.geometry(screen_size)  
        self.master.minsize(800, 600)
            
        self._is_running = True
        self.calibration_mode = False

        self.selecting = False
        self.start_point = None
        self.selection_rect = None
        
        self.show_detected_pattern = False
        self.detected_contour = None
        self.threshold_timer = None
        
                            
        self.cutter = PatchCutter()  
        self.create_main_layout()
                
        os.makedirs('data', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)
               
        # Initialize camera in GUI only
        self.video_source = video_source
        self.init_camera()
        self.current_image_rgb = None
        
        # Start camera feed if available
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.update_camera_feed_cam()
            
        self.update_pattern_list()
            
    def create_main_layout(self):       
    # Create main container frame
        self.main_container = ttk.Frame(self.master)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel for camera feed with weight
        self.left_panel = ttk.Frame(self.main_container)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Camera Feed
        self.camera_frame = ttk.LabelFrame(self.left_panel, text="Camera Feed")
        self.camera_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.camera_canvas = tk.Canvas(self.camera_frame)
        self.camera_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right panel with fixed width
        self.right_panel = ttk.Frame(self.main_container, width=250)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        self.right_panel.pack_propagate(False)  # Prevent right panel from shrinking
        self.master.grid_columnconfigure(1, weight=1)
        
        # 1. Mask Creation - Horizontal button layout
        self.pattern_creation_frame = ttk.LabelFrame(self.right_panel, text="1. Patern Creation")
        self.pattern_creation_frame.pack(fill=tk.X, padx=5, pady=5)


        # Create a frame for horizontal button arrangement
        button_frame = ttk.Frame(self.pattern_creation_frame)
        button_frame.pack(fill=tk.X, padx=2, pady=2)


        # In create_main_layout method, modify the button frame section:
        ttk.Button(button_frame, text="Select", 
                command=self.start_selection).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Load Image", 
                command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Generate", 
                command=self.generate_masks).pack(side=tk.LEFT, padx=2)
        
        # 2. Pattern Display
        self.pattern_frame = ttk.LabelFrame(self.right_panel, text="2. Detected Pattern")
        self.pattern_frame.pack(fill=tk.X, padx=5, pady=5)
        self.pattern_canvas = tk.Canvas(self.pattern_frame, width=150, height=150)
        self.pattern_canvas.pack(fill=tk.X, padx=5, pady=5)


        # 3. Pattern Library
        self.pattern_library_frame = ttk.LabelFrame(self.right_panel, text="3. Pattern Library")
        self.pattern_library_frame.pack(fill=tk.X, padx=5, pady=5)
        
        
        # Pattern list with scrollbar
        list_frame = ttk.Frame(self.pattern_library_frame)
        list_frame.pack(fill=tk.X, padx=5, pady=2)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.pattern_list = tk.Listbox(list_frame, height=5, yscrollcommand=scrollbar.set)
        self.pattern_list.pack(fill=tk.X, side=tk.LEFT, expand=True)
        scrollbar.config(command=self.pattern_list.yview)
        
        # Add pattern management buttons
        self.manage_patterns()
        
        # Bind selection event
        self.pattern_list.bind('<<ListboxSelect>>', self.on_pattern_select)


        # 4. Live Detection
        """Create detection panel with threshold slider"""
        detection_frame = ttk.LabelFrame(self.right_panel, text="4. Live Detection")
        detection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add threshold slider
        self.threshold_var = tk.DoubleVar(value=0.8)
        threshold_label = ttk.Label(detection_frame, text="Detection Threshold:")
        threshold_label.pack(fill=tk.X, padx=5, pady=2)
        
        threshold_slider = ttk.Scale(
            detection_frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=self.on_threshold_change
        )
        threshold_slider.pack(fill=tk.X, padx=5, pady=2)
        
        # Add threshold value display
        self.threshold_value_label = ttk.Label(detection_frame, text="0.8")
        self.threshold_value_label.pack(fill=tk.X, padx=5, pady=2)
        
        # 5. Process Controls
        self.process_frame = ttk.LabelFrame(self.right_panel, text="5. Process Controls")
        self.process_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create horizontal button frame with status indicator
        control_frame = ttk.Frame(self.process_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Connection status indicator
        self.connection_indicator = tk.Canvas(control_frame, width=20, height=20)
        self.connection_indicator.pack(side=tk.LEFT, padx=5)
        
        # Add buttons horizontally
        self.cut_button = ttk.Button(control_frame, text="Start Cutting", 
                                    command=self.start_cutting_process)
        self.cut_button.pack(side=tk.LEFT, padx=5)
        
        self.aruco_calibrate_button = ttk.Button(control_frame, text="ArUco Calibrate", 
                                                command=self.aruco_calibrate)
        self.aruco_calibrate_button.pack(side=tk.LEFT, padx=5)
        
        # Start connection status monitoring
        self.update_connection_status()

        # Exit button at bottom
        self.exit_button = ttk.Button(self.right_panel, text="Exit", 
                                    command=self.exit_application)
        self.exit_button.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=10)
        
        # Status text area
        self.status_frame = ttk.LabelFrame(self.right_panel, text="Status")
        self.status_frame.pack(fill=tk.X, padx=5, pady=2, expand=False)
        self.status_text = tk.Text(self.status_frame, height=4, wrap=tk.WORD, width=30)
        self.status_text.pack(fill=tk.X, padx=5, pady=2, expand=False)
        self.status_text.config(state='disabled')  
        
        
        # Add key bindings
        self.master.bind('<c>', self.toggle_offset_calibration)
        self.master.bind('<w>', lambda event: self.adjust_galvo_offset(0, -1))
        self.master.bind('<s>', lambda event: self.adjust_galvo_offset(0, 1))
        self.master.bind('<a>', lambda event: self.adjust_galvo_offset(-1, 0))
        self.master.bind('<d>', lambda event: self.adjust_galvo_offset(1, 0))
        self.master.bind('<r>', self.reset_galvo_offset)   
        
    ### Load pattern image
    def on_camera_frame_resize(self, event):
        # Get new frame dimensions
        frame_width = event.width
        frame_height = event.height
        
        if hasattr(self, 'original_width') and hasattr(self, 'original_height'):
            # Calculate new scaling factors
            self.scale_factor = min(frame_width/self.original_width, 
                                frame_height/self.original_height)
            
            # Update display dimensions
            self.display_width = int(self.original_width * self.scale_factor)
            self.display_height = int(self.original_height * self.scale_factor)
            
            # Update padding for centering
            self.pad_x = max(0, (frame_width - self.display_width) // 2)
            self.pad_y = max(0, (frame_height - self.display_height) // 2)
            
            # Configure canvas size
            self.camera_canvas.config(width=frame_width, height=frame_height)
            self.update_window_size()
            
    def update_window_size(self):
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        self.update_status(f"Window size: {width}x{height}")
        if int(width) >100 and int(height) >100:
            with open(self.screen_size_file, 'w') as f:
                f.write(f"{width}x{height}")

    def load_image(self):
        """Load image file for pattern creation"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            # Load and process the image
            image = cv2.imread(file_path)
            if image is not None:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Get background sample from corner
                bg_sample = image_rgb[0:10, 0:10]
                bg_analysis = self.cutter.analyze_background(bg_sample)
                self.cutter.set_background_range(bg_analysis)
                
                # Generate pattern ID
                pattern_id = f"P{int(time.time())}"
                
                # Create pattern data
                pattern_data = {
                    'pattern_id': pattern_id,
                    'background': bg_analysis,
                    'patch': self.cutter.extract_contour(image_rgb, bg_analysis),
                    'patch_image': image_rgb.copy()
                }
                
                # Display and save pattern
                self.display_pattern(pattern_data)
                self.cutter.save_pattern(pattern_data)
                self.update_pattern_list()
                self.update_status(f"Pattern {pattern_id} created from image")


    ### selection drag - drop for background  #########################################  
    def update_connection_status(self):
        """Update the connection indicator color based on galvo status"""
        if self.cutter.galvo_connection:
            self.connection_indicator.create_oval(2, 2, 18, 18, fill='green', outline='darkgreen')
            self.cut_button.config(state='normal')
        else:
            self.connection_indicator.create_oval(2, 2, 18, 18, fill='red', outline='darkred')
            self.cut_button.config(state='disabled')
        
        # Schedule next update
        self.master.after(1000, self.update_connection_status)
                   
    def start_selection(self):
        """Enhanced selection start with preview update"""
        self.selecting = True
        self.selection_type = "background"
        self.camera_canvas.bind('<Button-1>', self.on_selection_start)
        self.camera_canvas.bind('<B1-Motion>', self.on_selection_drag)
        self.camera_canvas.bind('<ButtonRelease-1>', self.on_selection_end)
        self.update_status("Select background area")

    def on_selection_start(self, event):
        if self.selecting:
            self.start_point = (event.x, event.y)
            self.selection_rect = None


    def on_selection_drag(self, event):
        if self.selecting and self.start_point:
            if self.selection_rect:
                self.camera_canvas.delete(self.selection_rect)
            self.selection_rect = self.camera_canvas.create_rectangle(
                self.start_point[0], self.start_point[1],
                event.x, event.y,
                outline='red', width=4
            )
                
    def on_selection_end(self, event):
        if self.selecting and self.start_point:
            x1, y1, x2, y2 = self.get_scaled_coordinates(event)
            
            bg_sample = self.original_frame[y1:y1+10, x1:x1+10]
            patch_region = self.original_frame[y1:y2, x1:x2]
            
            bg_analysis = self.cutter.analyze_background(bg_sample)
            self.cutter.set_background_range(bg_analysis)
            
            # Generate pattern_id first
            pattern_id = f"P{int(time.time())}"
            
            pattern_data = {
                'pattern_id': pattern_id,  # Add pattern_id here
                'background': bg_analysis,
                'patch': self.cutter.extract_contour(patch_region, bg_analysis),
                'patch_image': patch_region.copy()
            }
            
            # Now display_pattern will have access to pattern_id
            self.display_pattern(pattern_data)
            
            contour_points = np.array(pattern_data['patch']['points'])
            contour_points[:,:,0] += x1
            contour_points[:,:,1] += y1
            
            self.detected_contour = (contour_points * self.scale_factor).astype(np.int32)
            self.show_detected_pattern = True
            
            self.cutter.save_pattern(pattern_data)
            self.camera_canvas.delete(self.selection_rect)
            self.selecting = False
            self.update_status(f"Pattern {pattern_id} created and saved")
            self.update_pattern_list()
  
    def create_pattern(self, bg_sample, patch_region):
        # Extract pattern characteristics
        bg_color = self.analyze_background(bg_sample)
        contour = self.extract_contour(patch_region)
        dimensions = self.get_dimensions(contour)
        cutting_points = self.generate_cutting_points(contour)
        
        return {
            'background': bg_color,
            'patch': {
                'dimensions': dimensions,
                'contour': contour,
                'cutting_points': cutting_points
            }
        }


    def update_pattern_list(self):
        self.pattern_list.delete(0, tk.END)
        pattern_dir = os.path.join('data', 'patterns')
        
        if os.path.exists(pattern_dir):
            patterns = sorted([f for f in os.listdir(pattern_dir) if f.endswith('.json')])
            for pattern in patterns:
                self.pattern_list.insert(tk.END, pattern.replace('.json', ''))
        
             
    def on_threshold_changeX(self, event):
        """Handle threshold slider changes with enhanced status updates"""
        value = self.threshold_var.get()
        self.threshold_value_label.config(text=f"{value:.2f}")
        
        selection = self.pattern_list.curselection()
        if selection:
            pattern_id = self.pattern_list.get(selection[0])
            pattern_file = os.path.join('data', 'patterns', f'{pattern_id}.json')
            
            with open(pattern_file, 'r') as f:
                pattern_data = json.load(f)
            
            positions = self.detect_pattern_position(pattern_data)
            if positions:
                self.detected_contours = []
                for position in positions:
                    contour_points = np.array(pattern_data['patch']['points'])
                    contour_points = contour_points * position['scale']
                    contour_points[:,:,0] += position['x']
                    contour_points[:,:,1] += position['y']
                    self.detected_contours.append((contour_points * self.scale_factor).astype(np.int32))
                
                self.show_detected_pattern = True
                
                # Enhanced status updates
                avg_confidence = sum(p['confidence'] for p in positions) / len(positions)
                status_msg = f"Pattern: {pattern_id} | Threshold: {value:.2f} | Found: {len(positions)} | Avg Confidence: {avg_confidence:.2f}"
                self.update_status(status_msg)
            else:
                self.show_detected_pattern = False
                self.update_status(f"Pattern: {pattern_id} | Threshold: {value:.2f} | No matches found")

    def on_threshold_change(self, event):
        """Debounced threshold change handler"""
        # Cancel previous timer if exists
        if self.threshold_timer:
            self.master.after_cancel(self.threshold_timer)
        
        # Update displayed value immediately
        value = self.threshold_var.get()
        self.threshold_value_label.config(text=f"{value:.2f}")
        
        # Schedule new detection after delay
        self.threshold_timer = self.master.after(150, self.perform_detection)


    def on_threshold_change(self, event):
        """Debounced threshold change handler"""
        # Cancel previous timer if exists
        if self.threshold_timer:
            self.master.after_cancel(self.threshold_timer)
        
        # Update displayed value immediately
        value = self.threshold_var.get()
        self.threshold_value_label.config(text=f"{value:.2f}")
        
        # Schedule new detection after delay
        self.threshold_timer = self.master.after(150, self.perform_detection)

    def perform_detection(self):
        """Actual detection logic separated from slider event"""
        value = self.threshold_var.get()
        selection = self.pattern_list.curselection()
        
        if selection:
            pattern_id = self.pattern_list.get(selection[0])
            pattern_file = os.path.join('data', 'patterns', f'{pattern_id}.json')
            
            with open(pattern_file, 'r') as f:
                pattern_data = json.load(f)
            
            positions = self.detect_pattern_position(pattern_data)
            if positions:
                self.detected_contours = []
                for position in positions:
                    contour_points = np.array(pattern_data['patch']['points'])
                    contour_points = contour_points * position['scale']
                    contour_points[:,:,0] += position['x']
                    contour_points[:,:,1] += position['y']
                    self.detected_contours.append((contour_points * self.scale_factor).astype(np.int32))
                
                self.show_detected_pattern = True
                avg_confidence = sum(p['confidence'] for p in positions) / len(positions)
                status_msg = f"Pattern: {pattern_id} | Threshold: {value:.2f} | Found: {len(positions)} | Avg Confidence: {avg_confidence:.2f}"
                self.update_status(status_msg)
            else:
                self.show_detected_pattern = False
                self.update_status(f"Pattern: {pattern_id} | Threshold: {value:.2f} | No matches found")
                        
    def detect_pattern_position(self, pattern_data):
        """Pattern detection with dynamic threshold"""
        img_data = base64.b64decode(pattern_data['patch_image_encoded'])
        nparr = np.frombuffer(img_data, np.uint8)
        template = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        frame_gray = cv2.cvtColor(self.original_frame, cv2.COLOR_RGB2GRAY)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        frame_h, frame_w = frame_gray.shape[:2]
        template_h, template_w = template_gray.shape[:2]
        
        scales = np.linspace(0.5, 2.0, 20)
        matches = []
        threshold = self.threshold_var.get()  # Get current threshold value
        
        for scale in scales:
            scaled_w = int(template_w * scale)
            scaled_h = int(template_h * scale)
            
            if scaled_w >= frame_w or scaled_h >= frame_h:
                continue
                
            scaled_template = cv2.resize(template_gray, (scaled_w, scaled_h))
            result = cv2.matchTemplate(frame_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            
            locations = np.where(result >= threshold)
            for pt in zip(*locations[::-1]):
                matches.append({
                    'x': pt[0],
                    'y': pt[1],
                    'width': scaled_w,
                    'height': scaled_h,
                    'scale': scale,
                    'confidence': result[pt[1], pt[0]]
                })
        
        return self.filter_overlapping_matches(matches)


    def filter_overlapping_matches(self, matches, overlap_thresh=0.3):
        """Filter overlapping pattern detections"""
        if not matches:
            return []
            
        # Convert matches to boxes format for NMS
        boxes = np.array([[m['x'], m['y'], m['x'] + m['width'], m['y'] + m['height']] for m in matches])
        scores = np.array([m['confidence'] for m in matches])
        
        # Calculate area of boxes
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by confidence
        idxs = np.argsort(scores)[::-1]
        
        keep = []
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            
            if len(idxs) == 1:
                break
                
            # Calculate IoU with rest of boxes
            xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            overlap = (w * h) / areas[idxs[1:]]
            
            # Remove overlapping detections
            idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
        
        return [matches[i] for i in keep]
            
    def on_pattern_select(self, event):
        """Updated to handle multiple pattern instances with image display"""
        selection = self.pattern_list.curselection()
        if not selection:
            return
            
        pattern_id = self.pattern_list.get(selection[0])
        pattern_file = os.path.join('data', 'patterns', f'{pattern_id}.json')
        
        try:
            with open(pattern_file, 'r') as f:
                pattern_data = json.load(f)
            
            # Decode image data if it exists
            if 'patch_image_encoded' in pattern_data:
                img_data = base64.b64decode(pattern_data['patch_image_encoded'])
                nparr = np.frombuffer(img_data, np.uint8)
                pattern_data['patch_image'] = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                pattern_data['patch_image'] = cv2.cvtColor(pattern_data['patch_image'], cv2.COLOR_BGR2RGB)
            
            self.display_pattern(pattern_data)
            
            # Rest of your pattern detection code...
            positions = self.detect_pattern_position(pattern_data)
            if not positions:
                self.update_status(f"No patterns found: {pattern_id}")
                return
            
            self.detected_contours = []
            for position in positions:
                contour_points = np.array(pattern_data['patch']['points'])
                contour_points = contour_points * position['scale']
                contour_points[:,:,0] += position['x']
                contour_points[:,:,1] += position['y']
                self.detected_contours.append((contour_points * self.scale_factor).astype(np.int32))
            
            self.show_detected_pattern = True
            
            self.update_status(f"Located {len(positions)} instances of pattern: {pattern_id}")
            
        except (json.JSONDecodeError, KeyError) as e:
            self.update_status(f"Error loading pattern: {pattern_id}")
            self.pattern_list.delete(selection[0])


    def display_pattern(self, pattern_data):
        """Enhanced pattern display in the pattern preview window"""
        # Clear existing content
        self.pattern_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width = self.pattern_canvas.winfo_width()
        canvas_height = self.pattern_canvas.winfo_height()
        
        # Use patch_image directly from pattern_data
        patch_image = pattern_data.get('patch_image')
        
        if patch_image is not None:
            # Convert numpy array to PIL Image
            image = Image.fromarray(patch_image)
            
            # Resize to fit canvas while maintaining aspect ratio
            image = self.resize_image_to_canvas(image, canvas_width, canvas_height)
            photo = ImageTk.PhotoImage(image)
            
            # Calculate centering position
            x = (canvas_width - image.width) // 2
            y = (canvas_height - image.height) // 2
            
            # Display image
            self.pattern_canvas.create_image(x, y, image=photo, anchor=tk.NW)
            self.pattern_canvas._photo = photo  # Keep reference
            
            # Draw contour overlay if available
            if 'points' in pattern_data.get('patch', {}):
                contour_points = np.array(pattern_data['patch']['points'])
                # Scale contour points to match displayed image size
                scale_x = image.width / patch_image.shape[1]
                scale_y = image.height / patch_image.shape[0]
                scaled_points = contour_points * [scale_x, scale_y]
                scaled_points = scaled_points.reshape((-1, 2))
                
                # Draw contour as polygon
                points = [(x + int(p[0]), y + int(p[1])) for p in scaled_points]
                self.pattern_canvas.create_polygon(points, outline='green', width=2, fill='')


    def manage_patterns(self):
        # Pattern management buttons
        button_frame = ttk.Frame(self.pattern_library_frame)
        button_frame.pack(fill=tk.X, padx=2, pady=2)
        
        ttk.Button(button_frame, text="Rename", 
                command=self.rename_pattern).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Delete", 
                command=self.delete_pattern).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Export", 
                    command=self.export_pattern).pack(side=tk.LEFT, padx=2)


    def rename_pattern(self):
        selection = self.pattern_list.curselection()
        if selection:
            pattern_id = self.pattern_list.get(selection[0])
            new_name = tk.simpledialog.askstring("Rename Pattern", 
                                            "Enter new name:", 
                                            initialvalue=pattern_id)
            if new_name:
                self.rename_pattern_files(pattern_id, new_name)
                self.update_pattern_list()


    def delete_pattern(self):
        selection = self.pattern_list.curselection()
        if selection:
            pattern_id = self.pattern_list.get(selection[0])
            if messagebox.askyesno("Delete Pattern", 
                                f"Delete pattern {pattern_id}?"):
                self.delete_pattern_files(pattern_id)
                self.update_pattern_list()


    def export_pattern(self):
        selection = self.pattern_list.curselection()
        if selection:
            pattern_id = self.pattern_list.get(selection[0])
            export_dir = filedialog.askdirectory(title="Export Pattern")
            if export_dir:
                self.export_pattern_files(pattern_id, export_dir)
    def rename_pattern_files(self, old_id, new_id):
        pattern_dir = os.path.join('data', 'patterns')
        
        # Rename JSON file
        old_json = os.path.join(pattern_dir, f'{old_id}.json')
        new_json = os.path.join(pattern_dir, f'{new_id}.json')
        os.rename(old_json, new_json)
        
        # Rename reference image
        old_img = os.path.join(pattern_dir, f'{old_id}_reference.png')
        new_img = os.path.join(pattern_dir, f'{new_id}_reference.png')
        os.rename(old_img, new_img)
        
        # Update JSON content
        with open(new_json, 'r') as f:
            data = json.load(f)
        data['pattern_id'] = new_id
        with open(new_json, 'w') as f:
            json.dump(data, f, indent=4)
        
        self.update_status(f"Pattern renamed to {new_id}")


    def delete_pattern_files(self, pattern_id):
        pattern_dir = os.path.join('data', 'patterns')
        
        # Remove JSON file
        json_file = os.path.join(pattern_dir, f'{pattern_id}.json')
        if os.path.exists(json_file):
            os.remove(json_file)
        
        # Remove reference image
        img_file = os.path.join(pattern_dir, f'{pattern_id}_reference.png')
        if os.path.exists(img_file):
            os.remove(img_file)
        
        self.update_status(f"Pattern {pattern_id} deleted")


    def export_pattern_files(self, pattern_id, export_dir):
        pattern_dir = os.path.join('data', 'patterns')
        
        # Copy JSON file
        json_file = os.path.join(pattern_dir, f'{pattern_id}.json')
        shutil.copy2(json_file, export_dir)
        
        # Copy reference image
        img_file = os.path.join(pattern_dir, f'{pattern_id}_reference.png')
        shutil.copy2(img_file, export_dir)
        
        self.update_status(f"Pattern {pattern_id} exported to {export_dir}")  

    ### ARUCO cutter ############################################################
    def aruco_calibrate(self):
        """Handle ArUco calibration with status updates"""
        try:
            if hasattr(self, 'original_frame') and self.original_frame is not None:
                self.cutter.calibrate_cm_pixel_ratio(self.original_frame)
                self.update_status("ArUco calibration completed successfully")
            else:
                self.update_status("No image available for calibration")
        except Exception as e:
            self.update_status(f"Calibration failed: {str(e)}")
    
    def start_cutting_process(self):
        """Start cutting detected patterns sequentially"""
        if not self.cutter.galvo_connection:
            self.update_status("Galvo not connected")
            return
        
        if not hasattr(self, 'detected_contours') or not self.detected_contours:
            self.update_status("No patterns detected to cut")
            return
            
        # Get current pattern settings
        selection = self.pattern_list.curselection()
        if not selection:
            self.update_status("No pattern selected")
            return
            
        pattern_id = self.pattern_list.get(selection[0])
        
        try:
            # Start cutting process in a separate thread
            threading.Thread(target=self._execute_cutting, 
                            args=(self.detected_contours, pattern_id),
                            daemon=True).start()
        except Exception as e:
            self.update_status(f"Cutting failed: {str(e)}")

    def _execute_cutting(self, contours, pattern_id):
        """Handle the cutting execution process"""
        try:
            self.update_status(f"Starting to cut {len(contours)} patterns")
            self.cut_button.config(state='disabled')
            
            # Execute cutting sequence
            self.cutter.cut_detected_patterns(contours, pattern_id)
            
            self.update_status("Cutting completed successfully")
        except Exception as e:
            self.update_status(f"Cutting error: {str(e)}")
        finally:
            self.cut_button.config(state='normal')
            # Use correct cleanup method name
            self.cutter.cleanup()

            
    #### GUI Camera  ##########################################################
    def generate_masks(self):
        pass

    def resize_image_to_canvas(self, image, canvas_width, canvas_height):
        img_width, img_height = image.size
        scale = min(canvas_width/img_width, canvas_height/img_height)
        new_size = (int(img_width * scale), int(img_height * scale))
        # Use LANCZOS instead of deprecated ANTIALIAS
        return image.resize(new_size, Image.Resampling.LANCZOS)  


    def init_cameraX(self):
        # Load original image
        self.original_frame = cv2.imread('data/feed_img/p5.jpg')
        if self.original_frame is None:
            self.update_status("Test image not found")
            return False
        
        # Convert BGR to RGB immediately after loading
        self.original_frame = cv2.cvtColor(self.original_frame, cv2.COLOR_BGR2RGB)
        
        # Store original dimensions
        self.original_height, self.original_width = self.original_frame.shape[:2]
        
        # Set fixed canvas size
        self.canvas_width = 800
        self.canvas_height = 600
        self.camera_canvas.config(width=self.canvas_width, height=self.canvas_height)
        
        # Calculate scaling factors
        self.scale_factor = min(self.canvas_width/self.original_width, 
                            self.canvas_height/self.original_height)
        
        # Calculate actual display dimensions
        self.display_width = int(self.original_width * self.scale_factor)
        self.display_height = int(self.original_height * self.scale_factor)
        
        # Calculate padding for centering
        self.pad_x = (self.canvas_width - self.display_width) // 2
        self.pad_y = (self.canvas_height - self.display_height) // 2
        
        # Create display frame
        self.current_frame = cv2.resize(self.original_frame, 
                                    (self.display_width, self.display_height))
        return True
    
    def init_camera(self):
        # Try different camera indices
        for cam_index in range(2):  # Try camera 0 and 1
            self.cap = cv2.VideoCapture(cam_index)
            if self.cap.isOpened():
                # Camera successfully opened
                self.video_source = cam_index
                break
        
        if not self.cap.isOpened():
            self.update_status("No camera found. Using fallback mode.")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Lower resolution for better compatibility
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual dimensions (may differ from requested)
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set canvas size
        self.canvas_width = 800
        self.canvas_height = 600
        self.camera_canvas.config(width=self.canvas_width, height=self.canvas_height)
        
        # Calculate scaling factors
        self.scale_factor = min(self.canvas_width/self.original_width, 
                            self.canvas_height/self.original_height)
        
        # Calculate display dimensions
        self.display_width = int(self.original_width * self.scale_factor)
        self.display_height = int(self.original_height * self.scale_factor)
        
        # Calculate padding
        self.pad_x = (self.canvas_width - self.display_width) // 2
        self.pad_y = (self.canvas_height - self.display_height) // 2
        
        return True

    
 
    
    def show_frame_on_canvas(self, frame, canvas):
        if frame is None:
            return
        
        # Convert directly to PhotoImage since frame is already in RGB
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image)
        
        # Clear previous image
        canvas.delete("all")
        
        # Draw new image
        canvas.create_image(self.pad_x, self.pad_y, image=photo, anchor=tk.NW)
        canvas._photo = photo


    def get_scaled_coordinates(self, event):
        # Adjust for image position on canvas
        canvas_x1 = min(self.start_point[0], event.x) - self.pad_x
        canvas_y1 = min(self.start_point[1], event.y) - self.pad_y
        canvas_x2 = max(self.start_point[0], event.x) - self.pad_x
        canvas_y2 = max(self.start_point[1], event.y) - self.pad_y
        
        # Convert to original image coordinates
        x1 = int(canvas_x1 / self.scale_factor)
        y1 = int(canvas_y1 / self.scale_factor)
        x2 = int(canvas_x2 / self.scale_factor)
        y2 = int(canvas_y2 / self.scale_factor)
        
        # Ensure coordinates are within bounds
        x1 = max(0, min(x1, self.original_width))
        y1 = max(0, min(y1, self.original_height))
        x2 = max(0, min(x2, self.original_width))
        y2 = max(0, min(y2, self.original_height))
        
        return x1, y1, x2, y2
    
    def draw_calibration_target(self, canvas):
        """Draw a dart target style calibration marker at true (0,0)"""
        # Target dimensions relative to image scale
        radius = min(self.canvas_width, self.canvas_height) * 0.1
        
        # Position at true (0,0)
        target_x = self.pad_x
        target_y = self.pad_y
        

        
        # Draw quarter circles flowing northwest to southeast
        for i in range(3):
            r = radius - (i * radius/3)
            canvas.create_arc(
                target_x - r, target_y - r,
                target_x + r, target_y + r,
                start=271, extent=88,  # Rotated to flow NW to SE
                fill='',
                outline='green',
                width=2
            )
                # Draw crosshairs
        canvas.create_line(target_x, target_y, target_x + radius+10, target_y, fill='red', width=4)  # +x direction
        canvas.create_line(target_x, target_y, target_x, target_y + radius + 10, fill='red', width=2)  # -y direction
        
        # Draw center point at (0,0)
        point_size = 4
        canvas.create_oval(
            target_x - point_size, target_y - point_size,
            target_x + point_size, target_y + point_size,
            fill='red', outline='red'
        )


    def adjust_galvo_offset(self, dx, dy):
        if self.cutter and self.cutter.calibration_mode:
            self.cutter.adjust_galvo_offset(dx, dy)
            self.update_status(f"Galvo offset: X={self.cutter.galvo_offset_x}, Y={self.cutter.galvo_offset_y}")
    
    def reset_galvo_offset(self, event=None):
        if self.cutter and self.cutter.calibration_mode:
            self.cutter.galvo_offset_x = -45
            self.cutter.galvo_offset_y = 460
            self.update_status("Galvo offsets reset to initial values.")
            
    # Update the toggle_offset_calibration method
    def toggle_offset_calibration(self, event=None):
        self.calibration_mode = not self.calibration_mode
        status = "ON" if self.calibration_mode else "OFF"
        
        if self.calibration_mode:
            self.update_status("Calibration Mode ON\nAlign laser with red center point\nUse WASD to adjust position\nPress 'r' to reset\nPress 'c' to toggle")
        else:
            self.update_status("Calibration Mode OFF")
            
        if self.cutter:
            self.cutter.toggle_calibration_mode()
            
        
    # Modify the update_camera_feed method
    def update_camera_feed_img(self):
        if hasattr(self, 'current_frame'):
            display_frame = self.current_frame.copy()
            
            # Draw all detected patterns
            if self.show_detected_pattern and hasattr(self, 'detected_contours'):
                for contour in self.detected_contours:
                    cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
                    
            self.show_frame_on_canvas(display_frame, self.camera_canvas)
            
            # Draw calibration target when in calibration mode
            if self.calibration_mode:
                self.draw_calibration_target(self.camera_canvas)
                
        self.master.after(30, self.update_camera_feed_img)

   
    def update_camera_feed_cam(self):
        if not hasattr(self, 'cap') or not self.cap.isOpened():
            self.update_status("Camera not available")
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Store original frame for processing
            self.original_frame = frame.copy()
            
            # Resize frame for display using current dimensions
            self.current_frame = cv2.resize(frame, (self.display_width, self.display_height))
            
            # Draw detected patterns if any
            if self.show_detected_pattern and hasattr(self, 'detected_contours'):
                display_frame = self.current_frame.copy()
                for contour in self.detected_contours:
                    cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
                self.current_frame = display_frame
            
            # Convert to PhotoImage
            image = Image.fromarray(self.current_frame)
            self.photo = ImageTk.PhotoImage(image=image)
            
            # Update canvas with centered image
            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(self.pad_x, self.pad_y, 
                                        image=self.photo, anchor='nw')
            
            # Draw calibration target if needed
            if self.calibration_mode:
                self.draw_calibration_target(self.camera_canvas)
        
        # Schedule next update
        self.master.after(30, self.update_camera_feed_cam)

    
    def load_settings(self, setting_file ):
        # Load settings from file if cutter is not initialized
        self.settings = PATCH_SETTINGS_DEFAULT.copy()
        os.makedirs(os.path.dirname(setting_file), exist_ok=True)
        
        try:
            with open(setting_file, 'r') as f:
                loaded_settings = json.load(f)
                self.settings.update(loaded_settings)
        except (FileNotFoundError, json.JSONDecodeError):
            # Save default settings
            with open(setting_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
                
    # Update the update_status method to handle the text widget state
    def update_status(self, message):
        self.status_text.config(state='normal')
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, message)
        self.status_text.config(state='disabled')
    
    def exit_application(self):
        self._is_running = False
        # Allow time for threads to cleanup
        time.sleep(0.1)
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        # Clear all images and variables before destroying
        for widget in self.master.winfo_children():
            if hasattr(widget, '_photo'):
                widget._photo = None
                
        if self.cutter:
            self.cutter.cleanup()
        
        # Schedule destruction in main thread
        self.master.after(100, self._final_cleanup) 
        
    
    def _final_cleanup(self):
        self.master.quit()
        self.master.destroy()
            
if __name__ == "__main__":
    try: 
        root = tk.Tk()
        app = PatchCutterGUI(root, video_source=0)
        
        def signal_handler(sig, frame):
            app.exit_application()
        
        signal.signal(signal.SIGINT, signal_handler)
        root.mainloop()
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully close al windows  and release camera
        print("KeyboardInterrupt: Exiting...")
        app.exit_application()
    sys.exit(0)