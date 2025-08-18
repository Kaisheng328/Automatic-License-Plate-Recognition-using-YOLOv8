import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageDraw
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import re
import os
import json
from datetime import datetime
import threading
import time

# --- Core Application Logic ---

# Define the list of registered license plates (your "database")
REGISTERED_PLATES = ["VBG752", "KDH5527", "PLP1701", "VGT190", "VJJ252", "WDV5985"]

# Load the pre-trained YOLOv8 model for license plate detection
# try:
#     model = YOLO('best.pt')
# except Exception as e:
#     messagebox.showerror("Model Error", f"Failed to load YOLOv8 model. Make sure 'best.pt' is in the correct folder.\nError: {e}")
#     exit()

# Initialize the EasyOCR reader for English
try:
    reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    messagebox.showerror("EasyOCR Error", f"Failed to initialize EasyOCR.\nError: {e}")
    exit()

def preprocess_image(image, preprocessing_options):
    """Apply various preprocessing techniques to enhance image quality for better detection."""
    processed = image.copy()
    
    # 1. Noise Reduction
    if preprocessing_options.get('noise_reduction', False):
        processed = cv2.bilateralFilter(processed, 9, 75, 75)
    
    # 2. Contrast Enhancement using CLAHE
    if preprocessing_options.get('contrast_enhancement', False):
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        processed = cv2.merge([l, a, b])
        processed = cv2.cvtColor(processed, cv2.COLOR_LAB2BGR)
    
    # 3. Brightness and Contrast Adjustment
    if preprocessing_options.get('brightness_contrast', False):
        processed = auto_adjust_brightness_contrast(processed)
    
    # 4. Sharpening
    if preprocessing_options.get('sharpening', False):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        processed = cv2.filter2D(processed, -1, kernel)
    
    # 5. Gamma Correction
    if preprocessing_options.get('gamma_correction', False):
        processed = adjust_gamma(processed, gamma=1.2)
    
    # 6. Edge Enhancement
    if preprocessing_options.get('edge_enhancement', False):
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        processed = cv2.addWeighted(processed, 0.8, edges_colored, 0.2, 0)
    
    return processed

def auto_adjust_brightness_contrast(image, clip_hist_percent=1):
    """Automatically adjust brightness and contrast using histogram analysis."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    accumulator = np.cumsum(hist)
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0
    
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def adjust_gamma(image, gamma=1.0):
    """Adjust gamma for image brightness correction."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_plate_crop(plate_crop):
    """Specific preprocessing for the cropped license plate to improve OCR accuracy."""
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def clean_plate_text(text):
    """Cleans the recognized text by removing special characters and spaces."""
    return re.sub(r'[^A-Z0-9]', '', text).upper()

def process_image_batch(self, image_paths, preprocessing_options, progress_callback=None):
    """Process multiple images in batch mode."""
    results = []
    total = len(image_paths)
    
    for i, image_path in enumerate(image_paths):
        try:
            model_name = self.active_model_name.get()
            active_model = self.models.get(model_name)
            processed_frame, plate_text, status, confidence, original_frame = process_image(active_model,image_path, preprocessing_options)
            results.append({
                'path': image_path,
                'plate': plate_text,
                'status': status,
                'confidence': confidence,
                'success': processed_frame is not None
            })
            
            if progress_callback:
                progress_callback(i + 1, total)
                
        except Exception as e:
            results.append({
                'path': image_path,
                'error': str(e),
                'success': False
            })
    
    return results

def process_image(model, image_path, preprocessing_options):
    """Main function that processes the uploaded image with preprocessing options."""
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            return None, "Error: Could not read the image file.", "Error", None, None

        original_frame = frame.copy()
        
        if any(preprocessing_options.values()):
            frame = preprocess_image(frame, preprocessing_options)

        results = model(frame)
        all_detections = []

        for result in results:
            boxes = result.boxes
            if len(boxes) == 0:
                continue

            # Process all detected plates, not just the first one
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]

                # Crop from original image for cleaner OCR
                plate_crop = original_frame[y1:y2, x1:x2]
                
                if preprocessing_options.get('plate_ocr_enhancement', False):
                    plate_crop = preprocess_plate_crop(plate_crop)

                ocr_result = reader.readtext(plate_crop)

                if ocr_result:
                    raw_text = " ".join([res[1] for res in ocr_result])
                    plate_text = clean_plate_text(raw_text)
                    status = "Registered" if plate_text in REGISTERED_PLATES else "Not Registered"
                    
                    # Draw bounding box and text
                    color = (0, 255, 0) if status == "Registered" else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(frame, f"{plate_text} ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    all_detections.append({
                        'plate': plate_text,
                        'status': status,
                        'confidence': confidence.item(),
                        'bbox': (x1, y1, x2, y2)
                    })

        if all_detections:
            # Return the detection with highest confidence
            best_detection = max(all_detections, key=lambda x: x['confidence'])
            return frame, best_detection['plate'], best_detection['status'], best_detection['confidence'], original_frame
        else:
            return frame, "No Plate Detected", "N/A", None, original_frame

    except Exception as e:
        return None, f"An error occurred: {e}", "Error", None, None

# --- Advanced Modern GUI Implementation ---

class AdvancedPlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 AI License Plate Recognition Pro")
        self.root.geometry("1600x1000")
        self.root.configure(bg="#0f0f23")
        self.root.minsize(1400, 900)
        self.models = {}
        self.active_model_name = tk.StringVar()
        self.load_models() # Load models on startup
        # Advanced styling
        self.setup_advanced_styles()
        
        # Data storage
        self.detection_history = []
        self.statistics = {'total': 0, 'registered': 0, 'not_registered': 0}
        self.current_image_path = None
        self.batch_mode = False
        self.auto_save_enabled = tk.BooleanVar(value=False)
        
        # Preprocessing options with presets
        self.preprocessing_options = {
            'noise_reduction': tk.BooleanVar(),
            'contrast_enhancement': tk.BooleanVar(),
            'brightness_contrast': tk.BooleanVar(),
            'sharpening': tk.BooleanVar(),
            'gamma_correction': tk.BooleanVar(),
            'edge_enhancement': tk.BooleanVar(),
            'plate_ocr_enhancement': tk.BooleanVar()
        }
        
        self.setup_advanced_gui()
        self.load_settings()
    
    def load_models(self):
        """Load YOLO models at startup."""
        try:
            self.models['Specialized (best.pt)'] = YOLO('best.pt')
            self.active_model_name.set('Specialized (best.pt)') # Default model
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load 'best.pt'. This is the primary model and is required.\nError: {e}")
            self.root.destroy()
            return

        try:
            # This will download yolov8n.pt if it doesn't exist
            self.models['General (yolov8n.pt)'] = YOLO('yolov8n.pt')
        except Exception as e:
            messagebox.showwarning("Model Warning", f"Could not load or download 'yolov8n.pt'. The general model will be unavailable.\nError: {e}")

        
    def setup_advanced_styles(self):
        """Configure advanced dark theme styles with animations."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Advanced color scheme
        self.colors = {
            'bg_primary': '#0f0f23',
            'bg_secondary': '#1a1a3a',
            'bg_tertiary': '#252547',
            'accent_blue': '#00d4ff',
            'accent_green': '#00ff9f',
            'accent_red': '#ff3366',
            'accent_orange': '#ff9500',
            'text_primary': '#ffffff',
            'text_secondary': '#b4b4d1',
            'text_muted': '#7575a3'
        }
        
        # Configure advanced styles
        style.configure('Advanced.TFrame', background=self.colors['bg_secondary'])
        style.configure('Card.TFrame', background=self.colors['bg_tertiary'], relief='flat')
        style.configure('Header.TLabel', 
                       background=self.colors['bg_primary'], 
                       foreground=self.colors['accent_blue'], 
                       font=('Segoe UI', 24, 'bold'))

    def setup_advanced_gui(self):
        """Setup the advanced GUI with modern components."""
        # Create main layout
        self.create_advanced_header()
        self.create_main_dashboard()
        self.create_status_bar()

    def create_advanced_header(self):
        """Create an impressive header with gradient effect."""
        header_frame = tk.Frame(self.root, bg=self.colors['bg_primary'], height=120)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Main title with glow effect
        title_frame = tk.Frame(header_frame, bg=self.colors['bg_primary'])
        title_frame.pack(expand=True)
        
        # Icon and title
        title_container = tk.Frame(title_frame, bg=self.colors['bg_primary'])
        title_container.pack(pady=20)
        
        title_label = tk.Label(title_container,
                              text="🚗 AI LICENSE PLATE RECOGNITION PRO",
                              font=("Segoe UI", 28, "bold"),
                              fg=self.colors['accent_blue'],
                              bg=self.colors['bg_primary'])
        title_label.pack()
        
        subtitle_label = tk.Label(title_container,
                                 text="Advanced Computer Vision • Real-time Processing • Smart Analytics",
                                 font=("Segoe UI", 12),
                                 fg=self.colors['text_secondary'],
                                 bg=self.colors['bg_primary'])
        subtitle_label.pack(pady=(5, 0))

    def create_main_dashboard(self):
        """Create the main dashboard layout."""
        dashboard_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        dashboard_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
         # Left panel with scroll
        left_panel_container = tk.Frame(dashboard_frame, bg=self.colors['bg_secondary'], width=400)
        left_panel_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel_container.pack_propagate(False)

        # Canvas + Scrollbar
        canvas = tk.Canvas(left_panel_container, bg=self.colors['bg_secondary'], highlightthickness=0, borderwidth=0)
        scrollbar = tk.Scrollbar(left_panel_container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_secondary'])
        self.left_panel = scrollable_frame  # save reference
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )


        canvas.configure(yscrollcommand=scrollbar.set)
        def _on_mouse_wheel(event):
            # Windows & Mac use event.delta, Linux uses Button-4/5
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        scrollable_frame.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mouse_wheel))
        scrollable_frame.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        
        
        # Right panel - Image Display
        right_panel = tk.Frame(dashboard_frame, bg=self.colors['bg_primary'])
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.setup_advanced_left_panel(self.left_panel)
        self.setup_advanced_right_panel(right_panel)

    def setup_advanced_left_panel(self, parent):
        """Setup advanced left control panel."""
        # Quick Actions Card
        self.create_quick_actions_card(parent)
        self.create_model_selection_card(parent)
        # Smart Preprocessing Card
        self.create_preprocessing_card_with_auto_tune(parent)
         
        # Detection Results Card
        self.create_results_card(parent)
        
        # Analytics Dashboard Card
        self.create_analytics_card(parent)
        
        # History Management Card
        self.create_history_card(parent)
    def create_auto_tune_section(self, parent):
        """Add auto-tune section to preprocessing card."""
        # Auto-tune frame
        auto_tune_frame = tk.Frame(parent, bg=self.colors['bg_tertiary'])
        auto_tune_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Separator
        separator = tk.Frame(auto_tune_frame, height=1, bg=self.colors['text_muted'])
        separator.pack(fill=tk.X, pady=(0, 10))
        
        # Auto-tune header
        auto_header = tk.Label(auto_tune_frame, text="🎯 Auto-Tune Settings",
                            font=("Segoe UI", 10, "bold"),
                            fg=self.colors['accent_orange'],
                            bg=self.colors['bg_tertiary'])
        auto_header.pack(anchor="w", padx=10)
        
        # Auto-tune options
        tune_options_frame = tk.Frame(auto_tune_frame, bg=self.colors['bg_tertiary'])
        tune_options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Enable auto-tune
        self.auto_tune_enabled = tk.BooleanVar()
        auto_tune_cb = self.create_modern_checkbox(
            tune_options_frame, "🔧 Enable Auto-Tune", self.auto_tune_enabled
        )
        auto_tune_cb.pack(anchor="w")
        
        # Auto-tune mode selection
        mode_frame = tk.Frame(tune_options_frame, bg=self.colors['bg_tertiary'])
        mode_frame.pack(fill=tk.X, pady=(5, 0))
        
        tk.Label(mode_frame, text="Mode:", font=("Segoe UI", 9),
                fg=self.colors['text_secondary'], bg=self.colors['bg_tertiary']).pack(side=tk.LEFT)
        
        self.auto_tune_mode = tk.StringVar(value="fast")
        
        modes = [("⚡ Fast (5 combinations)", "fast"),
                ("🎯 Balanced (10 combinations)", "balanced"), 
                ("🔬 Thorough (20 combinations)", "thorough")]
        
        for text, mode in modes:
            rb = tk.Radiobutton(mode_frame, text=text, value=mode,
                            variable=self.auto_tune_mode,
                            font=("Segoe UI", 8),
                            bg=self.colors['bg_tertiary'],
                            fg=self.colors['text_primary'],
                            selectcolor=self.colors['bg_secondary'],
                            activebackground=self.colors['bg_tertiary'])
            rb.pack(anchor="w", padx=(20, 0))
        
        # Auto-apply best settings checkbox
        self.auto_apply_best = tk.BooleanVar(value=True)
        auto_apply_cb = self.create_modern_checkbox(
            tune_options_frame, "📋 Auto-apply best settings", self.auto_apply_best
        )
        auto_apply_cb.pack(anchor="w", pady=(5, 0))

    def get_auto_tune_combinations(self, mode="fast"):
        """Get preprocessing combinations for auto-tuning based on mode."""
        base_combinations = [
            # Basic combinations
            {},  # No preprocessing
            {'contrast_enhancement': True},
            {'brightness_contrast': True},
            {'sharpening': True},
            {'gamma_correction': True}
        ]
        
        balanced_combinations = [
            {'noise_reduction': True},
            {'edge_enhancement': True},
            {'contrast_enhancement': True, 'brightness_contrast': True},
            {'contrast_enhancement': True, 'sharpening': True},
            {'gamma_correction': True, 'brightness_contrast': True}
        ]
        
        thorough_combinations = [
            {'noise_reduction': True, 'contrast_enhancement': True},
            {'sharpening': True, 'edge_enhancement': True},
            {'gamma_correction': True, 'contrast_enhancement': True},
            {'brightness_contrast': True, 'sharpening': True},
            {'noise_reduction': True, 'brightness_contrast': True},
            {'contrast_enhancement': True, 'edge_enhancement': True},
            {'gamma_correction': True, 'sharpening': True},
            {'noise_reduction': True, 'gamma_correction': True},
            {'brightness_contrast': True, 'edge_enhancement': True},
            # Complex combinations
            {'noise_reduction': True, 'contrast_enhancement': True, 'sharpening': True},
            {'gamma_correction': True, 'brightness_contrast': True, 'edge_enhancement': True},
            {'contrast_enhancement': True, 'sharpening': True, 'edge_enhancement': True},
            {'noise_reduction': True, 'gamma_correction': True, 'brightness_contrast': True},
            {'noise_reduction': True, 'contrast_enhancement': True, 'edge_enhancement': True},
            {'gamma_correction': True, 'contrast_enhancement': True, 'sharpening': True}
        ]
        
        if mode == "fast":
            return base_combinations
        elif mode == "balanced":
            return base_combinations + balanced_combinations
        else:  # thorough
            return base_combinations + balanced_combinations + thorough_combinations

    def auto_tune_image(self, image_path, callback=None):
        """Auto-tune image preprocessing to find best settings."""
        if not self.auto_tune_enabled.get():
            return None
        
        mode = self.auto_tune_mode.get()
        combinations = self.get_auto_tune_combinations(mode)
        
        self.update_status(f"🎯 Auto-tuning with {len(combinations)} combinations...")
        self.show_progress(f"Auto-tuning ({mode} mode)")
        
        best_result = None
        best_confidence = 0
        best_combination = {}
        best_plate = "No Plate Detected"
        best_status = "N/A"
        
        model_name = self.active_model_name.get()
        active_model = self.models.get(model_name)
        
        if not active_model:
            if callback:
                callback(None)
            return None
        
        for i, combination in enumerate(combinations):
            try:
                # Update progress
                progress_text = f"Testing combination {i+1}/{len(combinations)}"
                if callback:
                    self.root.after(0, callback, i+1, len(combinations), progress_text)
                
                # Process with this combination
                processed_frame, plate_text, status, confidence, original_frame = process_image(
                    active_model, image_path, combination
                )
                
                # Check if this result is better
                if confidence is not None and confidence > best_confidence:
                    best_confidence = confidence
                    best_combination = combination.copy()
                    best_result = (processed_frame, plate_text, status, confidence, original_frame)
                    best_plate = plate_text
                    best_status = status
                    
            except Exception as e:
                continue  # Skip failed combinations
        
        # Apply best settings if auto-apply is enabled
        if self.auto_apply_best.get() and best_combination:
            self.apply_combination(best_combination)
        
        return {
            'result': best_result,
            'combination': best_combination,
            'confidence': best_confidence,
            'plate': best_plate,
            'status': best_status,
            'total_tested': len(combinations)
        }
    
    def apply_combination(self, combination):
        """Apply a preprocessing combination to the UI."""
        # Clear current settings
        for var in self.preprocessing_options.values():
            var.set(False)
        
        # Apply new combination
        for key, value in combination.items():
            if key in self.preprocessing_options:
                self.preprocessing_options[key].set(value)

    def show_auto_tune_results(self, tune_results):
        """Show auto-tune results in a dialog."""
        if not tune_results or not tune_results['result']:
            messagebox.showinfo("Auto-Tune Results", 
                            f"Auto-tuning completed but no improvements found.\n"
                            f"Tested {tune_results['total_tested'] if tune_results else 0} combinations.")
            return
        
        # Create results dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("🎯 Auto-Tune Results")
        dialog.geometry("500x500")
        dialog.configure(bg=self.colors['bg_secondary'])
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (500 // 2)
        y = (dialog.winfo_screenheight() // 2) - (400 // 2)
        dialog.geometry(f"500x400+{x}+{y}")
        
        # Header
        header = tk.Label(dialog, text="🎯 Auto-Tune Results",
                        font=("Segoe UI", 16, "bold"),
                        fg=self.colors['accent_orange'],
                        bg=self.colors['bg_secondary'])
        header.pack(pady=20)
        
        # Results frame
        results_frame = tk.Frame(dialog, bg=self.colors['bg_tertiary'], relief='solid', bd=1)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Best result info
        info_text = f"""
    ✅ Auto-tuning completed successfully!

    📊 Results:
    • License Plate: {tune_results['plate']}
    • Status: {tune_results['status']}
    • Confidence: {tune_results['confidence']:.1%}
    • Combinations Tested: {tune_results['total_tested']}

    🔧 Best Settings Applied:
    """
        
        for key, value in tune_results['combination'].items():
            if value:
                option_names = {
                    'noise_reduction': 'Noise Reduction',
                    'contrast_enhancement': 'Contrast Enhancement', 
                    'brightness_contrast': 'Auto Brightness/Contrast',
                    'sharpening': 'Sharpening',
                    'gamma_correction': 'Gamma Correction',
                    'edge_enhancement': 'Edge Enhancement'
                }
                info_text += f"   • {option_names.get(key, key)}\n"
        
        if not tune_results['combination']:
            info_text += "   • No preprocessing (original image was best)\n"
        
        info_label = tk.Label(results_frame, text=info_text,
                            font=("Segoe UI", 10),
                            fg=self.colors['text_primary'],
                            bg=self.colors['bg_tertiary'],
                            justify=tk.LEFT)
        info_label.pack(padx=20, pady=20, anchor="w")
        
        # Buttons
        button_frame = tk.Frame(dialog, bg=self.colors['bg_secondary'])
        button_frame.pack(pady=10)
        
        ok_btn = self.create_modern_button(button_frame, "✅ OK", dialog.destroy,
                                        self.colors['accent_green'], width=100, height=35)
        ok_btn.pack(side=tk.LEFT, padx=5)
        
        revert_btn = self.create_modern_button(button_frame, "↶ Revert", 
                                            lambda: self.revert_auto_tune(dialog),
                                            self.colors['accent_red'], width=100, height=35)
        revert_btn.pack(side=tk.LEFT, padx=5)

    def revert_auto_tune(self, dialog):
        """Revert auto-tune changes."""
        self.clear_presets()
        dialog.destroy()
        self.update_status("Auto-tune settings reverted")

    def create_quick_actions_card(self, parent):
        """Create quick actions card with modern buttons."""
        card_frame = self.create_card(parent, "🚀 Quick Actions", height=180)
        
        # File operations
        file_frame = tk.Frame(card_frame, bg=self.colors['bg_tertiary'])
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.upload_btn = self.create_modern_button(
            file_frame, "📂 Select Image", self.upload_action_with_auto_tune,
            bg_color=self.colors['accent_blue'], width=180, height=40
        )
        self.upload_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.batch_btn = self.create_modern_button(
            file_frame, "📁 Batch Process", self.batch_process_with_auto_tune,
            bg_color=self.colors['accent_green'], width=180, height=40
        )
        self.batch_btn.pack(side=tk.LEFT)
        
        # Processing operations
        process_frame = tk.Frame(card_frame, bg=self.colors['bg_tertiary'])
        process_frame.pack(fill=tk.X, pady=10)
        
        self.reprocess_btn = self.create_modern_button(
            process_frame, "🔄 Reprocess", self.reprocess_image,
            bg_color=self.colors['accent_orange'], width=180, height=40, state="disabled"
        )
        self.reprocess_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.export_btn = self.create_modern_button(
            process_frame, "💾 Export Results", self.export_results,
            bg_color=self.colors['accent_red'], width=180, height=40
        )
        self.export_btn.pack(side=tk.LEFT)

    def create_model_selection_card(self, parent):
        """Creates a new card for selecting the YOLO model."""
        card_frame = self.create_card(parent, "🧠 AI Model Selection")
        
        for model_name in self.models.keys():
            rb = ttk.Radiobutton(card_frame, text=model_name, value=model_name,
                                variable=self.active_model_name, command=self.on_model_change,
                                style="TRadiobutton")
            rb.pack(anchor="w", padx=10, pady=5)
    
    def create_preprocessing_card_with_auto_tune(self, parent):
        """Create smart preprocessing options card with auto-tune."""
        card_frame = self.create_card(parent, "⚙️ Smart Image Enhancement", height=350)  # Increased height
        
        # Preset buttons (existing code)
        preset_frame = tk.Frame(card_frame, bg=self.colors['bg_tertiary'])
        preset_frame.pack(fill=tk.X, pady=(0, 15))
        
        presets = [
            ("🌙 Night Mode", self.apply_night_preset),
            ("☀️ Daylight", self.apply_daylight_preset),
            ("🌧️ Weather", self.apply_weather_preset),
            ("🔧 Custom", self.clear_presets)
        ]
        
        for i, (text, command) in enumerate(presets):
            btn = self.create_compact_button(preset_frame, text, command, width=85)
            btn.grid(row=0, column=i, padx=2)
        
        preset_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Individual options (existing code)
        options_frame = tk.Frame(card_frame, bg=self.colors['bg_tertiary'])
        options_frame.pack(fill=tk.X, pady=10)
        
        options_display = {
            'noise_reduction': '🔧 Noise Reduction',
            'contrast_enhancement': '🌟 Contrast Boost', 
            'brightness_contrast': '☀️ Auto Exposure',
            'sharpening': '🔍 Sharpening',
            'gamma_correction': '🔆 Gamma Correction',
            'edge_enhancement': '🔍 Edge Enhancement'
        }

        for i, (key, display_name) in enumerate(options_display.items()):
            cb = self.create_modern_checkbox(
                options_frame, display_name, self.preprocessing_options[key]
            )
            cb.grid(row=i//2, column=i%2, sticky="w", padx=10, pady=3)
        
        # Add auto-tune section
        self.create_auto_tune_section(card_frame)
    
    def create_results_card(self, parent):
        """Create detection results display card."""
        card_frame = self.create_card(parent, "📊 Detection Results")
        
        # Results grid
        results_grid = tk.Frame(card_frame, bg=self.colors['bg_tertiary'])
        results_grid.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Plate number display
        plate_frame = tk.Frame(results_grid, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        plate_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(plate_frame, text="License Plate", 
                font=("Segoe UI", 9, "bold"), 
                fg=self.colors['text_muted'], 
                bg=self.colors['bg_secondary']).pack(pady=(5, 0))
        
        self.plate_display = tk.Label(plate_frame, text="─────", 
                                    font=("Consolas", 16, "bold"), 
                                    fg=self.colors['text_primary'], 
                                    bg=self.colors['bg_secondary'])
        self.plate_display.pack(pady=(0, 8))
        
        # Status and confidence row
        status_frame = tk.Frame(results_grid, bg=self.colors['bg_tertiary'])
        status_frame.pack(fill=tk.X)
        
        # Status
        status_container = tk.Frame(status_frame, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        status_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        tk.Label(status_container, text="Status", 
                font=("Segoe UI", 9, "bold"), 
                fg=self.colors['text_muted'], 
                bg=self.colors['bg_secondary']).pack(pady=(5, 0))
        
        self.status_display = tk.Label(status_container, text="Ready", 
                                     font=("Segoe UI", 12, "bold"), 
                                     fg=self.colors['text_secondary'], 
                                     bg=self.colors['bg_secondary'])
        self.status_display.pack(pady=(0, 8))
        
        # Confidence
        conf_container = tk.Frame(status_frame, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        conf_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        tk.Label(conf_container, text="Confidence", 
                font=("Segoe UI", 9, "bold"), 
                fg=self.colors['text_muted'], 
                bg=self.colors['bg_secondary']).pack(pady=(5, 0))
        
        self.confidence_display = tk.Label(conf_container, text="─", 
                                         font=("Segoe UI", 12, "bold"), 
                                         fg=self.colors['text_secondary'], 
                                         bg=self.colors['bg_secondary'])
        self.confidence_display.pack(pady=(0, 8))

    def create_analytics_card(self, parent):
        """Create analytics dashboard card."""
        card_frame = self.create_card(parent, "📈 Analytics Dashboard", height=160)
        
        # Statistics grid
        stats_frame = tk.Frame(card_frame, bg=self.colors['bg_tertiary'])
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Create stat boxes
        stat_boxes = [
            ("Total Scanned", "total", self.colors['accent_blue']),
            ("Registered", "registered", self.colors['accent_green']),
            ("Not Registered", "not_registered", self.colors['accent_red']),
            ("Success Rate", "success_rate", self.colors['accent_orange'])
        ]
        
        for i, (label, key, color) in enumerate(stat_boxes):
            stat_box = tk.Frame(stats_frame, bg=self.colors['bg_secondary'], relief='solid', bd=1)
            stat_box.grid(row=i//2, column=i%2, sticky="nsew", padx=2, pady=2)
            
            tk.Label(stat_box, text=label, 
                    font=("Segoe UI", 8, "bold"), 
                    fg=self.colors['text_muted'], 
                    bg=self.colors['bg_secondary']).pack(pady=(5, 0))
            
            stat_label = tk.Label(stat_box, text="0", 
                                 font=("Segoe UI", 14, "bold"), 
                                 fg=color, 
                                 bg=self.colors['bg_secondary'])
            stat_label.pack(pady=(0, 5))
            
            setattr(self, f"{key}_stat", stat_label)
        
        stats_frame.grid_columnconfigure((0, 1), weight=1)
        stats_frame.grid_rowconfigure((0, 1), weight=1)

    def create_history_card(self, parent):
        """Create history management card."""
        card_frame = self.create_card(parent, "📝 Detection History", expand=True)
        
        # History controls
        history_controls = tk.Frame(card_frame, bg=self.colors['bg_tertiary'])
        history_controls.pack(fill=tk.X, padx=15, pady=(10, 0))
        
        # Auto-save toggle
        auto_save_cb = self.create_modern_checkbox(
            history_controls, "Auto-save results", self.auto_save_enabled
        )
        auto_save_cb.pack(side=tk.LEFT)
        
        # Clear history button
        clear_btn = self.create_compact_button(
            history_controls, "🗑️ Clear", self.clear_history, width=80
        )
        clear_btn.pack(side=tk.RIGHT)
        
        # History list with advanced styling
        history_frame = tk.Frame(card_frame, bg=self.colors['bg_tertiary'])
        history_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Create Treeview for better history display
        columns = ('Time', 'Plate', 'Status', 'Confidence')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        self.history_tree.heading('Time', text='Time')
        self.history_tree.heading('Plate', text='License Plate')
        self.history_tree.heading('Status', text='Status')
        self.history_tree.heading('Confidence', text='Conf%')
        
        self.history_tree.column('Time', width=80, anchor='center')
        self.history_tree.column('Plate', width=100, anchor='center')
        self.history_tree.column('Status', width=90, anchor='center')
        self.history_tree.column('Confidence', width=60, anchor='center')
        
        # Scrollbar for history
        history_scroll = ttk.Scrollbar(history_frame, orient='vertical', command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scroll.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_advanced_right_panel(self, parent):
        """Setup advanced right panel with image displays."""
        # Image panel header
        header_frame = tk.Frame(parent, bg=self.colors['bg_primary'])
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(header_frame,
                text="🖼️ IMAGE ANALYSIS WORKSPACE",
                font=("Segoe UI", 18, "bold"),
                fg=self.colors['accent_blue'],
                bg=self.colors['bg_primary']).pack()
        
        # Image display container
        image_container = tk.Frame(parent, bg=self.colors['bg_primary'])
        image_container.pack(fill=tk.BOTH, expand=True)
        
        # Original image panel
        original_panel = self.create_image_panel(
            image_container, "📷 Original Image", 
            "Drag & drop or click 'Select Image' to load your license plate photo"
        )
        original_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Processed image panel  
        processed_panel = self.create_image_panel(
            image_container, "🔍 Processed & Detected",
            "Enhanced image with AI detection results will appear here"
        )
        processed_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

    def create_image_panel(self, parent, title, placeholder_text):
        """Create a modern image display panel."""
        panel_frame = tk.Frame(parent, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        
        # Panel header
        header = tk.Frame(panel_frame, bg=self.colors['bg_tertiary'], height=40)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text=title,
                font=("Segoe UI", 12, "bold"),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_tertiary']).pack(pady=12)
        
        # Image display area
        image_label = tk.Label(panel_frame,
                             bg=self.colors['bg_primary'],
                             fg=self.colors['text_muted'],
                             text=f"📷\n\n{placeholder_text}",
                             font=("Segoe UI", 11),
                             relief='sunken',
                             bd=2,
                             justify=tk.CENTER)
        image_label.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Store reference based on title
        if "Original" in title:
            self.original_image_label = image_label
        else:
            self.processed_image_label = image_label
        
        return panel_frame

    def create_card(self, parent, title, height=None, expand=False):
        """Create a modern card container."""
        card_container = tk.Frame(parent, bg=self.colors['bg_secondary'], relief='solid', bd=1)
        
        if expand:
            card_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15), padx=15)
        else:
            card_container.pack(fill=tk.X, pady=(0, 15), padx=15)
            if height:
                card_container.configure(height=height)
                card_container.pack_propagate(False)
        
        # Card header
        header_frame = tk.Frame(card_container, bg=self.colors['bg_tertiary'], height=35)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text=title,
                font=("Segoe UI", 11, "bold"),
                fg=self.colors['text_primary'],
                bg=self.colors['bg_tertiary']).pack(pady=8)
        
        # Card content
        content_frame = tk.Frame(card_container, bg=self.colors['bg_tertiary'])
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        return content_frame

    def create_modern_button(self, parent, text, command, bg_color, width=120, height=35, state="normal"):
        """Create a modern styled button."""
        button = tk.Button(parent,
                          text=text,
                          command=command,
                          font=("Segoe UI", 10, "bold"),
                          bg=bg_color,
                          fg="white",
                          relief="flat",
                          bd=0,
                          width=width//8,
                          cursor="hand2",
                          state=state)
        
        # Hover effects
        def on_enter(e):
            if button['state'] != 'disabled':
                button.configure(bg=self.lighten_color(bg_color, 0.2))
        
        def on_leave(e):
            if button['state'] != 'disabled':
                button.configure(bg=bg_color)
        
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        
        return button

    def create_compact_button(self, parent, text, command, width=100):
        """Create a compact button for toolbars."""
        return tk.Button(parent,
                        text=text,
                        command=command,
                        font=("Segoe UI", 9),
                        bg=self.colors['bg_secondary'],
                        fg=self.colors['text_primary'],
                        relief="solid",
                        bd=1,
                        width=width//8,
                        cursor="hand2")

    def create_modern_checkbox(self, parent, text, variable):
        """Create a modern styled checkbox."""
        return tk.Checkbutton(parent,
                             text=text,
                             variable=variable,
                             font=("Segoe UI", 10),
                             bg=self.colors['bg_tertiary'],
                             fg=self.colors['text_primary'],
                             selectcolor=self.colors['bg_secondary'],
                             activebackground=self.colors['bg_tertiary'],
                             activeforeground=self.colors['text_primary'],
                             cursor="hand2",
                             command=self.on_preprocessing_change)

    def create_status_bar(self):
        """Create an advanced status bar."""
        status_frame = tk.Frame(self.root, bg=self.colors['bg_secondary'], height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        # Status text
        self.status_text = tk.Label(status_frame,
                                   text="Ready • Load an image to begin analysis",
                                   font=("Segoe UI", 9),
                                   fg=self.colors['text_secondary'],
                                   bg=self.colors['bg_secondary'])
        self.status_text.pack(side=tk.LEFT, padx=15, pady=6)
        
        # Progress bar (initially hidden)
        self.progress_bar = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        
        # Version info
        version_label = tk.Label(status_frame,
                                text="AI Engine: YOLOv8 + EasyOCR | Version 2.0 Pro",
                                font=("Segoe UI", 8),
                                fg=self.colors['text_muted'],
                                bg=self.colors['bg_secondary'])
        version_label.pack(side=tk.RIGHT, padx=15, pady=6)

    def lighten_color(self, color, amount):
        """Utility to lighten colors for hover effects."""
        # Simple color lightening - you can enhance this
        if color == self.colors['accent_blue']:
            return "#33ddff"
        elif color == self.colors['accent_green']:
            return "#33ffb2"
        elif color == self.colors['accent_red']:
            return "#ff5577"
        elif color == self.colors['accent_orange']:
            return "#ffad33"
        else:
            return color

    # Preset functions
    def apply_night_preset(self):
        """Apply night/low-light preset."""
        self.clear_presets()
        self.preprocessing_options['gamma_correction'].set(True)
        self.preprocessing_options['brightness_contrast'].set(True)
        self.preprocessing_options['contrast_enhancement'].set(True)
        self.update_status("Night mode preset applied")

    def apply_daylight_preset(self):
        """Apply daylight preset."""
        self.clear_presets()
        self.preprocessing_options['sharpening'].set(True)
        self.update_status("Daylight preset applied")

    def apply_weather_preset(self):
        """Apply weather conditions preset."""
        self.clear_presets()
        self.preprocessing_options['noise_reduction'].set(True)
        self.preprocessing_options['contrast_enhancement'].set(True)
        self.preprocessing_options['edge_enhancement'].set(True)
        self.update_status("Weather conditions preset applied")

    def clear_presets(self):
        """Clear all preprocessing options."""
        for var in self.preprocessing_options.values():
            var.set(False)
        self.update_status("Custom preset selected")

    def on_preprocessing_change(self):
        """Handle preprocessing option changes."""
        if self.current_image_path:
            self.reprocess_btn.config(state="normal")

    def on_model_change(self):
        """Handles the model selection change."""
        model_name = self.active_model_name.get()
        self.update_status(f"Active model changed to: {model_name}")
        self.root.title(f"🚗 AI LPR Pro (Model: {model_name})")
        if self.current_image_path:
            self.reprocess_btn.config(state="normal")

    def upload_action_with_auto_tune(self):
        """Handle single image upload with auto-tune option."""
        filetypes = (
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        )
        
        filepath = filedialog.askopenfilename(
            title="Select License Plate Image",
            filetypes=filetypes
        )
        
        if filepath:
            self.current_image_path = filepath
            self.process_single_image_with_auto_tune()  # Use the new auto-tune method

    def batch_process_with_auto_tune(self):
        """Handle batch processing of multiple images with auto-tune options."""
        filepaths = filedialog.askopenfilenames(
            title="Select Multiple Images for Batch Processing",
            filetypes=(
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            )
        )
        
        if filepaths:
            if self.auto_tune_enabled.get():
                self.show_batch_auto_tune_dialog(filepaths)
            else:
                self.batch_mode = True
                self.process_batch_images(filepaths)

    def show_batch_auto_tune_dialog(self, filepaths):
        """Show dialog for batch auto-tune options."""
        dialog = tk.Toplevel(self.root)
        dialog.title("🚀 Batch Auto-Tune Options")
        dialog.geometry("600x700")
        dialog.configure(bg=self.colors['bg_secondary'])
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (300)
        y = (dialog.winfo_screenheight() // 2) - (250)
        dialog.geometry(f"600x700+{x}+{y}")
        
        # Header
        header = tk.Label(dialog, text="🚀 Batch Auto-Tune Configuration",
                        font=("Segoe UI", 16, "bold"),
                        fg=self.colors['accent_blue'],
                        bg=self.colors['bg_secondary'])
        header.pack(pady=20)
        
        # Info
        info_text = f"Selected {len(filepaths)} images for batch processing with auto-tuning."
        info_label = tk.Label(dialog, text=info_text,
                            font=("Segoe UI", 11),
                            fg=self.colors['text_secondary'],
                            bg=self.colors['bg_secondary'])
        info_label.pack(pady=(0, 20))
        
        # Options frame
        options_frame = tk.Frame(dialog, bg=self.colors['bg_tertiary'], relief='solid', bd=1)
        options_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Auto-tune strategy
        strategy_frame = tk.Frame(options_frame, bg=self.colors['bg_tertiary'])
        strategy_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(strategy_frame, text="🎯 Auto-Tune Strategy:",
                font=("Segoe UI", 12, "bold"),
                fg=self.colors['accent_orange'],
                bg=self.colors['bg_tertiary']).pack(anchor="w")
        
        self.batch_auto_tune_strategy = tk.StringVar(value="individual")
        
        strategies = [
            ("🔄 Individual - Optimize each image separately (Best Quality)", "individual"),
            ("⚡ Global - Find one setting for all images (Faster)", "global"),
            ("📊 Adaptive - Learn from each image (Smart)", "adaptive")
        ]
        
        for text, strategy in strategies:
            rb = tk.Radiobutton(strategy_frame, text=text, value=strategy,
                            variable=self.batch_auto_tune_strategy,
                            font=("Segoe UI", 10),
                            bg=self.colors['bg_tertiary'],
                            fg=self.colors['text_primary'],
                            selectcolor=self.colors['bg_secondary'],
                            activebackground=self.colors['bg_tertiary'],
                            wraplength=500)
            rb.pack(anchor="w", padx=20, pady=5)
        
        # Performance options
        perf_frame = tk.Frame(options_frame, bg=self.colors['bg_tertiary'])
        perf_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(perf_frame, text="⚡ Performance Options:",
                font=("Segoe UI", 12, "bold"),
                fg=self.colors['accent_green'],
                bg=self.colors['bg_tertiary']).pack(anchor="w")
        
        self.batch_early_stop = tk.BooleanVar(value=True)
        early_stop_cb = tk.Checkbutton(perf_frame,
                                    text="🎯 Early stop when confidence > 95%",
                                    variable=self.batch_early_stop,
                                    font=("Segoe UI", 10),
                                    bg=self.colors['bg_tertiary'],
                                    fg=self.colors['text_primary'],
                                    selectcolor=self.colors['bg_secondary'],
                                    activebackground=self.colors['bg_tertiary'])
        early_stop_cb.pack(anchor="w", padx=20, pady=5)
        
        self.batch_parallel_processing = tk.BooleanVar(value=False)
        parallel_cb = tk.Checkbutton(perf_frame,
                                    text="🚀 Parallel processing (Experimental)",
                                    variable=self.batch_parallel_processing,
                                    font=("Segoe UI", 10),
                                    bg=self.colors['bg_tertiary'],
                                    fg=self.colors['text_primary'],
                                    selectcolor=self.colors['bg_secondary'],
                                    activebackground=self.colors['bg_tertiary'])
        parallel_cb.pack(anchor="w", padx=20, pady=5)
        
        # Progress estimation
        est_frame = tk.Frame(options_frame, bg=self.colors['bg_tertiary'])
        est_frame.pack(fill=tk.X, padx=20, pady=20)
        
        mode = self.auto_tune_mode.get()
        combinations = len(self.get_auto_tune_combinations(mode))
        strategy = self.batch_auto_tune_strategy.get()
        
        if strategy == "individual":
            total_tests = len(filepaths) * combinations
        elif strategy == "global":
            total_tests = combinations
        else:  # adaptive
            total_tests = len(filepaths) * (combinations // 2)  # Average estimate
        
        est_text = f"📊 Estimated Tests: {total_tests:,}\n"
        est_text += f"⏱️ Estimated Time: {total_tests * 2} - {total_tests * 5} seconds"
        
        est_label = tk.Label(est_frame, text=est_text,
                            font=("Segoe UI", 9),
                            fg=self.colors['text_muted'],
                            bg=self.colors['bg_tertiary'],
                            justify=tk.LEFT)
        est_label.pack(anchor="w")
        
        # Buttons
        button_frame = tk.Frame(dialog, bg=self.colors['bg_secondary'])
        button_frame.pack(pady=20)
        
        start_btn = self.create_modern_button(button_frame, "🚀 Start Batch Auto-Tune", 
                                            lambda: self.start_batch_auto_tune(dialog, filepaths),
                                            self.colors['accent_blue'], width=200, height=40)
        start_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = self.create_modern_button(button_frame, "❌ Cancel", dialog.destroy,
                                            self.colors['accent_red'], width=100, height=40)
        cancel_btn.pack(side=tk.LEFT, padx=10)

    def process_single_image_with_auto_tune(self):
        """Process a single image with optional auto-tuning."""
        if not self.current_image_path:
            return
        
        if self.auto_tune_enabled.get():
            self.show_progress("Auto-tuning image...")
            threading.Thread(target=self._auto_tune_thread, daemon=True).start()
        else:
            self.show_progress("Processing image...")
            threading.Thread(target=self._process_single_thread, daemon=True).start()

    def _auto_tune_thread(self):
        """Thread function for auto-tuning."""
        try:
            def progress_callback(current, total, message):
                self.root.after(0, self.update_status, message)
            
            tune_results = self.auto_tune_image(self.current_image_path, progress_callback)
            
            if tune_results and tune_results['result']:
                processed_frame, plate_text, status, confidence, original_frame = tune_results['result']
                
                # Update UI with best result
                self.root.after(0, self._update_single_result, 
                            processed_frame, plate_text, status, confidence, original_frame)
                
                # Show auto-tune results
                self.root.after(0, self.show_auto_tune_results, tune_results)
            else:
                # Fall back to current settings
                self.root.after(0, self._process_single_thread)
                
        except Exception as e:
            self.root.after(0, self._handle_error, str(e))

    def _process_single_thread(self):
        """Thread function for single image processing."""
        try:
            options = {key: var.get() for key, var in self.preprocessing_options.items()}
            model_name = self.active_model_name.get()
            active_model = self.models.get(model_name)
            processed_frame, plate_text, status, confidence, original_frame = process_image(active_model,
                self.current_image_path, options)
            
            # Update UI in main thread
            self.root.after(0, self._update_single_result, 
                           processed_frame, plate_text, status, confidence, original_frame)
            
        except Exception as e:
            self.root.after(0, self._handle_error, str(e))

    def _update_single_result(self, processed_frame, plate_text, status, confidence, original_frame):
        """Update UI with single processing result."""
        self.hide_progress()
        
        if processed_frame is not None:
            # Display images
            self.display_images(original_frame, processed_frame)
            
            # Update results
            self.update_detection_results(plate_text, status, confidence)
            
            if status != "N/A":
                # Add to history
                self.add_to_history(plate_text, status, confidence)
        
                # Update statistics
                self.update_statistics(status)
        
                # Auto-save if enabled
                if self.auto_save_enabled.get():
                    self.auto_save_result(plate_text, status, confidence)
            
            self.update_status(f"Detection complete • Plate: {plate_text}")
            
        else:
            self.update_status("Error: Could not process image", error=True)
            messagebox.showerror("Processing Error", plate_text)

    def process_batch_images(self, filepaths):
        """Process multiple images in batch mode."""
        self.update_status(f"Starting batch processing • {len(filepaths)} images")
        self.show_progress("Batch processing...")
        
        # Process in separate thread
        threading.Thread(target=self._process_batch_thread, 
                        args=(filepaths,), daemon=True).start()
    
    def start_batch_auto_tune(self, dialog, filepaths):
        """Start the batch auto-tune process."""
        dialog.destroy()
        self.batch_mode = True
        
        strategy = self.batch_auto_tune_strategy.get()
        early_stop = self.batch_early_stop.get()
        parallel = self.batch_parallel_processing.get()
        
        self.update_status(f"🚀 Starting batch auto-tune ({strategy} strategy)")
        self.show_progress("Initializing batch auto-tune...")
        
        # Start processing thread
        threading.Thread(target=self._batch_auto_tune_thread, 
                        args=(filepaths, strategy, early_stop, parallel), 
                        daemon=True).start()
    
    def _batch_auto_tune_thread(self, filepaths, strategy, early_stop, parallel):
        """Thread function for batch auto-tuning."""
        try:
            if strategy == "individual":
                results = self._batch_individual_auto_tune(filepaths, early_stop)
            elif strategy == "global":
                results = self._batch_global_auto_tune(filepaths, early_stop)
            else:  # adaptive
                results = self._batch_adaptive_auto_tune(filepaths, early_stop)
            
            # Update UI with batch results
            self.root.after(0, self._update_batch_auto_tune_results, results)
            
        except Exception as e:
            self.root.after(0, self._handle_error, str(e))

    def _batch_individual_auto_tune(self, filepaths, early_stop):
        """Individual auto-tune: optimize each image separately."""
        results = []
        model_name = self.active_model_name.get()
        active_model = self.models.get(model_name)
        
        for i, filepath in enumerate(filepaths):
            try:
                self.root.after(0, self.update_status, 
                            f"Auto-tuning image {i+1}/{len(filepaths)}: {os.path.basename(filepath)}")
                
                # Auto-tune this individual image
                best_result = self._find_best_combination_for_image(
                    active_model, filepath, early_stop)
                
                if best_result:
                    results.append({
                        'path': filepath,
                        'plate': best_result['plate'],
                        'status': best_result['status'],
                        'confidence': best_result['confidence'],
                        'combination': best_result['combination'],
                        'tests_performed': best_result['tests_performed'],
                        'success': True
                    })
                    
                    # Add to history
                    if best_result['status'] != "N/A":
                        self.root.after(0, self.add_to_history, 
                                    best_result['plate'], best_result['status'], best_result['confidence'])
                        self.root.after(0, self.update_statistics, best_result['status'])
                else:
                    results.append({
                        'path': filepath,
                        'error': 'No valid detection found',
                        'success': False
                    })
                    
            except Exception as e:
                results.append({
                    'path': filepath,
                    'error': str(e),
                    'success': False
                })
        
        return results

    def _batch_global_auto_tune(self, filepaths, early_stop):
        """Global auto-tune: find one setting that works best for all images."""
        self.root.after(0, self.update_status, "Finding global optimal settings...")
        
        model_name = self.active_model_name.get()
        active_model = self.models.get(model_name)
        
        # Test combinations on a sample of images (max 5 for performance)
        sample_images = filepaths[:min(5, len(filepaths))]
        combinations = self.get_auto_tune_combinations(self.auto_tune_mode.get())
        
        best_global_combination = {}
        best_global_score = 0
        
        for i, combination in enumerate(combinations):
            total_confidence = 0
            valid_detections = 0
            
            self.root.after(0, self.update_status, 
                        f"Testing global combination {i+1}/{len(combinations)}")
            
            for sample_path in sample_images:
                try:
                    _, _, _, confidence, _ = process_image(active_model, sample_path, combination)
                    if confidence is not None:
                        total_confidence += confidence
                        valid_detections += 1
                except:
                    continue
            
            # Calculate average confidence for this combination
            if valid_detections > 0:
                avg_confidence = total_confidence / valid_detections
                if avg_confidence > best_global_score:
                    best_global_score = avg_confidence
                    best_global_combination = combination.copy()
            
            # Early stop if we find a very good combination
            if early_stop and best_global_score > 0.95:
                break
        
        # Apply best combination to all images
        self.root.after(0, self.update_status, "Applying best global settings to all images...")
        
        results = []
        for i, filepath in enumerate(filepaths):
            try:
                self.root.after(0, self.update_status, 
                            f"Processing with global settings {i+1}/{len(filepaths)}")
                
                processed_frame, plate_text, status, confidence, original_frame = process_image(
                    active_model, filepath, best_global_combination)
                
                results.append({
                    'path': filepath,
                    'plate': plate_text,
                    'status': status,
                    'confidence': confidence,
                    'combination': best_global_combination,
                    'success': processed_frame is not None
                })
                
                # Add to history
                if status != "N/A":
                    self.root.after(0, self.add_to_history, plate_text, status, confidence)
                    self.root.after(0, self.update_statistics, status)
                    
            except Exception as e:
                results.append({
                    'path': filepath,
                    'error': str(e),
                    'success': False
                })
        
        return results

    def _batch_adaptive_auto_tune(self, filepaths, early_stop):
        """Adaptive auto-tune: learn from each image and adapt settings."""
        results = []
        model_name = self.active_model_name.get()
        active_model = self.models.get(model_name)
        
        # Keep track of successful combinations
        successful_combinations = {}
        combination_history = []
        
        for i, filepath in enumerate(filepaths):
            try:
                self.root.after(0, self.update_status, 
                            f"Adaptive tuning {i+1}/{len(filepaths)}: {os.path.basename(filepath)}")
                
                # For first few images, do full auto-tune
                if i < 3:
                    best_result = self._find_best_combination_for_image(
                        active_model, filepath, early_stop)
                    
                    if best_result:
                        # Track successful combination
                        combo_key = str(sorted(best_result['combination'].items()))
                        successful_combinations[combo_key] = successful_combinations.get(combo_key, 0) + 1
                        combination_history.append(best_result['combination'])
                else:
                    # For later images, try successful combinations first
                    best_result = self._adaptive_find_best_combination(
                        active_model, filepath, successful_combinations, early_stop)
                
                if best_result:
                    results.append({
                        'path': filepath,
                        'plate': best_result['plate'],
                        'status': best_result['status'],
                        'confidence': best_result['confidence'],
                        'combination': best_result['combination'],
                        'tests_performed': best_result.get('tests_performed', 0),
                        'success': True
                    })
                    
                    # Add to history
                    if best_result['status'] != "N/A":
                        self.root.after(0, self.add_to_history, 
                                    best_result['plate'], best_result['status'], best_result['confidence'])
                        self.root.after(0, self.update_statistics, best_result['status'])
                else:
                    results.append({
                        'path': filepath,
                        'error': 'No valid detection found',
                        'success': False
                    })
                    
            except Exception as e:
                results.append({
                    'path': filepath,
                    'error': str(e),
                    'success': False
                })
        
        return results

    def _find_best_combination_for_image(self, active_model, image_path, early_stop):
        """Find the best preprocessing combination for a single image."""
        combinations = self.get_auto_tune_combinations(self.auto_tune_mode.get())
        
        best_result = None
        best_confidence = 0
        tests_performed = 0
        
        for combination in combinations:
            try:
                processed_frame, plate_text, status, confidence, original_frame = process_image(
                    active_model, image_path, combination)
                
                tests_performed += 1
                
                if confidence is not None and confidence > best_confidence:
                    best_confidence = confidence
                    best_result = {
                        'plate': plate_text,
                        'status': status,
                        'confidence': confidence,
                        'combination': combination.copy(),
                        'tests_performed': tests_performed
                    }
                    
                    # Early stop if confidence is very high
                    if early_stop and confidence > 0.95:
                        break
                        
            except:
                continue
        
        return best_result

    def _adaptive_find_best_combination(self, active_model, image_path, successful_combinations, early_stop):
        """Find best combination using adaptive strategy."""
        # Try most successful combinations first
        sorted_combos = sorted(successful_combinations.items(), key=lambda x: x[1], reverse=True)
        
        best_result = None
        best_confidence = 0
        tests_performed = 0
        
        # Try top 3 most successful combinations first
        for combo_str, success_count in sorted_combos[:3]:
            try:
                # Convert string back to dict
                combination = {}
                combo_items = eval(combo_str)  # Safe since we created it
                for key, value in combo_items:
                    combination[key] = value
                
                processed_frame, plate_text, status, confidence, original_frame = process_image(
                    active_model, image_path, combination)
                
                tests_performed += 1
                
                if confidence is not None and confidence > best_confidence:
                    best_confidence = confidence
                    best_result = {
                        'plate': plate_text,
                        'status': status,
                        'confidence': confidence,
                        'combination': combination.copy(),
                        'tests_performed': tests_performed
                    }
                    
                    if early_stop and confidence > 0.95:
                        break
                        
            except:
                continue
        
        # If no good result from successful combos, do limited full search
        if not best_result or best_confidence < 0.7:
            full_result = self._find_best_combination_for_image(active_model, image_path, early_stop)
            if full_result and (not best_result or full_result['confidence'] > best_confidence):
                best_result = full_result
        
        return best_result

    def _update_batch_auto_tune_results(self, results):
        """Update UI with batch auto-tune results."""
        self.hide_progress()
        self.batch_mode = False
        
        # Process results
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        # Calculate statistics
        total_tests = sum(r.get('tests_performed', 0) for r in successful)
        avg_confidence = sum(r['confidence'] for r in successful if r['confidence']) / len(successful) if successful else 0
        
        # Show comprehensive summary
        self.show_batch_auto_tune_summary(results, successful, failed, total_tests, avg_confidence)

    def show_batch_auto_tune_summary(self, all_results, successful, failed, total_tests, avg_confidence):
        """Show comprehensive batch auto-tune summary."""
        dialog = tk.Toplevel(self.root)
        dialog.title("🚀 Batch Auto-Tune Results")
        dialog.geometry("700x800")
        dialog.configure(bg=self.colors['bg_secondary'])
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (350)
        y = (dialog.winfo_screenheight() // 2) - (300)
        dialog.geometry(f"700x800+{x}+{y}")
        
        # Header
        header = tk.Label(dialog, text="🚀 Batch Auto-Tune Complete!",
                        font=("Segoe UI", 18, "bold"),
                        fg=self.colors['accent_blue'],
                        bg=self.colors['bg_secondary'])
        header.pack(pady=20)
        
        # Results frame with scrollbar
        results_container = tk.Frame(dialog, bg=self.colors['bg_secondary'])
        results_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        canvas = tk.Canvas(results_container, bg=self.colors['bg_tertiary'])
        scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_tertiary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Summary statistics
        summary_text = f"""
    📊 BATCH AUTO-TUNE SUMMARY

    ✅ Successfully Processed: {len(successful)} images
    ❌ Failed: {len(failed)} images  
    🔬 Total Tests Performed: {total_tests:,}
    📈 Average Confidence: {avg_confidence:.1%}
    ⚡ Strategy: {self.batch_auto_tune_strategy.get().title()}

    🏆 TOP PERFORMING SETTINGS:
    """
        
        # Find most common successful combinations
        if successful:
            combo_counts = {}
            for result in successful:
                combo_key = str(sorted(result.get('combination', {}).items()))
                combo_counts[combo_key] = combo_counts.get(combo_key, 0) + 1
            
            top_combos = sorted(combo_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            for i, (combo_str, count) in enumerate(top_combos, 1):
                try:
                    combo_items = eval(combo_str)
                    if combo_items:
                        summary_text += f"\n{i}. Used {count} times:\n"
                        for key, value in combo_items:
                            if value:
                                option_names = {
                                    'noise_reduction': 'Noise Reduction',
                                    'contrast_enhancement': 'Contrast Enhancement', 
                                    'brightness_contrast': 'Auto Brightness/Contrast',
                                    'sharpening': 'Sharpening',
                                    'gamma_correction': 'Gamma Correction',
                                    'edge_enhancement': 'Edge Enhancement'
                                }
                                summary_text += f"   • {option_names.get(key, key)}\n"
                    else:
                        summary_text += f"\n{i}. Original image (no preprocessing) - {count} times\n"
                except:
                    continue
        
        summary_label = tk.Label(scrollable_frame, text=summary_text,
                            font=("Consolas", 10),
                            fg=self.colors['text_primary'],
                            bg=self.colors['bg_tertiary'],
                            justify=tk.LEFT)
        summary_label.pack(padx=20, pady=20, anchor="w")
        
        # Individual results
        if successful:
            results_text = "📋 DETAILED RESULTS:\n\n"
            for result in successful[:10]:  # Show first 10
                filename = os.path.basename(result['path'])
                results_text += f"📄 {filename}\n"
                results_text += f"   Plate: {result['plate']}\n"
                results_text += f"   Status: {result['status']}\n"
                results_text += f"   Confidence: {result['confidence']:.1%}\n"
                results_text += f"   Tests: {result.get('tests_performed', '?')}\n\n"
            
            if len(successful) > 10:
                results_text += f"... and {len(successful) - 10} more successful results\n"
        
        results_label = tk.Label(scrollable_frame, text=results_text,
                            font=("Consolas", 9),
                            fg=self.colors['text_secondary'],
                            bg=self.colors['bg_tertiary'],
                            justify=tk.LEFT)
        results_label.pack(padx=20, pady=(0, 20), anchor="w")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons
        button_frame = tk.Frame(dialog, bg=self.colors['bg_secondary'])
        button_frame.pack(pady=20)
        
        export_btn = self.create_modern_button(button_frame, "💾 Export Results", 
                                            lambda: self.export_batch_auto_tune_results(all_results),
                                            self.colors['accent_green'], width=150, height=35)
        export_btn.pack(side=tk.LEFT, padx=5)
        
        ok_btn = self.create_modern_button(button_frame, "✅ OK", dialog.destroy,
                                        self.colors['accent_blue'], width=100, height=35)
        ok_btn.pack(side=tk.LEFT, padx=5)

    def export_batch_auto_tune_results(self, results):
        """Export batch auto-tune results."""
        filename = filedialog.asksaveasfilename(
            title="Export Batch Auto-Tune Results",
            defaultextension=".json",
            filetypes=(
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            )
        )
        
        if filename:
            try:
                export_data = {
                    'export_time': datetime.now().isoformat(),
                    'strategy': self.batch_auto_tune_strategy.get(),
                    'mode': self.auto_tune_mode.get(),
                    'total_images': len(results),
                    'successful': len([r for r in results if r.get('success', False)]),
                    'failed': len([r for r in results if not r.get('success', False)]),
                    'results': []
                }
                
                for result in results:
                    export_result = {
                        'filename': os.path.basename(result['path']),
                        'success': result.get('success', False)
                    }
                    
                    if result.get('success'):
                        export_result.update({
                            'plate': result['plate'],
                            'status': result['status'],
                            'confidence': result['confidence'],
                            'combination': result.get('combination', {}),
                            'tests_performed': result.get('tests_performed', 0)
                        })
                    else:
                        export_result['error'] = result.get('error', 'Unknown error')
                    
                    export_data['results'].append(export_result)
                
                if filename.endswith('.json'):
                    with open(filename, 'w') as f:
                        json.dump(export_data, f, indent=2)
                else:  # CSV
                    import csv
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Filename', 'Success', 'Plate', 'Status', 'Confidence', 'Tests'])
                        
                        for result in results:
                            if result.get('success'):
                                writer.writerow([
                                    os.path.basename(result['path']),
                                    'Yes',
                                    result['plate'],
                                    result['status'],
                                    f"{result['confidence']:.2%}",
                                    result.get('tests_performed', 0)
                                ])
                            else:
                                writer.writerow([
                                    os.path.basename(result['path']),
                                    'No',
                                    '',
                                    '',
                                    '',
                                    ''
                                ])
                
                messagebox.showinfo("Success", f"Batch results exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")
        
    def _batch_thread(self, filepaths):
        options = {key: var.get() for key, var in self.preprocessing_options.items()}
        model_name = self.active_model_name.get()
        active_model = self.models.get(model_name)
        if not active_model:
            self.root.after(0, messagebox.showerror, "Model Error", f"Selected model '{model_name}' is not loaded.")
            return

        all_results = []
        for i, path in enumerate(filepaths):
            self.root.after(0, self.update_status, f"Batch processing {i+1}/{len(filepaths)}...")
            _, plate, status, conf, _ = process_image(active_model, self.api, path, options)
            all_results.append({'plate': plate, 'status': status, 'confidence': conf})
        self.root.after(0, self._update_batch_results, all_results)
    
    def _process_batch_thread(self, filepaths):
        """Thread function for batch processing."""
        try:
            options = {key: var.get() for key, var in self.preprocessing_options.items()}
            
            def progress_callback(current, total):
                progress_text = f"Processing {current}/{total} images..."
                self.root.after(0, self.update_status, progress_text)
            
            results = process_image_batch(self, filepaths, options, progress_callback)
            
            # Update UI with batch results
            self.root.after(0, self._update_batch_results, results)
            
        except Exception as e:
            self.root.after(0, self._handle_error, str(e))

    def _update_batch_results(self, results):
        """Update UI with batch processing results."""
        self.hide_progress()
        self.batch_mode = False
        
        # Process results
        successful = [r for r in results if r.get('success', False)]
        failed = [r for r in results if not r.get('success', False)]
        
        # Add successful results to history
        for result in successful:
            self.add_to_history(result['plate'], result['status'], result['confidence'])
            self.update_statistics(result['status'])
        
        # Show summary
        summary = f"Batch processing complete\n"
        summary += f"✅ Successful: {len(successful)}\n"
        summary += f"❌ Failed: {len(failed)}\n"
        
        if failed:
            summary += "\nFailed files:\n"
            for fail in failed[:5]:  # Show first 5 failed files
                summary += f"• {os.path.basename(fail['path'])}\n"
            if len(failed) > 5:
                summary += f"• ... and {len(failed)-5} more"
        
        messagebox.showinfo("Batch Processing Complete", summary)
        self.update_status(f"Batch complete • {len(successful)} successful, {len(failed)} failed")

    def reprocess_image(self):
        """Reprocess current image with updated settings."""
        if self.current_image_path:
            self.process_single_image_with_auto_tune()

    def display_images(self, original_frame, processed_frame):
        """Display original and processed images with enhanced styling."""
        self.display_single_image(original_frame, self.original_image_label, "Original")
        self.display_single_image(processed_frame, self.processed_image_label, "Processed")

    def display_single_image(self, frame, label, image_type):
        """Display a single image with proper scaling and styling."""
        try:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            
            # Get optimal size for display
            label.update()
            max_width = max(label.winfo_width() - 30, 400)
            max_height = max(label.winfo_height() - 30, 300)
            
            # Resize maintaining aspect ratio
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # Create and display photo
            photo = ImageTk.PhotoImage(image=img)
            label.config(image=photo, text="")
            label.image = photo
            
        except Exception as e:
            label.config(text=f"Error displaying {image_type.lower()} image:\n{str(e)}")

    def update_detection_results(self, plate_text, status, confidence):
        """Update the detection results display."""
        # Update plate display
        self.plate_display.config(text=plate_text if plate_text != "No Plate Detected" else "─────")
        
        # Update status with colors
        if status == "Registered":
            self.status_display.config(text="✅ REGISTERED", fg=self.colors['accent_green'])
        elif status == "Not Registered":
            self.status_display.config(text="❌ NOT REGISTERED", fg=self.colors['accent_red'])
        else:
            self.status_display.config(text="❓ UNKNOWN", fg=self.colors['text_muted'])
        
        # Update confidence
        if confidence is not None:
            conf_percent = int(confidence * 100)
            self.confidence_display.config(text=f"{conf_percent}%")
        else:
            self.confidence_display.config(text="─")

    def add_to_history(self, plate_text, status, confidence):
        """Add detection to history with enhanced data."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        conf_text = f"{int(confidence*100)}%" if confidence else "N/A"
        
        # Add to treeview
        status_icon = "✅" if status == "Registered" else "❌" if status == "Not Registered" else "❓"
        self.history_tree.insert('', 0, values=(timestamp, plate_text, f"{status_icon} {status}", conf_text))
        
        # Add to internal history
        self.detection_history.insert(0, {
            'timestamp': datetime.now(),
            'plate': plate_text,
            'status': status,
            'confidence': confidence,
            'image_path': self.current_image_path
        })
        
        # Limit history size
        if len(self.detection_history) > 100:
            self.detection_history = self.detection_history[:100]
            # Remove excess items from treeview
            items = self.history_tree.get_children()
            for item in items[100:]:
                self.history_tree.delete(item)

    def update_statistics(self, status):
        """Update detection statistics."""
        self.statistics['total'] += 1
        
        if status == "Registered":
            self.statistics['registered'] += 1
        elif status == "Not Registered":
            self.statistics['not_registered'] += 1
        
        # Calculate success rate
        if self.statistics['total'] > 0:
            detected = self.statistics['registered'] + self.statistics['not_registered']
            success_rate = (detected / self.statistics['total']) * 100
        else:
            success_rate = 0
        
        # Update display
        self.total_stat.config(text=str(self.statistics['total']))
        self.registered_stat.config(text=str(self.statistics['registered']))
        self.not_registered_stat.config(text=str(self.statistics['not_registered']))
        self.success_rate_stat.config(text=f"{success_rate:.1f}%")

    def clear_history(self):
        """Clear detection history."""
        if messagebox.askyesno("Clear History", "Clear all detection history and statistics?"):
            self.history_tree.delete(*self.history_tree.get_children())
            self.detection_history.clear()
            self.statistics = {'total': 0, 'registered': 0, 'not_registered': 0}
            self.update_statistics_display()
            self.update_status("History cleared")

    def update_statistics_display(self):
        """Update statistics display."""
        self.total_stat.config(text="0")
        self.registered_stat.config(text="0")
        self.not_registered_stat.config(text="0")
        self.success_rate_stat.config(text="0%")

    def export_results(self):
        """Export detection history to file."""
        if not self.detection_history:
            messagebox.showwarning("No Data", "No detection history to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Detection History",
            defaultextension=".json",
            filetypes=(
                ("JSON files", "*.json"),
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            )
        )
        
        if filename:
            try:
                if filename.endswith('.json'):
                    self.export_json(filename)
                elif filename.endswith('.csv'):
                    self.export_csv(filename)
                else:
                    self.export_txt(filename)
                
                messagebox.showinfo("Success", f"History exported to {filename}")
                self.update_status(f"History exported • {len(self.detection_history)} records")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export history:\n{str(e)}")

    def export_json(self, filename):
        """Export history as JSON."""
        export_data = {
            'export_time': datetime.now().isoformat(),
            'statistics': self.statistics,
            'detections': []
        }
        
        for record in self.detection_history:
            export_data['detections'].append({
                'timestamp': record['timestamp'].isoformat(),
                'plate': record['plate'],
                'status': record['status'],
                'confidence': record['confidence'],
                'image_path': record['image_path']
            })
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

    def export_csv(self, filename):
        """Export history as CSV."""
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'License Plate', 'Status', 'Confidence', 'Image Path'])
            
            for record in self.detection_history:
                writer.writerow([
                    record['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    record['plate'],
                    record['status'],
                    f"{record['confidence']:.2%}" if record['confidence'] else "N/A",
                    record['image_path']
                ])

    def export_txt(self, filename):
        """Export history as formatted text."""
        with open(filename, 'w') as f:
            f.write("LICENSE PLATE DETECTION HISTORY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Detections: {self.statistics['total']}\n")
            f.write(f"Registered Plates: {self.statistics['registered']}\n")
            f.write(f"Not Registered: {self.statistics['not_registered']}\n\n")
            f.write("DETECTION RECORDS:\n")
            f.write("-" * 50 + "\n")
            
            for record in self.detection_history:
                f.write(f"Time: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Plate: {record['plate']}\n")
                f.write(f"Status: {record['status']}\n")
                f.write(f"Confidence: {record['confidence']:.2%}\n" if record['confidence'] else "Confidence: N/A\n")
                f.write(f"Image: {record['image_path']}\n")
                f.write("-" * 30 + "\n")

    def auto_save_result(self, plate_text, status, confidence):
        """Automatically save detection result."""
        if not os.path.exists("auto_saves"):
            os.makedirs("auto_saves")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"auto_saves/detection_{timestamp}.txt"
        
        try:
            with open(filename, 'w') as f:
                f.write(f"Auto-saved Detection Result\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"License Plate: {plate_text}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Confidence: {confidence:.2%}\n" if confidence else "Confidence: N/A\n")
                f.write(f"Source Image: {self.current_image_path}\n")
        except:
            pass  # Silent fail for auto-save

    def show_progress(self, message):
        """Show progress indication."""
        self.progress_bar.pack(side=tk.RIGHT, padx=10, pady=3)
        self.progress_bar.start(10)
        self.update_status(message)

    def hide_progress(self):
        """Hide progress indication."""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

    def update_status(self, message, error=False):
        """Update status bar message."""
        if error:
            self.status_text.config(text=f"❌ {message}", fg=self.colors['accent_red'])
        else:
            self.status_text.config(text=f"✓ {message}", fg=self.colors['text_secondary'])

    def load_settings(self):
        """Load application settings."""
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", 'r') as f:
                    settings = json.load(f)
                    self.auto_save_enabled.set(settings.get('auto_save', False))
        except:
            pass

    def save_settings(self):
        """Save application settings."""
        try:
            settings = {
                'auto_save': self.auto_save_enabled.get(),
                'window_geometry': self.root.geometry()
            }
            with open("settings.json", 'w') as f:
                json.dump(settings, f)
        except:
            pass

    def _handle_error(self, error_message):
        """Handle errors in main thread."""
        self.hide_progress()
        self.update_status("Processing failed", error=True)
        messagebox.showerror("Error", f"An error occurred:\n{error_message}")

    def __del__(self):
        """Save settings on app close."""
        try:
            self.save_settings()
        except:
            pass

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = AdvancedPlateRecognitionApp(root)
    
    # Handle window close
    def on_closing():
        app.save_settings()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()