## 🎯 Use Cases and Applications

### **Video Processing System:**
- **Traffic Monitoring**: Real-time vehicle identification and tracking in traffic intersections
- **Highway Surveillance**: Long-term vehicle tracking across multiple camera zones  
- **Traffic Analysis**: Vehicle flow patterns, speed analysis, and congestion monitoring
- **Law Enforcement**: Automated vehicle tracking for investigation and evidence collection
- **Smart City Integration**: Integration with traffic management and urban planning systems

### **GUI Application:**
- **Parking Management**: Quick license plate verification for entry/exit systems
- **Security Checkpoints**: Manual verification and logging at facility entrances
- **Evidence Processing**: Forensic analysis of license plates from images
- **Mobile Applications**: Integration into mobile apps for field work
- **Quality Control**: Testing and validation of license plate images

## 📈 Performance Characteristics

### **Video Processing Performance:**
- **Detection Accuracy**: High accuracy for vehicles and license plates in good lighting conditions
- **Tracking Consistency**: SORT algorithm provides robust multi-object tracking
- **Processing Speed**: Optimized for near real-time processing on modern hardware
- **Data Quality**: Interpolation algorithms ensure smooth tracking data

### **GUI Application Performance:**
- **Real-time Processing**: Instant license plate recognition upon image load
- **Enhancement Filters**: Smart image processing for challenging conditions
- **User Experience**: Responsive interface with immediate feedback
- **Detection Reliability**: High accuracy with confidence scoring

## 🔧 Technical Specifications

### **Video Processing System:**
- **Vehicle Detection**: YOLOv8n (Nano) for efficient real-time detection
- **License Plate Detection**: Custom YOLOv8 model trained on specialized dataset
- **Tracking Algorithm**: SORT (Simple Online and Realtime Tracking)
- **Output Formats**: CSV data files and annotated MP4 video
- **Supported Input**: MP4, AVI, and other common video formats

### **GUI Application:**
- **Detection Engine**: Custom YOLOv8 model (best.pt)
- **OCR Engine**: EasyOCR for character recognition
- **Image Processing**: OpenCV and Pillow for enhancement
- **Interface**: Tkinter-based desktop application
- **Supported Formats**: JPEG, PNG, BMP, and other common image formats

## 📝 Notes and Limitations

### **General Limitations:**
- **Lighting Conditions**: Performance may vary in low-light or harsh lighting conditions
- **Plate Orientation**: Works best with front-facing license plates
- **Image/Video Quality**: Higher resolution input provides better detection accuracy
- **Processing Requirements**: Performance depends on hardware capabilities

### **Video Processing Specific:**
- **Processing Time**: Video processing speed depends on length and resolution
- **Memory Usage**: Large videos may require significant RAM for processing
- **Real-time Constraints**: Not optimized for live streaming applications

### **GUI Application Specific:**
- **Model Loading**: Initial startup may take time to load detection models
- **Enhancement Processing**: Some filters may increase processing time
- **Single Image Focus**: Designed for individual image processing, not batch operations

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system's functionality and performance. When contributing:

- **Video Processing**: Focus on detection accuracy, tracking improvements, and performance optimization
- **GUI Application**: Enhance user experience, add new enhancement filters, or improve detection algorithms
- **Documentation**: Help improve setup instructions, usage guides, or technical documentation

## 📄 License

This project is provided for educational and research purposes. Please ensure compliance with local regulations regarding automated license plate recognition systems.

---

**Note**: This system is designed for educational and research purposes. Ensure appropriate permissions and compliance with local privacy laws when using in production environments.# AI License Plate Recognition Pro 🚗

This is an advanced **Automatic License Plate Recognition (ALPR)** system that combines state-of-the-art YOLOv8 deep learning architecture with comprehensive computer vision capabilities. The system provides both real-time video processing and individual image analysis through a user-friendly graphical interface (GUI), featuring end-to-end vehicle detection, license plate detection, tracking, and character recognition.

---

## 📋 System Overview

This project implements two main components:

### 🎬 **Video Processing System** 
A comprehensive video-based ALPR system using YOLOv8 for automated vehicle tracking and license plate recognition in traffic scenarios.

### 🖼️ **GUI Application** 
An interactive desktop application with Tkinter interface for real-time image processing, smart enhancements, and instant license plate recognition.

### 🎯 Key Features

- **Multi-Stage Detection Pipeline**: Combines vehicle detection and license plate detection for robust performance
- **Real-time Object Tracking**: Implements SORT (Simple Online and Realtime Tracking) algorithm for consistent vehicle tracking across frames
- **Interactive GUI Interface**: User-friendly desktop application for immediate image processing
- **Smart Image Enhancement**: Built-in filters for improving detection in challenging conditions
- **Data Interpolation**: Advanced missing data interpolation to ensure smooth tracking and reduce detection gaps
- **Visualization System**: Comprehensive output visualization with bounding boxes, tracking IDs, and license plate text
- **Detection History**: Complete logging of all recognition results
- **CSV Output Generation**: Structured data export for further analysis and integration

## 🏗️ System Architecture

### Core Components

#### **Video Processing Pipeline:**

1. **Vehicle Detection Module** (`main.py`)
   - Uses pre-trained YOLOv8n model for vehicle detection
   - Identifies cars, trucks, buses, and motorcycles in video frames
   - Provides bounding box coordinates for detected vehicles

2. **License Plate Detection Module**
   - Custom-trained YOLOv8 model specifically for license plate detection
   - Trained on a comprehensive license plate dataset from Roboflow Universe
   - Optimized for various lighting conditions and plate orientations

3. **Object Tracking System** (SORT Integration)
   - Maintains consistent tracking IDs for vehicles across video frames
   - Handles temporary occlusions and detection gaps
   - Reduces false positives through temporal consistency

4. **Data Processing Pipeline**
   - **Raw Data Extraction**: Extracts detection data and saves to `test.csv`
   - **Missing Data Interpolation** (`add_missing_data.py`): Fills gaps in tracking data using intelligent interpolation algorithms
   - **Visualization Engine** (`visualize.py`): Generates annotated video output with detection results

#### **GUI Application Components:**

1. **Image Processing Engine** (`app.py`)
   - Real-time license plate detection using custom YOLOv8 model (`best.pt`)
   - EasyOCR integration for optical character recognition
   - Smart image enhancement filters for improved detection accuracy

2. **User Interface System**
   - Tkinter-based GUI for intuitive user interaction
   - Real-time preview and results display
   - Detection history management and analytics

### 🔄 Workflow Process

#### Video Processing Workflow:
```
Input Video → Vehicle Detection → License Plate Detection → Object Tracking → 
Data Export → Missing Data Interpolation → Visualization → Output Video
```

#### GUI Application Workflow:
```
Image Upload → Smart Enhancement (Optional) → License Plate Detection → 
OCR Processing → Results Display → History Logging
```

## 🚀 Detailed System Functionality

### Video Processing System

#### Phase 1: Detection and Tracking
- **Input**: Video file containing vehicles
- **Processing**: 
  - Frame-by-frame vehicle detection using YOLOv8n
  - License plate detection within detected vehicle regions
  - Assignment of unique tracking IDs using SORT algorithm
- **Output**: Raw detection data saved to `test.csv`

#### Phase 2: Data Enhancement
- **Input**: Raw detection data from `test.csv`
- **Processing**:
  - Identifies missing frames in vehicle tracks
  - Applies interpolation algorithms to estimate missing positions
  - Smooths trajectory data for consistent tracking
- **Output**: Enhanced dataset with interpolated values

#### Phase 3: Visualization and Results
- **Input**: Enhanced tracking data and original video
- **Processing**:
  - Overlays bounding boxes on detected vehicles and license plates
  - Displays tracking IDs for consistent vehicle identification
  - Shows recognized license plate text (when available)
- **Output**: Annotated video with complete detection and tracking information

### GUI Application System

#### Image Processing Features:
- **Smart Enhancement Options**: Brightness adjustment, contrast enhancement, noise reduction
- **Real-time Detection**: Instant license plate recognition upon image load
- **Confidence Scoring**: Detection confidence levels for reliability assessment
- **Registration Status**: Automatic status determination based on recognition results

#### User Interface Features:
- **Drag-and-Drop Support**: Easy image loading functionality
- **Live Preview**: Real-time processed image display with detection overlays
- **Results Panel**: Comprehensive detection information display
- **History Management**: Complete log of all detection sessions

## 📊 Output Data Structure

### Video Processing Output:
The system generates CSV files containing:
- **Frame Number**: Video frame index
- **Vehicle ID**: Unique tracking identifier
- **Vehicle Coordinates**: Bounding box coordinates (x1, y1, x2, y2)
- **License Plate Coordinates**: Plate bounding box coordinates
- **Detection Confidence**: Model confidence scores
- **License Plate Text**: Recognized characters (when applicable)

### GUI Application Output:
- **Visual Results**: Processed images with bounding box overlays
- **Text Recognition**: Extracted license plate characters
- **Confidence Metrics**: Detection and OCR confidence scores
- **Detection History**: Timestamped log of all processed images

## 🎥 Demo Resources

- **Sample Video**: [Download Demo Video](https://drive.google.com/file/d/1JbwLyqpFCXmftaJY1oap8Sa6KfjoWJta/view?usp=sharing)
- **Pre-trained License Plate Model**: [Download Model](https://drive.google.com/file/d/1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw/view?usp=sharing)
- **Training Dataset**: [Roboflow License Plate Dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)

---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### **Prerequisites**

Before you begin, make sure you have the following installed on your system:

1. **Python:** This project requires Python 3.8 or newer (Python 3.10 recommended for video processing). You can download it from [python.org](https://www.python.org/downloads/).
2. **Git:** You'll need Git to clone the repository. You can download it from [git-scm.com](https://git-scm.com/downloads).
3. **Conda:** Required for video processing components. You can download it from [conda.io](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

---

## Setup Instructions ⚙️

Choose the setup method based on which components you want to use:

### **Option A: Complete System Setup (Video Processing + GUI)**

#### **Step 1: Clone Required Repositories**

First, open your terminal or command prompt, navigate to the directory where you want to store the project, and clone the repositories:

```bash
# Clone this repository
git clone https://github.com/Kaisheng328/Automatic-License-Plate-Recognition-using-YOLOv8.git
cd Automatic-License-Plate-Recognition-using-YOLOv8

# Clone SORT tracking module (required for video processing)
git clone https://github.com/abewley/sort
```

#### **Step 2: Environment Setup for Video Processing**

Create a Conda environment for the video processing components:

```bash
# Create Python 3.10 environment for video processing
conda create --prefix ./env python==3.10 -y

# Activate the environment
source activate ./env
```

#### **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

#### **Step 4: Download Required Models**
- Download the pre-trained license plate detection model from the provided link
- Place the model file in the project root directory
- Ensure the sample video is available for testing

### **Option B: GUI Application Only Setup**

#### **Step 1: Clone the Repository**

```bash
git clone <your-repository-url>
cd <repository-folder-name>
```

#### **Step 2: Create a Virtual Environment**

It is highly recommended to use a virtual environment to keep the project's dependencies isolated:

```bash
python -m venv .venv
```

#### **Step 3: Activate the Virtual Environment**

The command differs based on your operating system:

* **On Windows (using PowerShell):**
    ```powershell
    # If you get an error about script execution being disabled, run this command first:
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

    # Then, activate the environment:
    .\.venv\Scripts\Activate.ps1
    ```

* **On macOS and Linux:**
    ```bash
    source .venv/bin/activate
    ```

After activation, you should see `(.venv)` at the beginning of your terminal prompt.

#### **Step 4: Install Required Libraries**

With the virtual environment active, install all the necessary Python libraries:

```bash
pip install -r requirements.txt
```

**Note:** The first time you run the application, `easyocr` will automatically download its language models. This may take a minute and requires an internet connection.

---

## Running the Applications ▶️

### **🎬 Video Processing System**

Once the setup is complete, you can run the complete video processing pipeline:

#### Running the Complete Pipeline

1. **Initial Detection and Tracking**
   ```bash
   python main.py
   ```
   - Processes the input video
   - Generates `test.csv` with raw detection data
   - May contain gaps in tracking data

2. **Data Interpolation and Enhancement**
   ```bash
   python add_missing_data.py
   ```
   - Reads `test.csv`
   - Applies interpolation algorithms to fill missing frames
   - Outputs enhanced dataset for visualization

3. **Generate Final Visualization**
   ```bash
   python visualize.py
   ```
   - Creates annotated output video
   - Shows smooth tracking with interpolated data
   - Displays license plate recognition results

#### Expected Results
- **Raw Output**: Detection data with potential gaps
- **Enhanced Output**: Smooth tracking with interpolated missing frames
- **Final Video**: Annotated video showing vehicle tracking, license plate detection, and recognized text

### **🖼️ GUI Application**

Run the interactive desktop application:

```bash
python app.py
```

The application window should appear, and you are now ready to start processing images!

---

## How to Use the Applications

### **🎬 Video Processing System Usage**

1. **Prepare Input**: Place your video file in the project directory
2. **Run Pipeline**: Execute the three-step process (main.py → add_missing_data.py → visualize.py)
3. **View Results**: Check the generated CSV files and annotated video output
4. **Analyze Data**: Use the CSV output for further traffic analysis or integration with other systems

### **🖼️ GUI Application Usage**

1. **Load an Image:** Click the **"📂 Select Image"** button to open a file dialog and choose a picture of a car.
2. **Apply Enhancements (Optional):** Use the checkboxes under **"⚙️ Smart Image Enhancement"** to apply filters that may improve detection in difficult conditions (e.g., low light, rain).
3. **View Results:** The application will automatically process the image.
   * The **"🔍 Processed & Detected"** panel will show the image with a bounding box around the detected plate.
   * The **"📊 Detection Results"** card will display the recognized plate number, its registration status, and the model's confidence level.
4. **Check History:** All detections are automatically added to the **"📝 Detection History"** list for review.
5. **Reprocess:** If you change any enhancement settings, click the **"🔄 Reprocess"** button to run the detection again on the current image.

## 🎯 Use Cases and Applications

- **Traffic Monitoring**: Real-time vehicle identification and tracking
- **Parking Management**: Automated entry/exit logging
- **Security Systems**: Vehicle access control and monitoring
- **Law Enforcement**: License plate recognition for investigation
- **Traffic Analysis**: Vehicle flow and pattern analysis

## 📈 Performance Characteristics

- **Detection Accuracy**: High accuracy for vehicles and license plates in good lighting conditions
- **Tracking Consistency**: SORT algorithm provides robust multi-object tracking
- **Processing Speed**: Optimized for near real-time processing on modern hardware
- **Data Quality**: Interpolation algorithms ensure smooth tracking data

## 🔧 Technical Specifications

- **Vehicle Detection**: YOLOv8n (Nano) for efficient real-time detection
- **License Plate Detection**: Custom YOLOv8 model trained on specialized dataset
- **Tracking Algorithm**: SORT (Simple Online and Realtime Tracking)
- **Output Formats**: CSV data files and annotated MP4 video
- **Supported Input**: MP4, AVI, and other common video formats

---

## Project Structure

### **Video Processing Components:**
* `main.py`: Core detection and tracking script with vehicle and license plate detection
* `add_missing_data.py`: Data interpolation module for filling tracking gaps  
* `visualize.py`: Video annotation and output generation script
* `sort/`: SORT tracking algorithm implementation (cloned repository)
* `test.csv`: Raw detection data output from main processing
* `requirements.txt`: Python dependencies for the complete system

### **GUI Application Components:**
* `app.py`: Main GUI application with Tkinter interface and detection logic
* `best.pt`: Pre-trained custom YOLOv8 model for license plate detection
* `requirements.txt`: Python dependencies for GUI application
* `.gitignore`: Git ignore file for environment folders

### **Shared Resources:**
* Sample video files and demo images
* Pre-trained models and datasets
* Documentation and setup files
