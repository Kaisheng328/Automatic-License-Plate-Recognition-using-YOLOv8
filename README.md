# AI License Plate Recognition Pro 🚗

This is an advanced computer vision application that uses a YOLOv8 model to detect license plates in images and the EasyOCR library to read the plate numbers. The application features a user-friendly graphical interface (GUI) built with Tkinter, allowing for real-time image processing, smart enhancements, and analytics.



---

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### **Prerequisites**

Before you begin, make sure you have the following installed on your system:

1.  **Python:** This project requires Python 3.8 or newer. You can download it from [python.org](https://www.python.org/downloads/).
2.  **Git:** You'll need Git to clone the repository. You can download it from [git-scm.com](https://git-scm.com/downloads).

---

## Setup Instructions ⚙️

Follow these steps carefully to set up the project environment.

### **Step 1: Clone the Repository**

First, open your terminal or command prompt, navigate to the directory where you want to store the project, and clone the repository using the following command:

```bash
git clone <your-repository-url>
cd <repository-folder-name>
```

### **Step 2: Create a Virtual Environment**

It is highly recommended to use a virtual environment to keep the project's dependencies isolated. Run the following command inside the project folder to create an environment named `.venv`:

```bash
python -m venv .venv
```

### **Step 3: Activate the Virtual Environment**

You must activate the environment before installing dependencies. The command differs based on your operating system.

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

### **Step 4: Install Required Libraries**

With the virtual environment active, install all the necessary Python libraries from the `requirements.txt` file using pip:

```bash
pip install -r requirements.txt
```
This command will automatically download and install all the dependencies, such as `ultralytics`, `easyocr`, `opencv-python`, and `Pillow`.

**Note:** The first time you run the application, `easyocr` will automatically download its language models. This may take a minute and requires an internet connection.

---

## Running the Application ▶️

Once the setup is complete, you can run the application with the following command:

```bash
python app.py
```

The application window should appear, and you are now ready to start processing images!

---

## How to Use the Application

1.  **Load an Image:** Click the **"📂 Select Image"** button to open a file dialog and choose a picture of a car.
2.  **Apply Enhancements (Optional):** Use the checkboxes under **"⚙️ Smart Image Enhancement"** to apply filters that may improve detection in difficult conditions (e.g., low light, rain).
3.  **View Results:** The application will automatically process the image.
    * The **"🔍 Processed & Detected"** panel will show the image with a bounding box around the detected plate.
    * The **"📊 Detection Results"** card will display the recognized plate number, its registration status, and the model's confidence level.
4.  **Check History:** All detections are automatically added to the **"📝 Detection History"** list for review.
5.  **Reprocess:** If you change any enhancement settings, click the **"🔄 Reprocess"** button to run the detection again on the current image.

---

## Project Structure

* `app.py`: The main Python script that contains all the application logic and GUI code.
* `best.pt`: The pre-trained custom YOLOv8 model file used for license plate detection.
* `requirements.txt`: A list of all Python dependencies required to run the project.
* `.gitignore`: A file that tells Git to ignore the `.venv` folder.
