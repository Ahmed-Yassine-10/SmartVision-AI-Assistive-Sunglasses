import sys
import cv2
import time
import json
import threading
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
                             QTextEdit, QGroupBox, QProgressBar, QMessageBox, QHeaderView,
                             QFrame, QSplitter)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor, QIcon
import google.generativeai as genai

# === Configuration ===
API_KEY = ""
CAMERA_URL = "10.46.122.136"
IMG_PATH = "evive_image.jpg"

# === Gemini Setup ===
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

PROMPT = """
You are an intelligent assistant for the E-Vive swarm robotics system. Your job is to analyze an image of an electronic waste area and simulate how the robot swarm would act.

Tasks:
1. Identify and count visible electronic components such as chips, motors, sensors, wires.
2. For each identified component, classify it based on visible condition:
   - Good condition -> Reuse
   - Damaged but possibly repairable -> Repair
   - Severely damaged or burned -> Recycle
3. Provide a summary of how the E-Vive Collector bots and the Sorter bot would handle this image.

Return ONLY valid JSON using this format:
{
  "component_summary": "Total X components (e.g. 2 chips, 1 motor, 3 wires)",
  "component_distribution": {
    "Reuse": [ "chip", "motor" ],
    "Repair": [ "sensor" ],
    "Recycle": [ "wire", "board" ]
  },
  "swarm_action": "Collector bots will extract reusable and repairable items and deliver to Sorter bot.",
  "sorter_decision": "Sorter will place components into 3 bins accordingly.",
  "reasoning": "Explain how decisions were made based on the condition and component type."
}

Do not return any explanation outside of this JSON block.
"""


class GeminiWorker(QThread):
    """Worker thread for Gemini AI analysis"""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            with open(self.image_path, "rb") as img:
                response = model.generate_content([
                    PROMPT,
                    {"mime_type": "image/jpeg", "data": img.read()}
                ])

            raw_text = response.text.strip()

            # Clean up markdown code block if present
            if raw_text.startswith("```json"):
                raw_text = raw_text.strip("```json").strip("```").strip()

            result = json.loads(raw_text)
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class EViveMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.gemini_worker = None

    def initUI(self):
        self.setWindowTitle("E-Vive Swarm Robotics - Component Analysis System")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(self.get_stylesheet())

        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel - Controls and Image
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Results
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([600, 800])

        self.show()

    def create_left_panel(self):
        """Create left panel with controls and image display"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Title
        title_label = QLabel("🤖 E-Vive Swarm Control")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #2ecc71; margin: 10px;")
        left_layout.addWidget(title_label)

        # Control buttons
        controls_group = QGroupBox("Camera Controls")
        controls_layout = QVBoxLayout(controls_group)

        self.capture_btn = QPushButton("📸 Capture Image")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        controls_layout.addWidget(self.capture_btn)

        self.analyze_btn = QPushButton("🧠 Analyze with Gemini AI")
        self.analyze_btn.clicked.connect(self.analyze_image)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("QPushButton { padding: 10px; font-size: 14px; }")
        controls_layout.addWidget(self.analyze_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)

        left_layout.addWidget(controls_group)

        # Image display
        image_group = QGroupBox("Captured Image")
        image_layout = QVBoxLayout(image_group)

        self.image_label = QLabel("No image captured yet")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setStyleSheet("border: 2px dashed #bdc3c7; background-color: #ecf0f1;")
        image_layout.addWidget(self.image_label)

        left_layout.addWidget(image_group)
        left_layout.addStretch()

        return left_widget

    def create_right_panel(self):
        """Create right panel with results table and analysis"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Analysis summary
        summary_group = QGroupBox("🔍 Analysis Summary")
        summary_layout = QVBoxLayout(summary_group)

        self.summary_label = QLabel("No analysis performed yet")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-size: 14px; padding: 10px;")
        summary_layout.addWidget(self.summary_label)

        right_layout.addWidget(summary_group)

        # Components table
        table_group = QGroupBox("📦 Component Classification")
        table_layout = QVBoxLayout(table_group)

        self.components_table = QTableWidget()
        self.components_table.setColumnCount(3)
        self.components_table.setHorizontalHeaderLabels(["♻️ Reuse", "🔧 Repair", "🗑️ Recycle"])

        # Style the table
        header = self.components_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.components_table.setAlternatingRowColors(True)
        self.components_table.setStyleSheet("""
            QTableWidget {
                gridline-color: #bdc3c7;
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #34495e;
                color: white;
                padding: 8px;
                font-weight: bold;
            }
        """)

        table_layout.addWidget(self.components_table)
        right_layout.addWidget(table_group)

        # Swarm action details
        action_group = QGroupBox("🤖 Swarm Action Plan")
        action_layout = QVBoxLayout(action_group)

        self.action_text = QTextEdit()
        self.action_text.setMaximumHeight(150)
        self.action_text.setReadOnly(True)
        self.action_text.setPlaceholderText("Swarm actions will appear here after analysis...")
        action_layout.addWidget(self.action_text)

        right_layout.addWidget(action_group)

        # Reasoning section
        reasoning_group = QGroupBox("💭 AI Reasoning")
        reasoning_layout = QVBoxLayout(reasoning_group)

        self.reasoning_text = QTextEdit()
        self.reasoning_text.setReadOnly(True)
        self.reasoning_text.setPlaceholderText("AI reasoning will appear here after analysis...")
        reasoning_layout.addWidget(self.reasoning_text)

        right_layout.addWidget(reasoning_group)

        return right_widget

    def get_stylesheet(self):
        """Return the application stylesheet"""
        return """
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 5px;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                border-radius: 3px;
            }
        """

    def capture_image(self):
        """Capture image from IP camera"""
        try:
            self.capture_btn.setEnabled(False)
            self.capture_btn.setText("📸 Capturing...")

            cap = cv2.VideoCapture(CAMERA_URL)
            time.sleep(2)
            ret, frame = cap.read()

            if ret:
                cv2.imwrite(IMG_PATH, frame)

                # Display captured image
                pixmap = QPixmap(IMG_PATH)
                scaled_pixmap = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)

                self.analyze_btn.setEnabled(True)
                self.show_message("✅ Image captured successfully!", "success")
            else:
                self.show_message("❌ Failed to capture image. Check camera connection.", "error")

            cap.release()

        except Exception as e:
            self.show_message(f"❌ Camera error: {str(e)}", "error")

        finally:
            self.capture_btn.setEnabled(True)
            self.capture_btn.setText("📸 Capture Image")

    def analyze_image(self):
        """Analyze captured image with Gemini AI"""
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("🧠 Analyzing...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Start Gemini analysis in worker thread
        self.gemini_worker = GeminiWorker(IMG_PATH)
        self.gemini_worker.finished.connect(self.on_analysis_complete)
        self.gemini_worker.error.connect(self.on_analysis_error)
        self.gemini_worker.start()

    def on_analysis_complete(self, result):
        """Handle successful Gemini analysis"""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🧠 Analyze with Gemini AI")

        # Update summary
        self.summary_label.setText(f"🔍 {result['component_summary']}")

        # Update components table
        self.populate_components_table(result['component_distribution'])

        # Update action plan
        action_text = f"🤖 Swarm Action: {result['swarm_action']}\n\n"
        action_text += f"🧠 Sorter Decision: {result['sorter_decision']}"
        self.action_text.setText(action_text)

        # Update reasoning
        self.reasoning_text.setText(result['reasoning'])

        self.show_message("✅ Analysis completed successfully!", "success")

    def on_analysis_error(self, error_msg):
        """Handle Gemini analysis error"""
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🧠 Analyze with Gemini AI")
        self.show_message(f"❌ Analysis failed: {error_msg}", "error")

    def populate_components_table(self, distribution):
        """Populate the components table with classification results"""
        # Find maximum number of items in any category
        max_items = max(len(items) for items in distribution.values()) if distribution else 0
        self.components_table.setRowCount(max_items)

        # Clear existing content
        self.components_table.clearContents()

        # Color mapping for categories
        colors = {
            "Reuse": "#2ecc71",  # Green
            "Repair": "#f39c12",  # Orange
            "Recycle": "#e74c3c"  # Red
        }

        # Populate each column
        for col, (category, items) in enumerate(distribution.items()):
            for row, item in enumerate(items):
                table_item = QTableWidgetItem(item.capitalize())
                table_item.setTextAlignment(Qt.AlignCenter)

                # Set background color based on category
                if category in colors:
                    table_item.setBackground(QColor(colors[category]))
                    table_item.setForeground(QColor("white"))

                self.components_table.setItem(row, col, table_item)

    def show_message(self, message, msg_type="info"):
        """Show status message"""
        msg_box = QMessageBox()

        if msg_type == "success":
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Success")
        elif msg_type == "error":
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Error")
        else:
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Information")

        msg_box.setText(message)
        msg_box.exec_()


def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = EViveMainWindow()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()