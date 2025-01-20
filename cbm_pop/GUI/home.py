import os
import glob
import csv
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QMainWindow,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QGroupBox,
    QProgressBar,
    QSpinBox,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import subprocess
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CBM-POP Manager")

        # Main layout
        main_layout = QGridLayout()

        # Button for running CBM-POP (Bottom Right)
        self.button = QPushButton("Run CBM-POP")
        self.button.setCheckable(True)
        self.button.clicked.connect(self.start_progress)

        # Progress bar (Bottom Left)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Form sections
        self.agent_input = QSpinBox()  # Save reference to agent input
        self.runtime_input = QSpinBox()  # Save reference to runtime input
        agent_group = self.create_agent_form_group("Agent Configuration")
        tracker_group = self.create_tracker_form_group("Tracker Configuration")

        # Matplotlib canvas (Center area)
        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_title("CBM-POP Visualization")

        # Layout setup
        main_layout.addWidget(agent_group, 0, 1)  # Top right
        main_layout.addWidget(tracker_group, 1, 1)  # Below agent config
        main_layout.addWidget(self.canvas, 0, 0, 2, 1)  # Center (spanning 2 rows)
        main_layout.addWidget(self.progress_bar, 2, 0)  # Bottom left
        main_layout.addWidget(self.button, 2, 1, 1, 1, Qt.AlignmentFlag.AlignRight)  # Bottom right

        # Set column and row stretch
        main_layout.setColumnStretch(0, 3)  # Make the plot's column stretchable
        main_layout.setColumnStretch(1, 1)  # Keep the config column fixed in width
        main_layout.setRowStretch(0, 1)     # Allow the rows to stretch
        main_layout.setRowStretch(1, 1)
        main_layout.setRowStretch(2, 0)     # Bottom row (button and progress bar) doesn't stretch

        # Central widget and layout
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Timer for progress bar updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)

        # Variables for progress tracking
        self.runtime = 0
        self.elapsed_time = 0

    def create_agent_form_group(self, title):
        group_box = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        label1 = QLabel("Number of Agents:")
        self.agent_input.setRange(1, 100)  # Set range for agent input
        layout.addWidget(label1)
        layout.addWidget(self.agent_input)

        group_box.setLayout(layout)
        return group_box

    def create_tracker_form_group(self, title):
        group_box = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        label1 = QLabel("Run time:")
        self.runtime_input.setRange(1, 1000)  # Set range for runtime input (e.g., in seconds)
        layout.addWidget(label1)
        layout.addWidget(self.runtime_input)

        group_box.setLayout(layout)
        return group_box

    def start_progress(self):
        # Get the values from the input boxes
        instance_count = self.agent_input.value()
        self.runtime = self.runtime_input.value()
        self.elapsed_time = 0

        # Run the external process in a separate thread to avoid blocking the GUI
        subprocess.Popen(["./resources/launch_5_agents.sh", str(instance_count), str(float(self.runtime))])
        # Reset and start progress bar
        self.progress_bar.setValue(0)
        # Wait for 2 seconds before starting the progress bar updates
        QTimer.singleShot(3000, self.start_timer)

    def start_timer(self):
        # Start the progress timer to update every 100ms (0.1 seconds)
        self.timer.start(100)

    def update_progress(self):
        self.elapsed_time += 0.1  # Increment elapsed time by timer interval (0.1 seconds)

        # Calculate progress percentage
        progress_percentage = (self.elapsed_time / self.runtime) * 100
        self.progress_bar.setValue(min(int(progress_percentage), 100))

        # Stop the timer when progress reaches 100%
        if self.elapsed_time >= self.runtime:
            self.timer.stop()
            self.display_latest_run_log()

    def display_latest_run_log(self):
        # Find the most recent run log
        log_files = glob.glob("resources/run_logs/fitness_logs_*.csv")
        if not log_files:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No run logs found", horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return

        latest_file = max(log_files, key=os.path.getctime)  # Get the most recent file

        # Read the data from the log file
        times, fitness_values = [], []
        try:
            with open(latest_file, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header
                for row in reader:
                    times.append(float(row[0]))
                    fitness_values.append(float(row[1]))
        except Exception as e:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Error reading log: {e}", horizontalalignment='center', verticalalignment='center')
            self.canvas.draw()
            return

        # Plot the data on the Matplotlib canvas
        self.ax.clear()
        self.ax.plot(times, fitness_values, marker='o', linestyle='-')
        self.ax.set_title("Fitness Values Over Time")
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Fitness Value")
        self.ax.grid(True)
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    # Start the event loop.
    app.exec()
