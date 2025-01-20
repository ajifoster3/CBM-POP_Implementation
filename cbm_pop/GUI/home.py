import subprocess

from PyQt6.QtCore import Qt
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
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CBM-POP Manager")

        # Main layout
        main_layout = QGridLayout()

        # Button for running CBM-POP (Bottom Right)
        button = QPushButton("Run CBM-POP")
        button.setCheckable(True)
        button.clicked.connect(self.the_button_was_clicked)

        # Progress bar (Bottom Left)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        # Form sections
        self.agent_input = QSpinBox()  # Save reference to agent input
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
        main_layout.addWidget(button, 2, 1, 1, 1, Qt.AlignmentFlag.AlignRight)  # Bottom right

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
        tracker_input = QSpinBox()
        tracker_input.setRange(1, 100)  # Set range for tracker input
        layout.addWidget(label1)
        layout.addWidget(tracker_input)

        group_box.setLayout(layout)
        return group_box

    def the_button_was_clicked(self):
        # Get the value from the agent input box
        instance_count = self.agent_input.value()

        # Use subprocess to run the script with the number of agents as a parameter
        subprocess.run(["./resources/launch_5_agents.sh", str(instance_count)])

        # Simulate progress bar update
        for i in range(101):
            self.progress_bar.setValue(i)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    # Start the event loop.
    app.exec()
