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
    QSpinBox, QComboBox, QTabWidget, QCheckBox
)
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
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
        self.bulk_run_input = QSpinBox()  # Save reference to bulk run input
        self.bulk_run_checkbox = QCheckBox("Enable Bulk Run")  # Checkbox for bulk run
        self.bulk_run_checkbox.stateChanged.connect(self.toggle_bulk_run_input)
        self.bulk_run_count_record = 0
        self.bulk_run_count = 0

        agent_group = self.create_agent_form_group("Agent Configuration")
        tracker_group = self.create_tracker_form_group("Tracker Configuration")
        run_group = self.create_run_form_group("Run Configuration")

        # Tab widget for plots
        self.tab_widget = QTabWidget()  # Create a tab widget
        # Enable closable tabs
        self.tab_widget.setTabsClosable(True)
        # Connect the tab close signal to the remove_tab method
        self.tab_widget.tabCloseRequested.connect(self.remove_tab)
        self.add_new_tab("Initial Tab")  # Add an initial empty tab for the first run

        # Combine agent, tracker, and run configuration into one layout
        config_layout = QVBoxLayout()
        config_layout.addWidget(agent_group)
        config_layout.addWidget(tracker_group)
        config_layout.addWidget(run_group)
        config_layout.addStretch()  # Add stretch below the forms to push them to the top

        # Create a group box for the combined configuration forms
        config_group = QGroupBox()
        config_group.setLayout(config_layout)

        # Layout setup
        main_layout.addWidget(self.tab_widget, 0, 0, 2, 1)  # Place the tab widget in the left column, spanning 2 rows
        main_layout.addWidget(config_group, 0, 1, 1, 1, Qt.AlignmentFlag.AlignTop)  # Place the config group in the right column
        main_layout.addWidget(self.progress_bar, 2, 0)  # Bottom left
        main_layout.addWidget(self.button, 2, 1, 1, 1, Qt.AlignmentFlag.AlignRight)  # Bottom right

        # Adjust stretch settings
        main_layout.setColumnStretch(0, 3)  # Make the plot's column stretchable
        main_layout.setColumnStretch(1, 1)  # Keep the config column fixed in width
        main_layout.setRowStretch(0, 1)  # Allow the rows with content to stretch
        main_layout.setRowStretch(1, 0)  # Prevent unnecessary stretching
        main_layout.setRowStretch(2, 0)  # Bottom row (button and progress bar) doesn't stretch

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

    def toggle_bulk_run_input(self):
        """Enable or disable the bulk run input based on the checkbox."""
        self.bulk_run_input.setEnabled(self.bulk_run_checkbox.isChecked())

    def remove_tab(self, index):
        """Remove the tab at the specified index."""
        widget = self.tab_widget.widget(index)
        if widget:
            # Remove the widget from the tab
            self.tab_widget.removeTab(index)
            # Delete the widget to free resources
            widget.deleteLater()

    def add_new_tab(self, title, replace_first=False):
        """Add a new tab with a Matplotlib canvas or replace the first tab."""
        if replace_first and self.tab_widget.count() > 0:
            # Replace the first tab
            first_tab = self.tab_widget.widget(0)
            layout = first_tab.layout()

            # Clear existing widgets in the first tab
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            # Create a new Matplotlib canvas for the first tab
            canvas = FigureCanvas(Figure(figsize=(5, 3)))
            toolbar = NavigationToolbar(canvas, self)
            ax = canvas.figure.add_subplot(111)
            ax.set_title(title)

            # Add canvas and toolbar to the layout
            layout.addWidget(toolbar)
            layout.addWidget(canvas)

            # Rename the first tab
            self.tab_widget.setTabText(0, title)

            # Bring the first tab into focus
            self.tab_widget.setCurrentIndex(0)

            return ax, canvas
        else:
            # Add a new tab
            new_tab = QWidget()
            layout = QVBoxLayout()

            # Create a new Matplotlib canvas
            canvas = FigureCanvas(Figure(figsize=(5, 3)))
            toolbar = NavigationToolbar(canvas, self)
            ax = canvas.figure.add_subplot(111)
            ax.set_title(title)

            # Add canvas and toolbar to the layout
            layout.addWidget(toolbar)
            layout.addWidget(canvas)
            new_tab.setLayout(layout)

            # Add the tab to the tab widget
            self.tab_widget.addTab(new_tab, title)

            # Bring the new tab into focus
            self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)

            return ax, canvas

    def create_agent_form_group(self, title):
        group_box = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Number of Agents
        label1 = QLabel("Number of Agents:")
        self.agent_input.setRange(1, 100)  # Set range for agent input
        layout.addWidget(label1)
        layout.addWidget(self.agent_input)

        # Individual Learning Method
        label2 = QLabel("Individual Learning Method:")
        self.learning_method_dropdown = QComboBox()  # Create dropdown for learning methods
        self.learning_method_dropdown.addItems([
            "Ferreira_et_al.",
            "Q-Learning",
            "FEA_LLM",
            "QL_LLM"
        ])
        layout.addWidget(label2)
        layout.addWidget(self.learning_method_dropdown)

        group_box.setLayout(layout)
        return group_box

    def create_tracker_form_group(self, title):
        group_box = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        label1 = QLabel("Run time:")
        self.runtime_input.setRange(-1, 1000)  # Set range for runtime input (e.g., in seconds)
        layout.addWidget(label1)
        layout.addWidget(self.runtime_input)

        group_box.setLayout(layout)
        return group_box

    def create_run_form_group(self, title):
        group_box = QGroupBox(title)
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.bulk_run_input.setRange(1, 100)  # Set range for bulk run number
        self.bulk_run_input.setEnabled(False)  # Initially disabled

        layout.addWidget(self.bulk_run_checkbox)  # Add the checkbox
        layout.addWidget(self.bulk_run_input)  # Add the spin box

        group_box.setLayout(layout)
        return group_box

    def start_progress(self):
        # Get the values from the input boxes
        self.instance_count = self.agent_input.value()
        self.runtime = self.runtime_input.value()
        self.learning_method = self.learning_method_dropdown.currentText()
        self.bulk_run_count = self.bulk_run_input.value() if self.bulk_run_checkbox.isChecked() else 1
        self.bulk_run_count_record = self.bulk_run_input.value() if self.bulk_run_checkbox.isChecked() else 1
        self.elapsed_time = 0

        self.run_process()

    def run_process(self):
        # Run the external process in a separate thread to avoid blocking the GUI
        subprocess.Popen(["./resources/launch_5_agents.sh", str(self.instance_count), str(float(self.runtime)),
                          str(self.learning_method), str(self.bulk_run_count)])
        # Reset and start progress bar
        self.progress_bar.setValue(0)
        self.elapsed_time = 0
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
        if self.elapsed_time >= self.runtime+2:
            self.timer.stop()
            self.display_latest_run_log()
            if self.bulk_run_checkbox.isChecked() and self.bulk_run_count > 1:
                self.bulk_run_count = self.bulk_run_count - 1
                self.run_process()
            elif self.bulk_run_checkbox.isChecked() and self.bulk_run_count == 1:
                self.display_bulk_run_log()

    def display_latest_run_log(self):
        # Find the most recent run log
        log_files = glob.glob("resources/run_logs/fitness_logs_*.csv")
        if not log_files:
            # If no log files, update the initial tab or add a new error tab
            if self.tab_widget.count() == 1 and self.tab_widget.tabText(0) == "Initial Tab":
                ax, canvas = self.add_new_tab("No Run Logs Found", replace_first=True)
            else:
                ax, canvas = self.add_new_tab("No Run Logs Found")
            ax.text(0.5, 0.5, "No run logs found", horizontalalignment='center', verticalalignment='center')
            canvas.draw()
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
            # If there's an error, update the initial tab or add a new error tab
            if self.tab_widget.count() == 1 and self.tab_widget.tabText(0) == "Initial Tab":
                ax, canvas = self.add_new_tab("Error", replace_first=True)
            else:
                ax, canvas = self.add_new_tab("Error")
            ax.text(0.5, 0.5, f"Error reading log: {e}", horizontalalignment='center', verticalalignment='center')
            canvas.draw()
            return

        # Determine if the initial tab should be replaced
        replace_first = (self.tab_widget.count() == 1 and self.tab_widget.tabText(0) == "Initial Tab")

        # Add a new tab for the plot, replacing the first tab only for the initial run
        if replace_first:
            ax, canvas = self.add_new_tab("Run 1", replace_first=True)
        else:
            ax, canvas = self.add_new_tab(f"Run {self.tab_widget.count() + 1}")

        ax.plot(times, fitness_values, marker='o', linestyle='-')
        ax.set_title("Fitness Values Over Time")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Fitness Value")
        ax.grid(True)
        canvas.draw()

    def display_bulk_run_log(self):
        """Aggregate and display binned statistics for bulk run logs."""
        log_files = glob.glob("resources/run_logs/fitness_logs_*.csv")

        if not log_files or len(log_files) < self.bulk_run_count_record:
            self.plot_error("Insufficient run logs for bulk run.")
            return

        # Initialize lists for all times and fitness values across all runs
        all_times, all_fitness_values = [], []

        # Read and combine data from all run logs
        try:
            for log_file in sorted(log_files)[-self.bulk_run_count_record:]:
                with open(log_file, mode='r') as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header
                    for row in reader:
                        all_times.append(float(row[0]))
                        all_fitness_values.append(float(row[1]))
        except Exception as e:
            self.plot_error(f"Error reading logs: {e}")
            return

        # Sort combined data by time
        sorted_data = sorted(zip(all_times, all_fitness_values))
        all_times, all_fitness_values = zip(*sorted_data)

        # Calculate binned averages and standard deviations
        bins = 10  # Adjust the number of bins as needed
        binned_centers, binned_means, binned_stds = self.calculate_binned_statistics(all_times, all_fitness_values,
                                                                                     bins)

        # Add a new tab for the bulk run statistics
        ax, canvas = self.add_new_tab("Bulk Run Statistics")
        ax.errorbar(binned_centers, binned_means, yerr=binned_stds, fmt='o', ecolor='red', capsize=5,
                    label="Binned Data")
        ax.set_title("Binned Fitness Values Over Time (Bulk Runs)")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Fitness Value")
        ax.legend()
        ax.grid(True)
        canvas.draw()

        # Reset bulk run count record
        self.bulk_run_count_record = 0

    def calculate_binned_statistics(self, times, values, bins=10):
        """Calculate binned averages and standard deviations."""
        import numpy as np

        # Define bin edges and determine which bin each time falls into
        bin_edges = np.linspace(min(times), max(times), bins + 1)
        bin_indices = np.digitize(times, bin_edges) - 1

        # Calculate averages and standard deviations for each bin
        binned_means = []
        binned_stds = []
        binned_centers = []
        for i in range(bins):
            bin_values = [values[j] for j in range(len(times)) if bin_indices[j] == i]
            if bin_values:
                binned_means.append(np.mean(bin_values))
                binned_stds.append(np.std(bin_values))
                binned_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

        return binned_centers, binned_means, binned_stds

    def plot_error(self, message):
        """Display an error message in a new tab."""
        ax, canvas = self.add_new_tab("Error")
        ax.text(0.5, 0.5, message, horizontalalignment='center', verticalalignment='center')
        ax.set_title("Error")
        canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    # Start the event loop.
    app.exec()
