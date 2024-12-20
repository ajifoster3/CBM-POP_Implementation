import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re


class FitnessLogger(Node):
    def __init__(self):
        super().__init__('fitness_logger')

        # ROS2 loglarını dinlemek için abone oluyoruz
        self.get_logger().add_handler(self.log_handler)

        # Fitness değerlerini depolayacağız
        self.iteration_logs = []

        # Matplotlib setup
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-', marker='o', label="Fitness Value")
        self.ax.set_xlim(0, 100)  # X ekseni (iterasyon sayısı)
        self.ax.set_ylim(0, 1)  # Y ekseni (fitness aralığı)
        self.ax.set_title("Fitness Over Iterations (Live)")
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("Fitness Value")
        self.ax.grid(True)
        self.ax.legend()

        # Canlı grafiği güncelleme fonksiyonu
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)

    def log_handler(self, msg):
        # Log mesajını alıyoruz
        if 'Current best solution fitness_QL' in msg:
            # Fitness değerini regex ile çekiyoruz
            match = re.search(r"fitness_QL = (\d+\.\d+)", msg)
            if match:
                fitness_value = float(match.group(1))
                # Fitness değerini kaydet
                self.iteration_logs.append((len(self.iteration_logs) + 1, fitness_value))

    def update_plot(self, frame):
        # Verileri güncelle
        if len(self.iteration_logs) > 0:
            iterations = [log[0] for log in self.iteration_logs]
            fitness_values = [log[1] for log in self.iteration_logs]
            self.line.set_data(iterations, fitness_values)
            self.ax.relim()
            self.ax.autoscale_view()
        return self.line,


def main(args=None):
    rclpy.init(args=args)
    fitness_logger = FitnessLogger()

    # ROS 2 spin (logları dinlemek için)
    rclpy.spin(fitness_logger)

    fitness_logger.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
