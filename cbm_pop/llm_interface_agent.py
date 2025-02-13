import asyncio
import threading

import rclpy
from rclpy.node import Node

from cbm_pop.reevo import reevo_config
from cbm_pop.reevo.PopulationGenerator import PopulationGenerator
from cbm_pop.reevo.FailedOperatorMutation import FailedOperatorMutation
from cbm_pop.reevo.OECrossover import OECrossover
from cbm_pop_interfaces.msg import GeneratedPopulation, FailedOperatorResponse, FailedOperatorRequest, \
    OECrossoverResponse, OECrossoverRequest


class LLMInterfaceAgent(Node):
    def __init__(self, node_name, agent_id):
        super().__init__(node_name)
        self.agent_id = agent_id
        self.population_generator = PopulationGenerator()
        self.failed_operator_queue = []
        self.oe_crossover_queue = []

        self.failed_operator_mutation = FailedOperatorMutation()
        self.oe_crossover = OECrossover()

        self.population_publisher = self.create_publisher(GeneratedPopulation, 'generated_population_' + str(agent_id),
                                                          10)
        self.failed_operator_subscriber = self.create_subscription(FailedOperatorRequest,
                                                                   'failed_operator_request' + str(self.agent_id),
                                                                   self.failed_operator_response_callback,
                                                                   10)
        self.failed_operator_publisher = self.create_publisher(FailedOperatorResponse,
                                                               'failed_operator_response' + str(self.agent_id),
                                                               10)

        self.oe_crossover_publisher = self.create_publisher(OECrossoverResponse,
                                                            'oe_crossover_response' + str(self.agent_id),
                                                            10)

        self.oe_crossover_subscriber = self.create_subscription(OECrossoverRequest,
                                                                'oe_crossover_request' + str(self.agent_id),
                                                                self.oe_crossover_request_callback,
                                                                10)

        self.run_timer = self.create_timer(0.5, self.process_failed_operators_callback)
        self.run_timer = self.create_timer(0.5, self.process_oe_crossover_callback)

        # Create a dedicated event loop for background tasks
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    async def generate_population(self):
        # Placeholder async method to simulate population generation
        print("Generating population...")
        population = await self.population_generator.generate_population(10)

        # Publish the population after generation
        population_msg = GeneratedPopulation()
        for key, field in reevo_config.function_name.items():
            setattr(population_msg, key.lower(), population[field])  # Directly set the list of strings for each key
        self.population_publisher.publish(population_msg)
        print("Population published.")

    async def process_failed_operators(self):
        """Process one failed operator from the queue and return its result."""
        if not self.failed_operator_queue:
            return None  # Nothing to process

        # Get the next failed operator request from the queue
        operator_name, failed_function_index, failed_function, error = self.failed_operator_queue.pop(0)
        self.get_logger().info("Processing failed operator request.")
        # Run the synchronous function in a separate thread
        result = await asyncio.to_thread(
            self.failed_operator_mutation.perform_failed_operator_mutation,
            operator_name.replace("Operator.", "").lower(),
            reevo_config.problem_description["task_allocation"],
            failed_function,
            error
        )
        return result, operator_name, failed_function_index

    def failed_operator_response_callback(self, msg):
        """Callback for incoming failed operator requests."""
        self.get_logger().info("Failed operator request received. Processing...")
        operator_name = msg.operator_name
        failed_function_index = msg.failed_function_index
        failed_function = msg.failed_function
        error = msg.error
        self.failed_operator_queue.append((operator_name, failed_function_index, failed_function, error))

    def process_failed_operators_callback(self):
        """Schedule processing of the failed operator queue in a non-blocking way."""
        if self.failed_operator_queue:
            # Schedule the coroutine in the event loop running in another thread
            future = asyncio.run_coroutine_threadsafe(self.process_failed_operators(), self.loop)
            future.add_done_callback(self.handle_failed_operator_future_result)

    def handle_failed_operator_future_result(self, future):
        """Callback to handle the result of process_failed_operators once it completes."""
        try:
            new_function, operator_name, failed_function_index = future.result()  # Get the result (if any)
            if new_function is not None:
                failed_operator_response = FailedOperatorResponse()
                failed_operator_response.operator_name = operator_name
                failed_operator_response.failed_function_index = failed_function_index
                failed_operator_response.fixed_function = new_function
                self.failed_operator_publisher.publish(failed_operator_response)
        except Exception as e:
            self.get_logger().error(f"Error processing failed operators: {str(e)}")

    async def process_oe_crossover(self):
        if not self.oe_crossover_queue:
            return None  # Nothing to process

        better_function_code, worse_function_code, operator_name, worse_function_index = self.oe_crossover_queue.pop(0)
        self.get_logger().info("Processing oe crossover request.")
        # Run the synchronous function in a separate thread
        result = await asyncio.to_thread(
            self.oe_crossover.perform_oe_crossover,
            reevo_config.problem_description["task_allocation"],
            better_function_code,
            worse_function_code,
            worse_function_index
        )
        return result, operator_name, worse_function_index

    def oe_crossover_request_callback(self, msg):
        better_function_code = msg.better_function_code
        worse_function_code = msg.worse_function_code
        operator_name = msg.operator_name
        worse_function_index = msg.worse_function_index
        self.oe_crossover_queue.append((better_function_code, worse_function_code, operator_name, worse_function_index))

    def process_oe_crossover_callback(self):
        """Schedule processing of the failed operator queue in a non-blocking way."""
        if self.oe_crossover_queue:
            # Schedule the coroutine in the event loop running in another thread
            future = asyncio.run_coroutine_threadsafe(self.process_oe_crossover(), self.loop)
            future.add_done_callback(self.handle_oe_crossover_future_result)

    def handle_oe_crossover_future_result(self, future):
        """Callback to handle the result of process_failed_operators once it completes."""
        try:
            result, operator_name, worse_function_index = future.result()  # Get the result (if any)
            if result is not None:
                oe_crossover_response = OECrossoverResponse()
                oe_crossover_response.crossover_function_code = result
                oe_crossover_response.operator_name = operator_name
                oe_crossover_response.crossover_function_index = worse_function_index
                self.oe_crossover_publisher.publish(oe_crossover_response)

                # Remove the entry with the matching worse_function_index
                self.oe_crossover_queue = [
                    entry for entry in self.oe_crossover_queue if entry[3] != worse_function_index
                ]
        except Exception as e:
            self.get_logger().error(f"Error processing failed operators: {str(e)}")


def main():
    rclpy.init(args=None)
    temp_node = Node("parameter_loader")
    temp_node.declare_parameter("runtime", 60.0)
    temp_node.declare_parameter('agent_id', 0)
    runtime = temp_node.get_parameter("runtime").value
    agent_id = temp_node.get_parameter("agent_id").value

    temp_node.destroy_node()
    agent = LLMInterfaceAgent(f'llm_interface_agent_node_{agent_id}', agent_id)
    print("LLMInterfaceAgent has been initialized.")

    def shutdown_callback():
        agent.get_logger().info("LLM-Interface-agent Runtime completed. Shutting down.")
        agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # Create a timer for shutdown
    agent.create_timer(runtime, shutdown_callback)

    # asyncio.run(agent.generate_population())

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(agent)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
