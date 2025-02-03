import asyncio
import rclpy
from rclpy.node import Node

from cbm_pop.reevo import reevo_config
from cbm_pop.reevo.PopulationGenerator import PopulationGenerator
from cbm_pop_interfaces.msg import GeneratedPopulation


class LLMInterfaceAgent(Node):
    def __init__(self, node_name, agent_id):
        super().__init__(node_name)
        self.agent_id = agent_id
        self.population_generator = PopulationGenerator()
        self.population_publisher = self.create_publisher(GeneratedPopulation, 'generated_population_'+str(agent_id), 10)

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


def main():
    rclpy.init(args=None)
    temp_node = Node("parameter_loader")
    temp_node.declare_parameter("runtime", 60.0)
    temp_node.declare_parameter('agent_id', 0)  # Replace 'default_agent_id' with the default value
    runtime = temp_node.get_parameter("runtime").value
    agent_id = temp_node.get_parameter("agent_id").value

    temp_node.destroy_node()
    agent = LLMInterfaceAgent('llm_interface_agent_node', agent_id)
    print("LLMInterfaceAgent has been initialized.")

    def shutdown_callback():
        agent.get_logger().info("LLM-Interface-agent Runtime completed. Shutting down.")
        agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # Create a timer with the specified runtime
    agent.create_timer(runtime, shutdown_callback)

    # Run the asynchronous population generation task
    asyncio.run(agent.generate_population())

    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        if rclpy.ok():  # Prevent double shutdown
            rclpy.shutdown()


if __name__ == '__main__':
    main()