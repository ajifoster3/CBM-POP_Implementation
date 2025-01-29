import rclpy
from rclpy.node import Node
import asyncio

from cbm_pop.reevo import reevo_config
from cbm_pop.reevo.PopulationGenerator import PopulationGenerator

class LLMInterfaceAgent(Node):
    def __init__(self, node_name, agent_id):
        super().__init__(node_name)
        self.agent_id = agent_id
        self.population_generator = PopulationGenerator()

async def generate_population(node):
    # Placeholder async method to simulate population generation
    print("Generating population...")
    population = await node.population_generator.generate_population(10)
    for individual in population[reevo_config.function_name["TWO_SWAP"]]:
        print(individual)
    


def main():
    rclpy.init(args=None)
    temp_node = Node("parameter_loader")
    agent_id = temp_node.declare_parameter('agent_id', '0').value  # Replace 'default_agent_id' with the default value
    temp_node.destroy_node()
    agent = LLMInterfaceAgent('llm_interface_agent_node', agent_id)
    print("LLMInterfaceAgent has been initialized.")
    asyncio.run(generate_population(agent))
    try:
        rclpy.spin(agent)
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        if rclpy.ok():  # Prevent double shutdown
            rclpy.shutdown()
