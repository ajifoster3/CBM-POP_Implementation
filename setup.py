from setuptools import find_packages, setup

import cbm_pop

package_name = 'cbm_pop'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=['cbm_pop', 'cbm_pop.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ajifoster3',
    maintainer_email='ajifoster3@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'cbm_fitness_logger = cbm_pop.fitness_logger:main',
            'cbm_fitness_logger_offline = cbm_pop.fitness_logger_offline:main',
            'simple_fitness_logger = cbm_pop.SimpleSimulator.simple_fitness_logger:main',
            'cbm_population_agent_llm = cbm_pop.cbm_population_agent_llm:main',
            'llm_interface_agent = cbm_pop.llm_interface_agent:main',
            'cbm_population_agent_online = cbm_pop.cbm_population_agent_online:main',
            'cbm_population_agent_online_simple_simulation = cbm_pop.SimpleSimulator.cbm_population_agent_online_simple_simulation:main',
            'cbm_population_agent_online_simple_simulation_lock = cbm_pop.SimpleSimulator.cbm_population_agent_online_simple_simulation_lock:main',
            'cbm_population_agent_online_simple_simulation_nash = cbm_pop.SimpleSimulator.cbm_population_agent_online_simple_simulation_nash:main',
            'cbm_population_agent_online_simple_simulation_offline = cbm_pop.SimpleSimulator.cbm_population_agent_online_simple_simulation_offline:main',
            'simple_simulator = cbm_pop.SimpleSimulator.simple_simulator:main',
            'simulator_robot = cbm_pop.SimpleSimulator.simulator_robot:main',
            'cbm_population_agent_offline = cbm_pop.cbm_population_agent_offline:main',
            'kill_robot_at_time = cbm_pop.kill_robot_at_time:main',
            'decentralised_tracker = cbm_pop.decentralised_tracker:main',
        ],
    },
)
