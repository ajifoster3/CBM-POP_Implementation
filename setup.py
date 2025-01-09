from setuptools import find_packages, setup

import cbm_pop

package_name = 'cbm_pop'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
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
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cbm_population_agent = cbm_pop.cbm_population_agent:main',
            'cbm_fitness_logger = cbm_pop.fitness_logger:main',
        ],
    },
)
